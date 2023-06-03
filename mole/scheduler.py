from enum import Enum
import torch 
import mole.config as mole_config
from mole.memory import get_memory_allocator
from mole.utils import moe_global_counter
import time
import random
import numpy as np
from mole.gini import gini_coef

class Plan(Enum):
    NONE = 0 #do nothing, just return
    EVICT = 1 #evict tensor on x
    FETCH = 2 #fetch from other device
    CONVERT = 3 #convert to other dtype

#evict (tensor_id, dev)   
#fetch (tensor_id, dev) 

class NaiveScheduler():
    def __init__(self) -> None:

        self.mapping2info = {}

    def register(self, tensor_sig, dtyte, dev):
        if tensor_sig not in self.mapping2info:
            self.mapping2info[tensor_sig] = [(dtyte, dev)]
        else:  
            self.mapping2info[tensor_sig].append((dtyte, dev))
    
    #on same device
    def fp16_for_forward(self, tensor_sig, dev):
        assert tensor_sig[3] == 0, "fp16_for_forward: tensor_sig not param"
        infos = self.mapping2info[tensor_sig]
        for idx, info in enumerate(infos):
            if dev != info[1]:
                continue
            if info[0] == torch.float32:
                self.mapping2info[tensor_sig].append((torch.float16, dev))
                return [(Plan.CONVERT, tensor_sig, torch.float16, dev)]
            else:
                assert False, "fp16_for_forward: dtype not fp32 found, check correctness"

    def drop_param_fp16(self, tensor_sig, dev):
        assert tensor_sig[3] == 0, "drop_param_fp16: tensor_sig not param"
        infos = self.mapping2info[tensor_sig]
        for idx, info in enumerate(infos):
            if dev == info[1] and torch.float16 == info[0]:
                self.mapping2info[tensor_sig].pop(idx)
                return [(Plan.EVICT, tensor_sig, torch.float16, dev)]
        
        assert False, "drop_param_fp16: dtype not fp16 found, check correctness"
    
    #new generated grad
    def regitster_grad(self, tensor_sig, dtype, dev):
        assert tensor_sig[3] == 1, "register_grad: tensor_sig not grad"
        self.register(tensor_sig, dtype, dev)
    
    def drop_grad(self, tensor_sig, dtype, dev):
        assert tensor_sig[3] == 1, "drop_grad: tensor_sig not grad"
        infos = self.mapping2info[tensor_sig]
        for idx, info in enumerate(infos):
            if dev == info[1] and dtype == info[0]:
                self.mapping2info[tensor_sig].pop(idx)
                return [(Plan.EVICT, tensor_sig, dtype, dev)]
            
        assert False, "drop_grad: dtype not fp16 found, check correctness"

    #on same device
    def fp32_for_update(self, tensor_sig, dev):
        assert tensor_sig[3] == 1, "fp32_for_update: tensor_sig not grad"
        infos = self.mapping2info[tensor_sig]
        for idx, info in enumerate(infos):
            if dev != info[1]:
                continue
            if info[0] == torch.float16:
                self.mapping2info[tensor_sig].append((torch.float32, dev))
                return [(Plan.CONVERT, tensor_sig, torch.float32, dev)]
            else:
                assert False, "fp32_for_update: dtype not fp16 found, check correctness"
    #
    #infer from tensor_sig: optimizer states never convert dtype, gradients convert from fp16 to fp32 and evict fp16, params convert from fp32 to fp16
    def get(self, tensor_sig, dtype, dev):
        assert tensor_sig in self.mapping2info, "tensor_sig not registered"
        infos = self.mapping2info[tensor_sig] #(dtype, dev)
        
        for idx, info in enumerate(infos):
            if dtype != info[0]:
                continue 
            if dev == info[1]:
                return [(Plan.NONE, tensor_sig)]
            else:
                may_return_id = idx
                may_return = [(Plan.FETCH, tensor_sig, dtype, info[1], dev), (Plan.EVICT, tensor_sig, dtype, info[1])]
                #if not found info[1] == dev, return this
        
        self.mapping2info[tensor_sig].pop(may_return_id)
        return may_return

    

class RoundRobinScheduler(NaiveScheduler):
    def __init__(self) -> None:
        super().__init__()
    
    def get(self, tensor_id, cur_dev, dev):
        return 

class PriorityScheduler():
    def __init__(self, global_experts_num, idx, topk, local_rank, world_size) -> None:
        self.global_experts_num = global_experts_num
        self.local_experts_num = global_experts_num // world_size
        self.idx = idx 
        self.topk = topk
        self.world_size = world_size
        self.local_rank = local_rank
        self.drop_P = mole_config.DROP_P
        self.select_most_popular = False
        self.random = True

        self.advanced_priority = False

        self.memory_allocator = get_memory_allocator()

    @property
    def importance(self):
        return self.memory_allocator.importance
    
    @property
    def all_policies(self):
        return self.memory_allocator.all_policies

    @property
    def acc_counts(self):
        return self.memory_allocator.acc_counts

    def set_expert_importance(self, eid, val):
        idx = self.memory_allocator.sig2id[(0, self.idx, eid)]
        self.importance[idx] = val * mole_config.MOVING_AVG + self.importance[idx] * (1 - mole_config.MOVING_AVG)
        idx = self.memory_allocator.sig2id[(1, self.idx, eid)]
        self.importance[idx] = val * mole_config.MOVING_AVG + self.importance[idx] * (1 - mole_config.MOVING_AVG)
    
    def cpu_expert(self, expert_id):
        rank = self.global_to_rank(expert_id)
        local_id = self.global_to_local(expert_id)
        
        return self.all_policies[rank][local_id] >= 3 
    
    def cpu_optimizer_states(self, expert_id):
        rank = self.global_to_rank(expert_id)
        local_id = self.global_to_local(expert_id)

        return self.all_policies[rank][local_id] == 2 
    
    def mark_accumulate(self, eid):
        self.acc_counts[eid] += 1

    def is_local(self, id):
        return id // self.local_experts_num == self.local_rank
    
    def global_to_local(self, id):
        return id % self.local_experts_num
    
    def global_to_rank(self, id):
        return id // self.local_experts_num

    def reroute(self, topk_indices):
        if mole_config.FORCE_BALANCED_ROUTING and torch.is_grad_enabled():
            #print("topk", topk_indices)
            each_count = topk_indices.numel() // self.global_experts_num
            new_idx = torch.arange(0, self.global_experts_num, dtype=topk_indices.dtype, device=topk_indices.device).unsqueeze(1).expand(-1, each_count).reshape(topk_indices.shape).contiguous()
            
            return new_idx
        

        if self.drop_P == -1 or not torch.is_grad_enabled():
            return topk_indices
        

        boundary = self.topk * topk_indices.numel() // self.global_experts_num * self.drop_P
        
        ids, counts = topk_indices.unique(sorted=True, return_counts=True)


        ids = ids.cpu().numpy()
        counts = counts.cpu().numpy()
        


        drop_list = []
        count0_list = []
        popular_list = []

        mov = 0
        count_local = 0
        for e in range(self.global_experts_num):
            if mov >= len(ids) or ids[mov] != e:
                count0_list.append(e)
                
                if self.is_local(e):
                    count_local += 1
                    self.set_expert_importance(self.global_to_local(e), 0)
                
                continue
            
            if self.is_local(e):
                self.set_expert_importance(self.global_to_local(e), counts[mov])

            if counts[mov] < boundary:
                drop_list.append(e)
                if self.is_local(e):
                    count_local += 1
            else:
                popular_list.append((e, counts[mov]))
            mov += 1
        
        if self.random:
            random.shuffle(popular_list)
        else:
            popular_list.sort(key=lambda x: x[1], reverse=self.select_most_popular)


        moe_global_counter["reducedlayer"] += count_local * 2 
        if mole_config.SAVE_ROUTING:
            moe_global_counter["route"][f"l{self.idx}"] = [ids, counts, drop_list, count0_list, popular_list]
            moe_global_counter["route"][f"p{self.idx}"] = [self.importance]

        if self.advanced_priority:
            new_drop_list = []
            for d in drop_list:
                if self.cpu_expert(d):
                    new_drop_list.append(d)
                elif self.is_local(d) and self.cpu_optimizer_states(d):
                    self.mark_accumulate(self.global_to_local(d))

            drop_list = new_drop_list


        assert self.topk == 1
        for k in range(self.topk): 
            cond = None 
            for d in drop_list:
                if cond is None:
                    cond = (topk_indices[:,k] == d)
                else:
                    cond = cond.logical_or(topk_indices[:,k] == d)
            if cond is not None:
                topk_indices[:,k] = torch.where(cond, popular_list[k][0], topk_indices[:,k])

        return topk_indices

molePrioritySchedulers = []