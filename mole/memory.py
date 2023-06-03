import torch
import random
import mole.config as mole_config
import cvxopt
from cvxopt.glpk import ilp
from cvxopt import matrix
import numpy as np
import time
from mole.utils import moe_global_counter
cpu_map = {}
def tensor_to_cpu(tensor, idx, key, tag=None):
    non_blocking= False
    if mole_config.cache_cpu_tensors:
        if key in cpu_map:
            mm = cpu_map[key]
        else:
            cpu_map[key] = {}
            mm = cpu_map[key]
        if idx in mm:
            ret = mm[idx]
        else:
            ret = torch.empty_like(tensor, device="cpu", pin_memory=True)
            mm[idx] = ret
    else:
        ret = torch.empty_like(tensor, device="cpu", pin_memory=True)
    t0 = time.time()
    ret.copy_(tensor, non_blocking=non_blocking)
    t1 = time.time()
    if mole_config.PROFILE_REAL and tag != "fp32-to-fp16":
        moe_global_counter["real"]["tran"] += t1 - t0
    if tag == "fp32-to-fp16":
        moe_global_counter["real"]["convert"] += t1 - t0
    return ret 

def tensor_to_gpu(tensor, gpu, tag=None):
    #print("to gpu--",tag)
    t0 = time.time()
    ret = tensor.to(gpu, non_blocking=False)
    t1 = time.time()
    if mole_config.PROFILE_REAL:
        moe_global_counter["real"]["tran"] += t1 - t0
    return ret 

class MoEMemoryAllocator():
    def __init__(self, fc_size, device, opts) -> None:
        print("Initing MoEMemoryAllocator...")
        self.fc_size = fc_size
        self.gpu = device 
        print("----------------", self.gpu)

        self._params = []
        self._grads = []

        self._fp16_params = []
        self._fp16_grads = []
        self._optimizer_states = {}
        self.importance = []
        self.acc_counts = []
        self.opts = opts

        for o in range(opts):
            self._optimizer_states[o] = []

        self.sig2id = {}
        self.id2sig = {}

        self._first_step = True
        self._second_step = True

        self.free_memory = torch.cuda.get_device_properties(self.gpu).total_memory - torch.cuda.memory_allocated(self.gpu) #please not use all free_memory

        self.free_fp32_size = 100

        self._tensor_policy = []

        self._collect = False

        self.remain_list = []


    def get_free_memory(self):
        return torch.cuda.get_device_properties(self.gpu).total_memory - torch.cuda.memory_allocated(self.gpu)
    
    def max_memory_allocated(self):
        return torch.cuda.max_memory_allocated(self.gpu)
    
    def get_total_memory(self):
        return torch.cuda.get_device_properties(self.gpu).total_memory

    def get_allocated_memory(self):
        return torch.cuda.memory_allocated(self.gpu)
    
    def record_before_activation(self):
        self.before_act = self.get_free_memory()
    
    def record_after_activation(self, token_count):
        self.act_avg = (self.get_free_memory() - self.before_act) / token_count


    def init_fc(self, fc_id, layer_id, expert_id, dev):
        self.sig2id[(fc_id, layer_id, expert_id)] = len(self._params)
        self.id2sig[len(self._params)] = (fc_id, layer_id, expert_id)

        if dev == "cpu":
            ret = torch.zeros(self.fc_size[fc_id], device=dev, dtype=torch.float32, pin_memory=True)
        else:
            ret = torch.zeros(self.fc_size[fc_id], device=dev, dtype=torch.float32)

        self._params.append(ret)
        self._fp16_params.append(None)
        self._grads.append(None)
        self._fp16_grads.append(None)
        self._tensor_policy.append(None)
        for v in self._optimizer_states.values():
            v.append(None)
        self.importance.append(0)
        self.remain_list.append(None)


        self.acc_counts.append(1)
        return ret 

    def print_grads_list(self):
        print([1 if g is not None else 0 for g in self._grads])

    def register_opt_state_idx(self, idx, opt_id, tensor):
        assert tensor.dtype == torch.float32, "opt state should be float32!"
        self._optimizer_states[opt_id][idx] = tensor
    
    def backward_set_grad(self, fc_id, layer_id, expert_id, tensor):
        idx = self.sig2id[(fc_id, layer_id, expert_id)]
        if tensor.dtype == torch.float16:
            if self._first_step:
                if self._fp16_grads[idx] is None:
                    self._fp16_grads[idx] = tensor_to_cpu(tensor, idx, 'g16')
                else:
                    self._fp16_grads[idx] += tensor_to_cpu(tensor, idx, 'g16')
            else:
                if self._fp16_grads[idx] is None:
                    self._fp16_grads[idx] = tensor
                else:
                    self._fp16_grads[idx] += tensor
            return 
        
        if self._first_step:
            if self._grads[idx] is None:
                self._grads[idx] = tensor_to_cpu(tensor, idx, 'g')
            else:
                self._grads[idx] += tensor_to_cpu(tensor, idx, 'g')
        else:
            if self._grads[idx] is None:
                self._grads[idx] = tensor
            else:
                self._grads[idx] += tensor

    def zero_grad(self, idx):
        assert self._grads[idx] is not None or self._fp16_grads[idx] is not None, "grad is None!"
        self._grads[idx] = None 
        self._fp16_grads[idx] = None
    
    def forward_get_param(self, fc_id, layer_id, expert_id, dtype): #on gpu
        idx = self.sig2id[(fc_id, layer_id, expert_id)]
        
        if str(self._params[idx].device) != "cpu" and not self._first_step:
            self._params[idx] = tensor_to_gpu(self._params[idx], self.gpu) #, non_blocking=False)
            ret = self._params[idx]
        else:
            ret = tensor_to_gpu(self._params[idx], self.gpu) #, non_blocking=False)
        if dtype == torch.float32:
            return ret
        
        return ret.half()
    

    def backward_get_param(self, fc_id, layer_id, expert_id, dtype): #on gpu
        idx = self.sig2id[(fc_id, layer_id, expert_id)]
        
        if str(self._params[idx].device) != "cpu" and not self._first_step:
            self._params[idx] = tensor_to_gpu(self._params[idx], self.gpu) #, non_blocking=False)
            ret = self._params[idx]
        else:
            ret = tensor_to_gpu(self._params[idx], self.gpu) #, non_blocking=False)
        if dtype == torch.float32:
            return ret
        
        return ret.half()

    def optimizer_get_all_cpu(self, param_id, opt_ids): #on cpu, dtype float32
        if str(self._params[param_id].device) != "cpu":
            self._params[param_id] = tensor_to_cpu(self._params[param_id], param_id, 'p')

        if self._grads[param_id] is None:
            self._grads[param_id] = self._fp16_grads[param_id].float()
            self._fp16_grads[param_id] = None 

        if str(self._grads[param_id].device) != "cpu":
            self._grads[param_id] = tensor_to_cpu(self._grads[param_id], param_id, 'g')

        opts = []
        for opt_id in opt_ids:
            if str(self._optimizer_states[opt_id][param_id].device) != "cpu":
                self._optimizer_states[opt_id][param_id] = tensor_to_cpu(self._optimizer_states[opt_id][param_id], (opt_id, param_id), 'o')

            opts.append(self._optimizer_states[opt_id][param_id])

        return self._params[param_id], self._grads[param_id], opts #list

    def optimizer_get_all_gpu(self, param_id, opt_ids): #on cpu, dtype float32
        if str(self._params[param_id].device) == "cpu":
            self._params[param_id] = tensor_to_gpu(self._params[param_id], self.gpu) #, non_blocking=False)

        if self._grads[param_id] is None:
            self._grads[param_id] = self._fp16_grads[param_id].float()
            self._fp16_grads[param_id] = None 
    
        if str(self._grads[param_id].device) == "cpu":
            self._grads[param_id] = tensor_to_gpu(self._grads[param_id], self.gpu) #, non_blocking=False)

        opts = []

        for opt_id in opt_ids:
            if str(self._optimizer_states[opt_id][param_id].device) == "cpu":
                self._optimizer_states[opt_id][param_id] = tensor_to_gpu(self._optimizer_states[opt_id][param_id], self.gpu) #, non_blocking=False)
            opts.append(self._optimizer_states[opt_id][param_id])
        return self._params[param_id], self._grads[param_id], opts

    def force_grad_cpu(self, param_id): #get grads on cpu force
        if self._grads[param_id] is None:
            self._grads[param_id] = self._fp16_grads[param_id].float()
            self._fp16_grads[param_id] = None 
        if str(self._grads[param_id].device) != "cpu":
            self._grads[param_id] = tensor_to_cpu(self._grads[param_id], param_id, 'g')
        return self._grads[param_id]
    
    def sync_after_early_update(self, fcid, layer_id, expert_id):
        pass 

    def early_update(self, fcid, layer_id, expert_id):
        return random.random() > 0.5

#enmulate deepspeed
class CPUMoEMemoryAllocator(MoEMemoryAllocator):
    def __init__(self, fc_size, device, opts) -> None:
        super().__init__(fc_size, device, opts)



    def backward_set_grad(self, fc_id, layer_id, expert_id, tensor):
        idx = self.sig2id[(fc_id, layer_id, expert_id)]
        if tensor.dtype == torch.float16:
            if self._fp16_grads[idx] is None: #fp16 on gpu
                self._fp16_grads[idx] = tensor
            else:
                self._fp16_grads[idx] += tensor
            return 
        
        if self._grads[idx] is None:
            self._grads[idx] = tensor_to_cpu(tensor, idx, 'g')
        else:               
            self._grads[idx] += tensor_to_cpu(tensor, idx, 'g')

    def zero_grad(self, idx):
        assert self._grads[idx] is not None or self._fp16_grads[idx] is not None, "grad is None!"
        self._grads[idx] = None 
        self._fp16_grads[idx] = None
    
    #_first_step must put the param on cpu
    def forward_get_param(self, fc_id, layer_id, expert_id, dtype): #on gpu
        idx = self.sig2id[(fc_id, layer_id, expert_id)]
        
        ret = tensor_to_gpu(self._params[idx], self.gpu) #, non_blocking=False)

        if dtype == torch.float32:
            return ret
        
        return ret.half()
    
    #_first_step must put the param on cpu
    def backward_get_param(self, fc_id, layer_id, expert_id, dtype): #on gpu
        idx = self.sig2id[(fc_id, layer_id, expert_id)]
        
        ret = tensor_to_gpu(self._params[idx], self.gpu) #, non_blocking=False)

        if dtype == torch.float32:
            return ret
        
        return ret.half()

    def optimizer_get_all_cpu(self, param_id, opt_ids): #on cpu, dtype float32
        if self._grads[param_id] is None:
            self._grads[param_id] = self._fp16_grads[param_id].float()
            self._fp16_grads[param_id] = None 

        if str(self._grads[param_id].device) != "cpu":
            self._grads[param_id] = tensor_to_cpu(self._grads[param_id], param_id, 'g')

        opts = []
        for opt_id in opt_ids:
            opts.append(self._optimizer_states[opt_id][param_id])

        return self._params[param_id], self._grads[param_id], opts #list

    def optimizer_get_all_gpu(self, param_id, opt_ids): #on cpu, dtype float32
        assert False, "not happen"
        return self._params[param_id], self._grads[param_id], opts

    def force_grad_cpu(self, param_id):
        if self._grads[param_id] is None:
            self._grads[param_id] = self._fp16_grads[param_id].float()
            self._fp16_grads[param_id] = None

        if str(self._grads[param_id].device) != "cpu":
            self._grads[param_id] = tensor_to_cpu(self._grads[param_id], param_id, 'g')

        return self._grads[param_id] #always keep on cpu
    
    def early_update(self, fcid, layer_id, expert_id):
        return False
    

class moleMemoryAllocator(MoEMemoryAllocator):
    def __init__(self, fc_size, device, opts) -> None:
        super().__init__(fc_size, device, opts)
        self.remain_dict = {1: True, 3: True,5 : False, 6:False, 7:False, 9:True}
        self.early_dict = {1:True, 3:True, 5:True, 6:False, 7:True, 8:False}

        self.cpu_param = {1: False, 2:False, 3:False, 4:True, 5:True, 6:True, 7:True, 8:True, 9:True}
        self.cpu_opt = {1: False, 2:True, 3:True, 4:False, 5:False, 6:True, 7:True, 8:True, 9:True}

        self._collect = True
        
        self._move_costs = []
        self._cpu_costs = []
        self._move_costs_val = []
        self._cpu_costs_val = []
        self.total_costs_val = []
        self.convert_cost = []
        self._gpu_mem = []

        self.remain = []
        self.early = []
        self.param_dev = []
        self.opt_dev = []

        self.useful_policy = []
        self.cost_tuple = []

        
        self.policy_made = False
        self._memory_budget = -1
        self.advcount = 0
    
    def print_policy(self):
        for policy in self.useful_policy:
            idx = policy
            print("policy: ", policy, "remain: ", self.remain[idx], "early: ", self.early[idx], "move_cost: ", self._move_costs[idx], "cpu_cost: ", self._cpu_costs[idx], "gpu_mem: ", self._gpu_mem[idx], "cost tuple: ", self.cost_tuple[idx])
    def set_policy(self, idx, p, o, early, remain, move_cost, cpu_cost, gpu_mem, convert_count):

        
        
        #policy id start from 1
        assert idx == len(self._move_costs)
        self.param_dev.append(p)
        self.opt_dev.append(o)
        self.remain.append(remain)
        self.early.append(early)
        self._move_costs.append(move_cost) #move cost: how many 
        self._cpu_costs.append(cpu_cost)
        self._gpu_mem.append(gpu_mem)
        
        self.convert_cost.append(convert_count * self.use_fp16 * self.avg_convert_time) #when cpu is very weak, it is not ignorable
        self._cpu_costs_val.append(cpu_cost * self.avg_optim_time)

        self._cpu_costs_val[-1] += self.convert_cost[-1] 
        self._move_costs_val.append(move_cost * self.avg_fetch_time / 4) #avg fetch time: time for fetch one single param32
        self.total_costs_val.append(self._cpu_costs_val[-1] + self._move_costs_val[-1])
        self.cost_tuple.append((self.total_costs_val[-1], self._gpu_mem[-1]))

        

    def set_budget(self, mem, expert_size):
        #import time
        self.memory_budget = int(mem / expert_size)
        print("current peak & set budget: ", mem, self.memory_budget, self._memory_budget)
        #return 

    @property
    def memory_budget(self):
        assert self._memory_budget != -1
        return self._memory_budget

    @memory_budget.setter
    def memory_budget(self, mem):
        self._memory_budget = mem
        

    @property
    def mem_opt(self):
        return self.opts * 4
    
    @property
    def mem_param32(self):
        return 4 
    
    @property
    def mem_param(self):
        return 2 if self.use_fp16 else 4    

    @property
    def use_fp16(self):
        return mole_config.USE_FP16

    def set_optimizer_collect(self, all_time, param_num):
        self._collect_time = all_time
        self._collect_param_num = param_num
        self.avg_optim_time = all_time / param_num
        t = torch.tensor([self.avg_optim_time]).to(self.gpu, non_blocking=False)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.AVG)
        print("optimixer avg time: ", self.avg_optim_time, t.item())
        self.avg_optim_time = t.item()
        
    def set_fetch_collect(self, all_time, convert_time, param_num):
        self._fetch_time = all_time
        self._fetch_param_num = param_num
        self.avg_fetch_time = all_time / param_num
        
        t = torch.tensor([self.avg_fetch_time]).to(self.gpu, non_blocking=False)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX) #need to wait for the max one
        
        print("fetch avg time: ", self.avg_fetch_time, t.item())
        self.avg_fetch_time = t.item()


        self.avg_convert_time = convert_time / param_num
        t = torch.tensor([self.avg_convert_time]).to(self.gpu, non_blocking=False)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.AVG) #need to wait for the max one

        print("convert avg time: ", self.avg_convert_time, t.item())
        self.avg_convert_time = t.item()

    def make_policy(self):
        self.policy_made = True

        print("==========MADA: ", self.mem_param, self.use_fp16, mole_config.USE_FP16)
        self.set_policy(0, 'g', 'g', 1, 1, 0, 0, self.mem_opt + self.mem_param32, 0)
        self.set_policy(1, 'g', 'c', 0, 1, 2 * self.mem_param32 + self.mem_param, 1, self.mem_param32, 1)
        self.set_policy(2, 'g', 'c', 1, 1,2* self.mem_opt, 0, self.mem_param32, 0)
        self.set_policy(3, 'c', 'g', 1, 1,2* self.mem_param32, 0, self.mem_opt + self.mem_param32, 0)
        self.set_policy(4, 'c', 'g', 1, 0, self.mem_param+2*self.mem_param32, 0, self.mem_opt, 1)
        self.set_policy(5, 'c', 'c', 0, 0, 3*self.mem_param, 1, 0, 3)
        self.set_policy(6, 'c', 'c', 1, 0, self.mem_param+2*self.mem_param32+2*self.mem_opt, 0, 0, 1)
        self.set_policy(7, 'c', 'c', 0, 1, 2*self.mem_param, 1, self.mem_param, 2)
        self.set_policy(8, 'c', 'c', 1, 1, 2*self.mem_opt+self.mem_param32, 0, self.mem_param32, 0)


        useless = set()
        for i in range(len(self.cost_tuple)):
            for j in range(len(self.cost_tuple)):
                if i == j:
                    continue
                if self.cost_tuple[i][0] <= self.cost_tuple[j][0] and self.cost_tuple[i][1] <= self.cost_tuple[j][1] and j not in mole_config.POLICY:
                    useless.add(j)
        
        print("POL: ", mole_config.POLICY)
        for i in range(0, 9):
            if i not in mole_config.POLICY:
                useless.add(i)

        for i in range(0, 9):
            if i not in useless:
                self.useful_policy.append(i)

        assert 1 in useless and 3 in useless and 8 in useless, "policy 1, 3, 8 should be useless"
        self.useful_policy = sorted(self.useful_policy, key=lambda x: self.cost_tuple[x][0])
        #cost from smallest to largest
        if torch.distributed.get_rank() == 0:
            self.print_policy()
        self.solve()
        self.assign_policy()
        torch.cuda.empty_cache()
        self.reset_tensors()
        torch.cuda.empty_cache()

        
    def advrepolicy(self, reduced_params):
        self.memory_budget += self.mem_usage
        self.advresolve(reduced_params)
        self.advassign_policy()
        torch.cuda.empty_cache()
        self.reset_tensors()
        torch.cuda.empty_cache()
        
    def reassign(self):
        self.assign_policy()
        torch.cuda.empty_cache()
        self.reset_tensors()
        torch.cuda.empty_cache()

    def repolicy(self, memory_budget=-1):
        if not self.policy_made:
            return
        if memory_budget > 0:
            self.memory_budget = memory_budget
        self.memory_budget += self.mem_usage
        print("budget: ", self.fixed_memory_budget, self.memory_budget)
        self.solve()
        self.assign_policy()
        torch.cuda.empty_cache()
        self.reset_tensors()
        torch.cuda.empty_cache()

    def advresolve(self, reduced_params):
        clst = []
        for i in self.useful_policy:
            clst.append(self.cost_tuple[i][0])
            clst.append(self.cost_tuple[i][0] / mole_config.MAX_SKIP_STEPS)
        c = matrix(clst)
        Gm = [[]]
        gmlst = []
        for i in self.useful_policy:
            gmlst.append(self.cost_tuple[i][1])
            gmlst.append(self.cost_tuple[i][1])
        Gm[0] = gmlst

        for i in range(0, len(self.useful_policy) * 2):
            tmp = [0.0] * (len(self.useful_policy) * 2)
            tmp[i] = -1.0
            Gm.append(tmp)

        G = matrix(np.array(Gm)) #
        h = matrix([self.fixed_memory_budget] + [0.0] *  (2 * len(self.useful_policy)))
        
        As = []
        As.append([1.0] * ( 2 * len(self.useful_policy)))
        tmp = [0.0] * len(self.useful_policy) * 2
        for i in range(len(self.useful_policy)):
            tmp[i * 2 + 1] = 1.0
        As.append(tmp)
        A = matrix(np.array(As))
        b = matrix([float(len(self._params)), float(reduced_params)])
        
        if torch.distributed.get_rank() == 0:
            print("INFO: ", self.useful_policy, self.fixed_memory_budget)
            print("Matrix Setting: ", "C: " ,c, "G:", G,"h: "  ,h," A: ", A,"b:", b)

        time0 = time.time()
        (status, count_each_policy_with_zero) = ilp(c=c,G=G,h=h,I=set(range(len(self.useful_policy))),A=A,b=b)
        time1 = time.time()
        
        self.count_each_policy_called = [int(count_each_policy_with_zero[i * 2] + 0.5) for i in range(len(count_each_policy_with_zero) // 2)] 
        self.count_each_policy_non_called = [int(count_each_policy_with_zero[i * 2 + 1] + 0.5) for i in range(len(count_each_policy_with_zero) // 2)] 

        mem_usage = 0
        latency = 0
        for i in range(len(self.useful_policy)):
            policy = self.useful_policy[i]
            latency += self.count_each_policy_called[i] * self.cost_tuple[policy][0] + self.count_each_policy_non_called[i] * self.cost_tuple[policy][0] / mole_config.MAX_SKIP_STEPS
            mem_usage += self.count_each_policy_called[i] * self.cost_tuple[policy][1] + self.count_each_policy_non_called[i] * self.cost_tuple[policy][1] 
            
        if torch.distributed.get_rank() == 0:
            print("Solver output: ", self.count_each_policy_called, self.count_each_policy_non_called , self.useful_policy, " taking ", time1 - time0)
            import os 
            pathx = mole_config.output_dir
            est_cpu = 0
            est_tran = 0
            for i, ct in enumerate(self.count_each_policy_called):
                po = self.useful_policy[i]
                est_cpu += self._cpu_costs_val[po] * ct
                est_tran += self._move_costs_val[po] * ct
            for i, ct in enumerate(self.count_each_policy_non_called):
                po = self.useful_policy[i]
                est_cpu += self._cpu_costs_val[po] * ct / mole_config.MAX_SKIP_STEPS
                est_tran += self._move_costs_val[po] * ct / mole_config.MAX_SKIP_STEPS
            torch.save([mem_usage, latency, self.avg_optim_time, self.avg_fetch_time,est_cpu,est_tran, self.memory_budget, self._cpu_costs_val,self._move_costs_val, self.count_each_policy_called, self.count_each_policy_non_called, self.useful_policy, self.cost_tuple, self.avg_convert_time], pathx+f"/a{self.advcount}save.pt")
        else:
            import os 
            pathx = mole_config.output_dir
            est_cpu = 0
            est_tran = 0
            for i, ct in enumerate(self.count_each_policy_called):
                po = self.useful_policy[i]
                est_cpu += self._cpu_costs_val[po] * ct
                est_tran += self._move_costs_val[po] * ct
            for i, ct in enumerate(self.count_each_policy_non_called):
                po = self.useful_policy[i]
                est_cpu += self._cpu_costs_val[po] * ct / mole_config.MAX_SKIP_STEPS
                est_tran += self._move_costs_val[po] * ct / mole_config.MAX_SKIP_STEPS
            torch.save([mem_usage, latency, self.avg_optim_time, self.avg_fetch_time,est_cpu,est_tran, self.memory_budget, self._cpu_costs_val,self._move_costs_val, self.count_each_policy_called, self.count_each_policy_non_called, self.useful_policy, self.cost_tuple, self.avg_convert_time], pathx+f"/a{self.advcount}save_{torch.distributed.get_rank()}.pt")

        self.advcount += 1

        self.mem_usage = mem_usage
        

    def advassign_policy(self):

        
        move = 0
        importance_ids = [i for i in range(len(self._params))]
        importance_ids = sorted(importance_ids, key=lambda x: self.importance[x], reverse=True)
        #importance from largest to smallest
        for idx, count in enumerate(self.count_each_policy_called):
            for _ in range(count):
                self._tensor_policy[importance_ids[move]] = self.useful_policy[idx]
                move += 1

        for idx, count in enumerate(self.count_each_policy_non_called):
            for _ in range(count):
                self._tensor_policy[importance_ids[move]] = self.useful_policy[idx]
                move += 1
        
        self.all_policies = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(self.all_policies, self._tensor_policy)

        print("finalized policy: ", self.memory_budget, self._tensor_policy, self.importance, sum(self.count_each_policy_called), sum(self.count_each_policy_non_called))


    def solve(self):


        c = matrix([self.cost_tuple[i][0] for i in self.useful_policy])

        #<= mem budget, and count>=0
        Gm = [[]]
        Gm[0] = [self.cost_tuple[i][1] for i in self.useful_policy]
        for i in range(0, len(self.useful_policy)):
            tmp = [0.0] * (len(self.useful_policy))
            tmp[i] = -1.0
            Gm.append(tmp)
        
        self.fixed_memory_budget = self.memory_budget
        G = matrix(np.array(Gm)) #
        h = matrix([self.memory_budget] + [0.0] * len(self.useful_policy))
        A = matrix(np.array([[1.0] * len(self.useful_policy)]))
        b = matrix([float(len(self._params))])
        
        if torch.distributed.get_rank() == 0:
            print("INFO: ", self.useful_policy, self.memory_budget)
            print("Matrix Setting: ", "C: " ,c, "G:", G,"h: "  ,h," A: ", A,"b:", b)

        time0 = time.time()
        (status, count_each_policy) = ilp(c=c,G=G,h=h,I=set(range(len(self.useful_policy))),A=A,b=b)
        time1 = time.time()
        
        if torch.distributed.get_rank() == 0:
            print("Pol: ", self.useful_policy, count_each_policy, self.memory_budget)
        self.count_each_policy = [int(count_each_policy[i] + 0.5) for i in range(len(self.useful_policy))]

        mem_usage = 0
        latency = 0
        for i in range(len(self.useful_policy)):
            policy = self.useful_policy[i]
            latency += count_each_policy[i] * self.cost_tuple[policy][0] 
            mem_usage += count_each_policy[i] * self.cost_tuple[policy][1] 
        
        print("==============================Solution===========================")
        print("Time and Mem: ", self.avg_optim_time, self.avg_fetch_time, self.memory_budget)
        print("Cost: ", self.cost_tuple)
        print("Policy: ", self.useful_policy, count_each_policy)
        print("Mem usage: ", mem_usage, " Latency: ", latency )
        print("==============================Solution===========================")
        
        moe_global_counter["sol"]["mem"] = mem_usage
        moe_global_counter["sol"]["lat"] = latency
        self.mem_usage = mem_usage
        if torch.distributed.get_rank() == 0:
            print("Solver output: ", self.count_each_policy, self.useful_policy, " taking ", time1 - time0)
            import os 
            pathx = mole_config.output_dir
            est_cpu = 0
            est_tran = 0
            for i, ct in enumerate(self.count_each_policy):
                po = self.useful_policy[i]
                est_cpu += self._cpu_costs_val[po] * ct
                est_tran += self._move_costs_val[po] * ct
            torch.save([mem_usage, latency, self.avg_optim_time, self.avg_fetch_time,est_cpu,est_tran, self.memory_budget, self._cpu_costs_val,self._move_costs_val, self.count_each_policy, self.useful_policy, self.cost_tuple, self.avg_convert_time], pathx+"/save.pt")
        else:
            import os 
            pathx = mole_config.output_dir
            est_cpu = 0
            est_tran = 0
            for i, ct in enumerate(self.count_each_policy):
                po = self.useful_policy[i]
                est_cpu += self._cpu_costs_val[po] * ct
                est_tran += self._move_costs_val[po] * ct
            torch.save([mem_usage, latency, self.avg_optim_time, self.avg_fetch_time,est_cpu,est_tran, self.memory_budget, self._cpu_costs_val,self._move_costs_val, self.count_each_policy, self.useful_policy, self.cost_tuple, self.avg_convert_time], pathx+f"/save_{torch.distributed.get_rank()}.pt")


    def assign_policy(self):
        move = 0
        importance_ids = [i for i in range(len(self._params))]
        importance_ids = sorted(importance_ids, key=lambda x: self.importance[x], reverse=True)

        for idx, count in enumerate(self.count_each_policy):
            for _ in range(count):
                self._tensor_policy[importance_ids[move]] = self.useful_policy[idx]
                move += 1
        
        self.all_policies = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(self.all_policies, self._tensor_policy)

        print("finalized policy: ", self.memory_budget, self._tensor_policy, self.importance, self.all_policies)


    def reset_tensors(self):
        print("enter reset")
        torch.cuda.synchronize()
        before_reset = self.get_allocated_memory()
        for idx in range(len(self._params)):
            policy = self._tensor_policy[idx]
            if self.param_dev[policy] != "c":
                if str(self._params[idx].device) == "cpu":
                    self._params[idx] = tensor_to_gpu(self._params[idx], self.gpu) #, non_blocking=False)

            if self.opt_dev[policy] != "c":
                for opt_id in range(self.opts):
                    if str(self._optimizer_states[opt_id][idx].device) == "cpu":
                        self._optimizer_states[opt_id][idx] = tensor_to_gpu(self._optimizer_states[opt_id][idx], self.gpu) #, non_blocking=False)

        for idx in range(len(self._params)):
            policy = self._tensor_policy[idx]
            if self.param_dev[policy] == "c":
                if str(self._params[idx].device) != "cpu":
                    self._params[idx] = tensor_to_cpu(self._params[idx], idx, 'p')

            if self.opt_dev[policy] == "c":
                for opt_id in range(self.opts):
                    if str(self._optimizer_states[opt_id][idx].device) != "cpu":
                        self._optimizer_states[opt_id][idx] = tensor_to_cpu(self._optimizer_states[opt_id][idx], (opt_id, idx), 'o')
        after_reset = self.get_allocated_memory()
        print("memory reset: ", after_reset - before_reset)
        torch.cuda.synchronize()


    def check_tensors(self):
        for idx in range(len(self._params)):
            policy = self._tensor_policy[idx]
            assert (self.param_dev[policy] == "c" and str(self._params[idx].device) == "cpu") or \
                (self.param_dev[policy] == "g" and str(self._params[idx].device) != "cpu"), ""
            if self.opt_dev[policy] == "c":
                for opt_id in range(self.opts):
                    assert str(self._optimizer_states[opt_id][idx].device) == "cpu", ""
            else:
                for opt_id in range(self.opts):
                    assert str(self._optimizer_states[opt_id][idx].device) != "cpu", ""
            assert self._fp16_params[idx] is None and self._grads[idx] is None and self._fp16_grads[idx] is None 


    def backward_set_grad(self, fc_id, layer_id, expert_id, tensor):
        idx = self.sig2id[(fc_id, layer_id, expert_id)]
        policy = self._tensor_policy[idx]


        if self._first_step:        #?

            self._grads[idx] = tensor_to_cpu(tensor, idx, 'g') 

            return 
        
        if self.early[policy]:
            self._grads[idx] = tensor.to(torch.float32)
        elif policy == 5 or policy == 7:
            tmp = torch.empty_like(tensor, device="cpu", pin_memory=True)

            t0 = time.time()
            tmp.copy_(tensor, non_blocking=False)
            t1 = time.time()
            if mole_config.PROFILE_REAL:
                moe_global_counter["real"]["tran"] += t1 - t0

            self._grads[idx] = tmp 
        else:
            assert False, "non support"
            self._grads[idx] = tensor_to_cpu(tensor.to(torch.float32), idx, 'g')


    def zero_grad(self, idx):
        assert self._grads[idx] is not None or self._fp16_grads[idx] is not None, "grad is None!"
        self._grads[idx] = None 
        self._fp16_grads[idx] = None
    
    def sync_after_early_update(self, fc_id, layer_id, expert_id):

        idx = self.sig2id[(fc_id, layer_id, expert_id)]
        policy = self._tensor_policy[idx]
        
        if self.param_dev[policy] == "c":
            if self.remain[policy]:
                self._params[idx] = tensor_to_cpu(self.remain_list[idx], idx, 'p')
                self.remain_list[idx] = None 
            else:
                
                self._params[idx] = tensor_to_cpu(self._params[idx], idx, 'p')
        if self.opt_dev[policy] == "c":
            for opt_id in range(self.opts):
                #pass
                self._optimizer_states[opt_id][idx] = tensor_to_cpu(self._optimizer_states[opt_id][idx], (opt_id, idx), 'o')


    def tmp_get_param(self, fc_id, layer_id, expert_id):
        idx = self.sig2id[(fc_id, layer_id, expert_id)]
        return self._params[idx]
    

    def forward_get_param(self, fc_id, layer_id, expert_id, dtype): #on gpu
        idx = self.sig2id[(fc_id, layer_id, expert_id)]
        policy = self._tensor_policy[idx]

        if self._first_step:
            ret = tensor_to_gpu(self._params[idx], self.gpu) 

            if dtype == torch.float32:
                return ret 
            else:
                return ret.half()
        
        if self.param_dev[policy] == "g":
            ret = self._params[idx]
            if dtype == torch.float32:
                return ret 
            else:
                return ret.half()
        
        if policy == 7:
            self.remain_list[idx] = tensor_to_gpu(self._params[idx].to(dtype), self.gpu, "syncto") #, non_blocking=False)
            return self.remain_list[idx]
        
        if self.remain[policy]:
            self.remain_list[idx] = tensor_to_gpu(self._params[idx], self.gpu) #, non_blocking=False)
            ret = self.remain_list[idx]
            if dtype == torch.float32:
                return ret
            else:
                return ret.half()

        ret = tensor_to_gpu(self._params[idx].to(dtype), self.gpu) #, non_blocking=False)
        return ret 
    
    #_first_step must put the param on cpu
    def backward_get_param(self, fc_id, layer_id, expert_id, dtype): #on gpu
        idx = self.sig2id[(fc_id, layer_id, expert_id)]
        policy = self._tensor_policy[idx]
        
        if self._first_step:
            ret = tensor_to_gpu(self._params[idx], self.gpu)# non_blocking=False)
            if dtype == torch.float32:
                return ret 
            else:
                return ret.half()

        if self.param_dev[policy] == "g":
            ret = self._params[idx]
            if dtype == torch.float32:
                return ret 
            else:
                return ret.half()
        
        if self.remain[policy]:
            ret = self.remain_list[idx]
            if policy == 7:
                self.remain_list[idx] = None
            return ret.to(dtype)
        
        if policy == 4 or policy == 6:
            self._params[idx] = tensor_to_gpu(self._params[idx], self.gpu) #, non_blocking=False)
            ret = self._params[idx].to(dtype)
        else:
            ret = tensor_to_gpu(self._params[idx].to(dtype), self.gpu) #, non_blocking=False)
        #torch.cuda.nvtx.range_pop()
        return ret

    def optimizer_get_all_cpu(self, param_id, opt_ids): #on cpu, dtype float32
        policy = self._tensor_policy[param_id]
        assert not self.early[policy], "policy is early!"

        ret_p = None 

        ret_p = self._params[param_id]

        ret_g = self._grads[param_id]

        opts = []
        assert self.opt_dev[policy] == "c", "optimizer is not on cpu!"
        for opt_id in opt_ids:
            opts.append(self._optimizer_states[opt_id][param_id])

        return ret_p, ret_g, opts
    

    def optimizer_get_all_gpu(self, param_id, opt_ids): 
        policy = self._tensor_policy[param_id]
        assert self.early[policy], "policy is not early!"

        ret_p = None
        if self.param_dev[policy] == "g":
            ret_p = self._params[param_id]
        elif self.remain_list[policy]:
            ret_p = self.remain_list[param_id] #need transfer back after update
        else:
            ret_p = self._params[param_id]

        ret_g = self._grads[param_id]
        #gradients must on gpu

        opts = []

        if self.opt_dev[policy] == "g":
            for opt_id in opt_ids:
                opts.append(self._optimizer_states[opt_id][param_id])
        else:
            for opt_id in opt_ids:
                self._optimizer_states[opt_id][param_id] = tensor_to_gpu(self._optimizer_states[opt_id][param_id], self.gpu) #, non_blocking=False)
                opts.append(self._optimizer_states[opt_id][param_id])

        return ret_p, ret_g, opts

    def force_grad_cpu(self, param_id): 
        
        assert self._grads[param_id] is not None and self._fp16_grads[param_id] is None, "precision not set correctly"
        assert str(self._grads[param_id].device) == "cpu", "grads are not on cpu"

        return self._grads[param_id]

    def force_grad_cpu_convert(self, param_id):
        
        assert self._grads[param_id] is not None and self._fp16_grads[param_id] is None, "precision not set correctly"
        assert str(self._grads[param_id].device) == "cpu", "grads are not on cpu"
        if self._grads[param_id].dtype != torch.float32:
            #print("non fp32")
            self._grads[param_id] = self._grads[param_id].to(torch.float32)
        self._grads[param_id] += 0
        return self._grads[param_id]


    def profile_grad_fetch(self, param_id):
        assert str(self._grads[param_id].device) == "cpu" and self._fp16_grads[param_id] is None, "grads are not on cpu"
        tmp = tensor_to_gpu(self._grads[param_id], self.gpu) 
        del tmp

    def early_update(self, fc_id, layer_id, expert_id):
        if self._first_step:
            return False
        idx = self.sig2id[(fc_id, layer_id, expert_id)]
        return self.early[self._tensor_policy[idx]]

mem_allocator_ = []

def set_memory_allocator(name = "native", fc_size=None, device=None, opts=None):
    assert len(mem_allocator_) == 0, "memory allocator already set"
    if name == "native":
        mem_allocator_.append(MoEMemoryAllocator(fc_size, device, opts))
    elif name == "cpu":
        mem_allocator_.append(CPUMoEMemoryAllocator(fc_size, device, opts))
    elif name == "mole":
        mem_allocator_.append(moleMemoryAllocator(fc_size, device, opts))
    

def get_memory_allocator():
    return mem_allocator_[0]

