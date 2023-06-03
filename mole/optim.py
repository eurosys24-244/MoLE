from .monitor import Moniter
from .opt_helper import get_helper, get_movable_states, get_other_states
from .memory import get_memory_allocator
import torch
import time
from .utils import moe_global_counter
import mole.config as mole_config


#from DeepSpeed
# `x` is a torch.Tensor
def _has_inf_or_nan(x):
    try:
            # if x is half, the .float() incurs an additional deep copy, but it's necessary if
            # Pytorch's .sum() creates a one-element tensor of the same type as x
            # (which is true for some recent version of pytorch).
        cpu_sum = float(x.float().sum())
            # More efficient version that can be used if .sum() returns a Python scalar
            # cpu_sum = float(x.sum())
    except RuntimeError as instance:
            # We want to check if inst is actually an overflow exception.
            # RuntimeError could come from a different error.
            # If so, we still want the exception to propagate.
        if "value cannot be converted" not in instance.args[0]:
            raise
        return True
    else:
        if cpu_sum in [float('inf'), -float('inf')] or cpu_sum != cpu_sum:
            return True
        return False
    
class moleOptim():
    def __init__(self, opt,gpu,single_update=get_helper("adamw"),movable_states=get_movable_states("adamw"),other_states=get_other_states("adamw"),**kwargs):
        self.memory_allocator = get_memory_allocator()
        self.optim = opt(self.params, **kwargs)
        self.single_update = single_update
        self.gpu = gpu
        print("building: ", self.optim)
        assert len(self.optim.param_groups) == 1, "Only support one single param group"
        #one parameter group
        self.first_step = True
        
        
        self.early_stepped = [False] * len(self.params)

        self.movable_states = movable_states 
        self.other_states = other_states

        self.non_opt_states = {}


        for state_name in other_states:
            self.non_opt_states[state_name] = {}

    @property
    def collect(self):
        return self.memory_allocator._collect

    @property
    def params(self):
        return self.memory_allocator._params

    @property
    def acc_counts(self):
        return self.memory_allocator.acc_counts
    
    def set_optimizer_collect(self, all_time, param_num):
        self.memory_allocator.set_optimizer_collect(all_time, param_num)
    
    def set_fetch_collect(self, all_time, convert, param_num):
        self.memory_allocator.set_fetch_collect(all_time, convert, param_num)

    def make_policy(self):
        self.memory_allocator.make_policy()
    
    def repolicy(self, memory_budget=-1):
        self.memory_allocator.repolicy(memory_budget)

    def reassign(self):
        self.memory_allocator.reassign()
    
    def advrepolicy(self):
        self.memory_allocator.advrepolicy(moe_global_counter["reduced"])
        
    def force_grad_cpu(self, param_id): #grad cpu for check
        return self.memory_allocator.force_grad_cpu(param_id)
    
    def force_grad_cpu_convert(self, param_id):
        return self.memory_allocator.force_grad_cpu_convert(param_id)

    def profile_grad_fetch(self, param_id):
        return self.memory_allocator.profile_grad_fetch(param_id)

    def optimizer_cpu(self, param_id, opt_ids):
        return self.memory_allocator.optimizer_get_all_cpu(param_id, opt_ids)

    def optimizer_gpu(self, param_id, opt_ids):
        return self.memory_allocator.optimizer_get_all_gpu(param_id, opt_ids) 

    @property
    def first_step(self):
        return self.memory_allocator._first_step

    @first_step.setter
    def first_step(self, value):
        self.memory_allocator._first_step = value

    @property
    def second_step(self):
        return self.memory_allocator._second_step

    @second_step.setter
    def second_step(self, value):
        self.memory_allocator._second_step = value

    @torch.no_grad()
    def fake_early_step(self, fc_id, layer_id, expert_id, scale=-1, early_zero_grad=True):
        if self.first_step:
            return 
        if not Moniter.sync_this_step:
            return 
        
        param_id = self.memory_allocator.sig2id[(fc_id, layer_id, expert_id)]

        self.early_stepped[param_id] = True  #this before overflow detect because no need to release grad
        self.acc_counts[param_id] += 1

        if mole_config.SAVE_ROUTING:
            if "skip" in moe_global_counter["route"]:
                moe_global_counter["route"]["skip"].append(param_id)
            else:
                moe_global_counter["route"]["skip"] = [param_id]

        if Moniter.overflow:
            #if early_zero_grad:
            #    self.memory_allocator.zero_grad(param_id)
            #    self.early_stepped[param_id] = True
            print("Detect overflow, early step", param_id)
            return

    def need_update(self, layer_id, expert_id):
        param_id0 = self.memory_allocator.sig2id[(0, layer_id, expert_id)]
        param_id1 = self.memory_allocator.sig2id[(1, layer_id, expert_id)]
        return self.acc_counts[param_id0] > mole_config.MAX_SKIP_STEPS or self.acc_counts[param_id1] > mole_config.MAX_SKIP_STEPS 

    @torch.no_grad()
    def early_step(self, fc_id, layer_id, expert_id, scale=-1, early_zero_grad=True):
        if self.first_step:
            return 

        if not Moniter.sync_this_step:
            return 
        
        param_id = self.memory_allocator.sig2id[(fc_id, layer_id, expert_id)]
                

        if Moniter.overflow:
            if early_zero_grad:
                self.memory_allocator.zero_grad(param_id)
                self.early_stepped[param_id] = True
            print("Detect overflow, early step", param_id)
            return
        
        param_gpu, grad_gpu, opts_gpu = self.optimizer_gpu(param_id,[i for i in range(len(self.movable_states))]) 

        if _has_inf_or_nan(grad_gpu):
            Moniter.set_overflow(True)
            return

        scale = Moniter.loss_scale if scale == -1 else scale

        self.early_stepped[param_id] = True 

        if scale != 1 or self.acc_counts[param_id] != 1:
            grad_gpu.data.div_(scale * self.acc_counts[param_id])
            if self.acc_counts[param_id] != 1:
                self.acc_counts[param_id] = 1
        
        self.single_update(param=param_gpu, grad=grad_gpu,\
                           opt_state=opts_gpu,\
                            non_opt_state=[self.non_opt_states[state_name][param_id] for state_name in self.other_states],group=self.optim.param_groups[0])

        if early_zero_grad:
            self.memory_allocator.zero_grad(param_id)
            

        from deepspeed.ops.adam.cpu_adam import DeepSpeedCPUAdam
        if type(self.optim) == DeepSpeedCPUAdam:
            self.non_opt_states['step'][param_id] += 1

    @torch.no_grad()
    def _once_step(self, scale=-1):
        scale = Moniter.loss_scale if scale == -1 else scale

        group = self.optim.param_groups[0]

        for idx, p in enumerate(group['params']):
            p.grad = self.force_grad_cpu_convert(idx)  
            p.grad.data.div_(scale)
            
        self.optim.step()
        
        for p in group["params"]:
            p.grad = p.grad.pin_memory()

        self.early_stepped = [False] * len(self.params)
        group = self.optim.param_groups[0]
        
    
        for idx, p in enumerate(group["params"]):
            assert id(p) == id(self.params[idx]), "param id not match"
            state = self.optim.state[p]
            for opt_id, state_name in enumerate(self.movable_states):
                opt_tensor = state[state_name]
                self.memory_allocator.register_opt_state_idx(idx, opt_id, opt_tensor)
            for state_name in self.other_states:
                self.non_opt_states[state_name][idx] = state[state_name]

        self.zero_grad()
        self.skip_zero = True
    
    @torch.no_grad()
    def collect_step(self, scale=-1):

        scale = Moniter.loss_scale if scale == -1 else scale
 
        group = self.optim.param_groups[0]

        grads = []
        
        tconvert = time.time()
        for idx, p in enumerate(group['params']):
            p.grad = self.force_grad_cpu_convert(idx)  
            
        convert = time.time() - tconvert


        for p in group["params"]:
            p.grad = p.grad.pin_memory()

        if self.collect:
            torch.cuda.synchronize()
            before_fetch = time.time()
            for idx, p in enumerate(group['params']):
                self.profile_grad_fetch(idx)
            torch.cuda.synchronize()
            after_fetch = time.time()
            self.set_fetch_collect(after_fetch - before_fetch, convert, len(self.params))

            torch.cuda.synchronize()
            before_optim = time.time()
            for idx, p in enumerate(group['params']):
                if scale != 1:
                    p.grad.data.div_(scale)
            self.optim.step()
            torch.cuda.synchronize()
            after_optim = time.time()
            self.set_optimizer_collect(after_optim - before_optim, len(self.params))

        print("=========information: ", after_fetch - before_fetch, after_optim - before_optim, convert)

        self.zero_grad()
        self.skip_zero = True
        if self.collect:
            self.make_policy()

    def rebuild_params(self, scale):
        group = self.optim.param_groups[0]
        group["params"] = []
        self.optim.state = {}
        

        for param_id, stepped in enumerate(self.early_stepped):
            if not stepped:


                param_cpu, grad_cpu, opts_cpu = self.optimizer_cpu(param_id,[i for i in range(len(self.movable_states))])
                p = param_cpu

                if grad_cpu.dtype != torch.float32:
                    grad_cpu = grad_cpu.to(torch.float32)
                p.grad = grad_cpu

                if scale != 1 or self.acc_counts[param_id] != 1:
                    p.grad.data.div_(scale * self.acc_counts[param_id]) #scale for fp16
                    if self.acc_counts[param_id] != 1:
                        self.acc_counts[param_id] = 1

                group["params"].append(p)  
                self.optim.state[p] = {}
                for opt_id, state_name in enumerate(self.movable_states):
                    self.optim.state[p][state_name] = opts_cpu[opt_id] #self.optimizer_states_cpu(opt_id, param_id) 
                for state_name in self.other_states:
                    self.optim.state[p][state_name] = self.non_opt_states[state_name][param_id]

        print('rebuilding ', len(group["params"]), 'experts')

    @torch.no_grad()
    def step(self, scale=-1):
        self.skip_zero = False
        if not Moniter.sync_this_step:
            return 
        
        if Moniter.overflow:
            Moniter.set_overflow(False)
            return
        
        scale = Moniter.loss_scale if scale == -1 else scale
        #torch.cuda.synchronize()
        if self.first_step:
            if self.second_step:
                self._once_step(scale)
                self.second_step = False 
            else:
                self.collect_step(scale) 
                self.first_step = False

            return 
        
        self.rebuild_params(scale)

        group = self.optim.param_groups[0]
        numel = 0
        for idx, p in enumerate(group['params']):
            numel += p.numel()

        t0 = time.time()
        self.optim.step()
        t1 = time.time()

        if mole_config.PROFILE_REAL:
            moe_global_counter["real"]["cpu-update"] = t1 - t0
        print("OPT TIME: ", t1 - t0, "count: ", numel, self.optim)


    def set_lr(self, lr):
        #for Adam
        group = self.optim.param_groups[0]
        group['lr'] = lr

    def zero_grad(self, early_zero_grad=True):
        if self.skip_zero:
            return 

        if not Moniter.sync_this_step:
            return 
        if self.first_step:
            group = self.optim.param_groups[0]
            for idx, p in enumerate(group['params']):
                del p.grad 
                self.memory_allocator.zero_grad(idx)
            return 


        for idx, p in enumerate(self.params):
            del p.grad 
            if not early_zero_grad or not self.early_stepped[idx]:   
                self.memory_allocator.zero_grad(idx)
        
        self.early_stepped = [False] * len(self.params)
        
        if not self.first_step:
            self.memory_allocator.check_tensors()
    def get_grad_list(self):
        grad_list = []
        for param_id, stepped in enumerate(self.early_stepped):
            if not stepped:
                grad_list.append(self.force_grad_cpu(param_id))
        return grad_list

smart_optim = []

def build_optimizer(opt,**kwargs):
    assert Moniter.done 
    assert len(smart_optim) == 0, "only build optim once"
    smart_optim.append(moleOptim(opt,**kwargs))
    return smart_optim[0]

def get_optimizer():
    return smart_optim[0]
