import torch
import torch.distributed as dist
import time

moe_global_counter = {}
moleStreams = {}
import mole.config as mole_config
from mole.optim import get_optimizer, _has_inf_or_nan

def check_expert(slic, rank0_print=False):
    if not mole_config.CHECK_EP:
        return
    tensor_list = [torch.empty_like(slic) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensor_list, slic)
    
    if rank0_print and torch.distributed.get_rank() == 0:
        print([torch.allclose(slic, t) for t in tensor_list])
    elif not rank0_print:
        print([torch.allclose(slic, t) for t in tensor_list])

#from fmoe
def get_torch_default_comm():
    r"""
    The NCCL communicator is needed so that Fast MoE can perform customized
    communication operators in the C code. However, it is not a publicly
    available variable. Therefore, a hacking class of the `ProcessGroupNCCL`
    in Fast MoE's C code takes the `_default_pg` and tries to dig the
    communicator out from the object. As PyTorch's private interface varies from
    time to time, different hacking techniques are tried one-by-one to be
    compatible with various versions of PyTorch.
    """
    try:
        comm = dist.distributed_c10d._get_default_group()
        return comm
    except Exception as _:
        pass
    try:
        comm = dist.distributed_c10d._default_pg
        if comm is not None:
            return comm
    except Exception as _:
        pass
    raise RuntimeError("Unsupported PyTorch version")



def mole_setoverflow(overflow):
    if not mole_config.MOLE_ACTIVATED:
        return False
    from .monitor import Moniter
    return Moniter.set_overflow(overflow)

def mole_set_global(sync_this_step, loss_scale):
    if not mole_config.MOLE_ACTIVATED:
        return
    from .monitor import Moniter
    Moniter.sync_this_step = sync_this_step
    Moniter.loss_scale = loss_scale


    
def mole_checkoverflow():
    if not mole_config.MOLE_ACTIVATED:
        return False
    ##################slow because p on CPU!
    from .monitor import Moniter
    if Moniter.overflow:
        print("old overflow")
        return True
    grad_list = get_optimizer().get_grad_list()
    for g in grad_list:
        if _has_inf_or_nan(g.data):
            print("this overflow", torch.distributed.get_rank())
            return True
    return False


def create_PriorityRouter(global_experts_num, idx, topk, local_rank, world_size):
    from .scheduler import molePrioritySchedulers, PriorityScheduler
    assert idx == len(molePrioritySchedulers)
    molePrioritySchedulers.append(PriorityScheduler(global_experts_num, idx, topk, local_rank, world_size))

def get_free_memory():
    from .memory import get_memory_allocator
    return get_memory_allocator().get_free_memory()

memory_calculator = [0, 0]

def before_step_memory():
    from .memory import get_memory_allocator
    memory_calculator[0] = get_memory_allocator().get_allocated_memory()
    

def after_step_memory(use_zero, model_dim, moe_dim, total_memory_waste_rate=0.1):
    from .memory import get_memory_allocator
    expert_size = model_dim * moe_dim
    total = get_memory_allocator().get_total_memory() 
    total = 15000000000 
    print("current budget: ",total, get_memory_allocator().max_memory_allocated(), max( total * (1 - total_memory_waste_rate) - get_memory_allocator().max_memory_allocated(), 0) / expert_size)
    if get_optimizer().first_step:
        activation_memory = memory_calculator[1] - memory_calculator[0]
        all_model_memory = get_memory_allocator().get_allocated_memory()
        
        print("peak memory pred: ", activation_memory + all_model_memory)
        
        print("budget: ",total, get_memory_allocator().max_memory_allocated(), max( total * (1 - total_memory_waste_rate) - get_memory_allocator().max_memory_allocated(), 0) / expert_size)
        if use_zero:
            get_memory_allocator().set_budget(max( total * (1 - total_memory_waste_rate) - get_memory_allocator().max_memory_allocated(), 0), expert_size)
        else:
            get_memory_allocator().set_budget(max( total * (1 - total_memory_waste_rate) - activation_memory - all_model_memory, 0), expert_size)
    else:
        get_memory_allocator().set_budget(max( total * (1 - total_memory_waste_rate) - get_memory_allocator().max_memory_allocated(), 0) , expert_size)
        

def measure_peak_memory():
    from .memory import get_memory_allocator
    memory_calculator[1] = get_memory_allocator().get_allocated_memory()

def mole_syncoverflow():
    return mole_config.SYNC_OVERFLOW


def comm_stream():
    if "c" in moleStreams:
        return moleStreams["c"]
    moleStreams["c"] = torch.cuda.Stream()
    return moleStreams["c"]

def comp_stream():
    if "p" in moleStreams:
        return moleStreams["p"]
    moleStreams["p"] = torch.cuda.Stream()
    return moleStreams["p"]

timer_dict_cpu = {}
timer_dict_cuda = {}

def update_ds_timer(timer_saver):
    if not mole_config.DEEPSPEED_PROFILE:
        return
    moe_global_counter["ds_log_timer"].update(timer_saver)

def timer_clear():
    if not mole_config.DEEPSPEED_PROFILE:
        return
    timer_dict_cpu.clear()
    timer_dict_cuda.clear()

def timer_get_dict():
    if not mole_config.DEEPSPEED_PROFILE:
        return
    torch.cuda.synchronize()
    ret = {}
    for k, v in timer_dict_cpu.items():
        if k not in ret:
            ret[k] = 0
        for start, end in v:
            ret[k] += (end - start) * 1000
    for k, v in timer_dict_cuda.items():
        if k not in ret:
            ret[k] = 0
        for start, end in v:
            ret[k] += start.elapsed_time(end) 
    moe_global_counter["ds_log_timer"].update(ret)
    return ret

def timer_cuda_start(name):
    if not mole_config.DEEPSPEED_PROFILE:
        return
    if name in timer_dict_cuda:
        timer_dict_cuda[name].append([torch.cuda.Event(enable_timing=True)])
    else:
        timer_dict_cuda[name] = [[torch.cuda.Event(enable_timing=True)]]
    timer_dict_cuda[name][-1][0].record(torch.cuda.current_stream())

def timer_cuda_end(name):
    if not mole_config.DEEPSPEED_PROFILE:
        return
    timer_dict_cuda[name][-1].append(torch.cuda.Event(enable_timing=True))
    timer_dict_cuda[name][-1][1].record(torch.cuda.current_stream())


def timer_cpu_start(name):
    if not mole_config.DEEPSPEED_PROFILE:
        return
    if name in timer_dict_cpu:
        timer_dict_cpu[name].append([time.time()])
    else:
        timer_dict_cpu[name] = [[time.time()]]

def timer_cpu_end(name):
    if not mole_config.DEEPSPEED_PROFILE:
        return 
    timer_dict_cpu[name][-1].append(time.time())
