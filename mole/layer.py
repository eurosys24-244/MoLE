import torch
import tutel
from .functions import ensure_comm
from .functions import MOEScatter, MOEGather, MOEScatterShard, MOEGatherShard, mole_global_scatter, mole_global_gather, mole_global_scatter_all, mole_global_gather_all
from .gate import TutelGate
import torch.nn.functional as F
from .monitor import Moniter
from .optim import get_optimizer
from .memory import get_memory_allocator
from .scheduler import molePrioritySchedulers
from .functions import py_global_scatter, py_global_gather
import mole.config as moleConfig
import random
import time
import logging
from .utils import comm_stream, comp_stream

def sync_ops(p2p_op_list):
    if len(p2p_op_list) > 0:
        reqs =  torch.distributed.batch_isend_irecv(p2p_op_list)
        for req in reqs:
            req.wait()

def expert_pruning(local_expert_count, prune_ratio):

    return local_expert_count

def _tutel_style_fmoe_general_global_forward(inp, topk, gate_score, expert_fn, merged_fn, num_expert, world_size, capacity_factor, gate,**kwargs):
    r"""
    A private function that performs the following steps to complete the MoE
    computation.
    * Count the number of tokens from each worker to each expert.
    * Send the features to their target position so that input features to each
    expert are contiguous in memory.
    * Perform the forward computation of the experts using `expert_fn`
    * Gather the output features of experts back, and reorder them as sentences.
    Intermediate results like expert counts are hidden from users by this
    function.
    """

    _loss_fn = lambda gates, topk_ids: tutel.impls.losses.gshard_loss(gates, topk_ids)

    (num_global_experts, offset_s, local_expert_count, locations_s, gates_s, capacity), l_loss = tutel.impls.fast_dispatch.extract_critical(gate_score,
                top_k = topk,
                loss_fn = _loss_fn,
                capacity_factor = capacity_factor if gate.training else 0, #0 for inf
                batch_prioritized_routing = False,
                normalize_gate = True,
                group = None,
                alignment = 1,
                inequivalent_tokens = False,
                is_tight=True,
                Priority_Router=molePrioritySchedulers[kwargs["layer_id"]]
            )
    gate.set_loss(l_loss)

    crit = (num_global_experts, offset_s, locations_s, gates_s, capacity)
    inp = tutel.impls.fast_dispatch.fast_encode(inp, crit, True, is_tight=True, cf0= capacity_factor == 0 )

    global_expert_count = torch.empty_like(local_expert_count)
    torch.distributed.all_to_all_single(global_expert_count, local_expert_count)

    with torch.no_grad():
        fwd_expert_count = global_expert_count.view(world_size,
                num_expert).sum(dim=0).cpu()
        fwd_batch_size = int(fwd_expert_count.sum().item())

    inp = inp.view(-1, inp.size(-1))[:local_expert_count.sum().item()]
    inp_sz = inp.size(0)
    local_expert_count = local_expert_count.cpu()
    global_expert_count = global_expert_count.cpu()
    

    if merged_fn is not None:
        outp = merged_fn(inp, local_expert_count, global_expert_count, fwd_batch_size,fwd_expert_count, world_size, inp_sz, topk)
    
    else:
        
        torch.cuda.synchronize()
        tx = time.time()
        def scatter_func(tensor):
            return MOEScatter.apply(
            tensor,
            None,
            local_expert_count,
            global_expert_count,
            fwd_batch_size,
            world_size,
            True,
            inp_sz
            )
        x = scatter_func(inp)

        
        torch.cuda.synchronize()
        tc = time.time()
        x = expert_fn(x, fwd_expert_count)

        
        torch.cuda.synchronize()
        t0 = time.time()

        out_batch_size = inp.shape[0]

        out_batch_size *= topk

        def gather_func(tensor):
            return MOEGather.apply(
            tensor,
            None,
            local_expert_count,
            global_expert_count,
            out_batch_size,
            world_size,
            True,
            inp_sz
            )

        outp = gather_func(x)   
        
        torch.cuda.synchronize()
        t1 = time.time()

        kwargs["moe"].logtime["native_fn"] = t1 - tx
        kwargs["moe"].logtime["native_experts"] = t0 - tc
        kwargs["moe"].logtime["gather"] = t1 - t0
        kwargs["moe"].logtime["scatter"] = tc - tx

        #print("fwd time: ", t1 - t0, t0 - tc, tc - tx)

    outp = tutel.impls.fast_dispatch.fast_decode(outp, crit, True, is_tight=True, cf0= capacity_factor == 0)
    return outp

class SetBias(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, bias1, bias2, self, expert_id):
        ctx.expert_id = expert_id
        ctx.self = self
        return inp 
    @staticmethod
    def backward(ctx, grad):
        self = ctx.self
        return grad, self.get_bias(ctx.expert_id, 0), self.get_bias(ctx.expert_id, 1), None
    
#fix overlap
class FusedOverlap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, local_expert_count, global_expert_count, fwd_batch_size,fwd_expert_count, world_size, inp_sz, topk, self, *biases):
        
        output_all = torch.empty_like(inp)
        assert world_size > 1, "world_size must be > 1"
        out_batch_size = inp.shape[0]
        out_batch_size *= topk
        if isinstance(fwd_expert_count, torch.Tensor):
            fwd_expert_count_cpu = fwd_expert_count.cpu().numpy()
        shards = moleConfig.OVERLAP_SHARDS 
        shards = min(shards, self.num_expert)
        expert_per_shard = self.num_expert // shards
        expert_base = 0

        out = [None] * self.num_expert * world_size

        shard_size = []

        ctx.shards = shards
        ctx.expert_per_shard = expert_per_shard
        ctx.fwd_expert_count_cpu = fwd_expert_count_cpu
        ctx.local_expert_count, ctx.global_expert_count, ctx.fwd_batch_size = local_expert_count, global_expert_count, fwd_batch_size
        ctx.self = self 
        ctx.world_size = world_size
        ctx.inp_sz = inp_sz

        ctx.before_l1 = [None] * self.num_expert 
        ctx.before_act = [None] * self.num_expert
        ctx.before_l2 = [None] * self.num_expert
        ctx.expert_info = [None] * self.num_expert

        ctx.allocator = get_memory_allocator()
        #print("report")
        global_input_buf_all = []
        comm1 = []
        comp1 = []
        #torch.cuda.synchronize()
        comm_stream().wait_stream(torch.cuda.current_stream())

        torch.cuda.nvtx.range_push("overlap-forward")
        for i in range(shards):
            if i == shards - 1:
                this_shard_experts = self.num_expert - expert_base
            else:
                this_shard_experts = min(expert_per_shard, self.num_expert - expert_base)

            shard_size.append(this_shard_experts)

            with torch.cuda.stream(comm_stream()):
                global_input_buf = mole_global_scatter_all(
                    inp,
                    local_expert_count,
                    global_expert_count,
                    fwd_batch_size,
                    world_size,
                    expert_base, 
                    expert_base + this_shard_experts,
                )
                e_comm = torch.cuda.Event()
                e_comm.record()
                comm1.append(e_comm)
            
            global_input_buf_all.append(global_input_buf)
            
            expert_base += this_shard_experts
        #print("Get")
        expert_base = 0
        out_expert_all = []

        for shard_id in range(shards):
 
            this_shard_experts = shard_size[shard_id]

            inp_expert = global_input_buf_all[shard_id]

            outputs = []

            inp_base = 0
            comp_stream().wait_event(comm1[shard_id])

            with torch.cuda.stream(comp_stream()):
                for i in range(expert_base, expert_base + this_shard_experts):
                    batch_size = fwd_expert_count_cpu[i]
                    inp_slice = inp_expert[inp_base : inp_base + batch_size]

                    
                    if batch_size == 0 and not get_optimizer().first_step and moleConfig.SkipBS0 and not get_optimizer().need_update(self.experts_[i].layer_id, self.experts_[i].expert_id):
                        outputs.append(inp_slice)
                        inp_base += batch_size 
                        continue
                    expert_info = [self.experts_[i].layer_id, self.experts_[i].expert_id]
                    ctx.expert_info[i] = expert_info
                    fc1w_gpu = ctx.allocator.forward_get_param(0, expert_info[0], expert_info[1], inp_slice.dtype) # fc1w.to(dtype = dtype, device= dev)
                    fc2w_gpu = ctx.allocator.forward_get_param(1, expert_info[0], expert_info[1], inp_slice.dtype) #fc2w.to(dtype = dtype, device= dev)
                    ctx.before_l1[i] = inp_slice 

                    if hasattr(self, "dummy_bias"):
                        x = torch.mm(inp_slice, fc1w_gpu.t())
                    else:
                        x = torch.addmm(self.experts_[i].fc1b, inp_slice, fc1w_gpu.t())  #, 
                    ctx.before_act[i] = x 
                    ctx.before_act[i].requires_grad_()
                    with torch.enable_grad():
                        x = self.experts_[i].activation(x)
                    ctx.before_l2[i] = x
                    if hasattr(self, "dummy_bias"):
                        x = torch.mm(x, fc2w_gpu.t())
                    else:
                        x = torch.addmm(self.experts_[i].fc2b, x, fc2w_gpu.t())

                    

                    outputs.append(x)
                    inp_base += batch_size

                out_expert = torch.cat(outputs, dim=0)
                e_comp = torch.cuda.Event()
                e_comp.record()
                comp1.append(e_comp)
            
            out_expert_all.append(out_expert)
            expert_base += this_shard_experts


        expert_base = 0
        for i in range(shards):
            out_expert = out_expert_all[i]
            this_shard_experts = shard_size[i]
            comm_stream().wait_event(comp1[i])
            with torch.cuda.stream(comm_stream()):
                mole_global_gather_all(
                output_all,
                out_expert,
                local_expert_count,
                global_expert_count,
                inp_sz,
                world_size,
                expert_base, 
                expert_base + this_shard_experts,
                )
                e_comm = torch.cuda.Event()
                e_comm.record()
                comm1.append(e_comm)

            expert_base += this_shard_experts
        
        torch.cuda.current_stream().wait_stream(comm_stream())

        torch.cuda.nvtx.range_pop()

        return output_all 

    @staticmethod
    def backward(ctx, grad_out):

        shards = ctx.shards
        expert_per_shard = ctx.expert_per_shard
        local_expert_count, global_expert_count, fwd_batch_size = ctx.local_expert_count, ctx.global_expert_count, ctx.fwd_batch_size
        self = ctx.self
        world_size = ctx.world_size
        grad_in_all = torch.empty_like(grad_out)
        inp_sz = ctx.inp_sz
        fwd_expert_count_cpu = ctx.fwd_expert_count_cpu 

        out = [None] * self.num_expert * world_size
        expert_base = 0
        gradfc1s = [None] * self.num_expert
        gradfc2s = [None] * self.num_expert

        shard_size = []
        global_grad_out_buf_all = []
        out_expert_grad_all = []

        comm1 = []
        comp1 = []
        #first_ops = []
        #second_ops = []
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push("overlap-backward")

        for i in range(shards):
            if i == shards - 1:
                this_shard_experts = self.num_expert - expert_base
            else:
                this_shard_experts = min(expert_per_shard, self.num_expert - expert_base)

            shard_size.append(this_shard_experts)

            with torch.cuda.stream(comm_stream()):
                global_grad_out_buf = mole_global_scatter_all(
                grad_out,
                local_expert_count,
                global_expert_count,
                fwd_batch_size,
                world_size,
                expert_base, 
                expert_base + this_shard_experts,
                )
                e_comm = torch.cuda.Event()
                e_comm.record()
                comm1.append(e_comm)


            global_grad_out_buf_all.append(global_grad_out_buf)
            expert_base += this_shard_experts

        expert_base = 0
        for shard_id in range(shards):
            
            global_grad_out_buf = global_grad_out_buf_all[shard_id]
            inp_base = 0
            this_shard_experts = shard_size[shard_id]
            outputs = []

            comp_stream().wait_event(comm1[shard_id])

            with torch.cuda.stream(comp_stream()):
                for i in range(expert_base, expert_base + this_shard_experts):
                    batch_size = fwd_expert_count_cpu[i]
                    grad_exp_in = global_grad_out_buf[inp_base : inp_base + batch_size]
                    if batch_size == 0 and not get_optimizer().first_step and moleConfig.SkipBS0 and not get_optimizer().need_update(self.experts_[i].layer_id, self.experts_[i].expert_id):

                        inp_base += batch_size
                        outputs.append(grad_exp_in)

                        opt = get_optimizer()
                        opt.fake_early_step(0, self.layer_id, i)
                        opt.fake_early_step(1, self.layer_id, i)

                        gradfc1s[i] = torch.zeros_like(self.experts_[i].fc1b)
                        gradfc2s[i] = torch.zeros_like(self.experts_[i].fc2b)
                        continue
                    expert_info = ctx.expert_info[i]
                    fc1w_gpu = ctx.allocator.backward_get_param(0, expert_info[0], expert_info[1], grad_exp_in.dtype) #fc1w.to(dtype = ctx.dtype, device= ctx.dev)
                    fc2w_gpu = ctx.allocator.backward_get_param(1, expert_info[0], expert_info[1], grad_exp_in.dtype) #fc2w.to(dtype = ctx.dtype, device= ctx.dev)
                
                
                    grad_fc2w = torch.mm(grad_exp_in.t(), ctx.before_l2[i])
                    grad_before_l2 = torch.mm(grad_exp_in, fc2w_gpu)
                    grad_fc2b = torch.sum(grad_exp_in, dim=0)
                    ctx.before_l2[i].backward(grad_before_l2)

                    grad_fc1w = torch.mm(ctx.before_act[i].grad.t(), ctx.before_l1[i])
                    grad_fc1b = torch.sum(ctx.before_act[i].grad, dim=0)
                    if hasattr(self, "dummy_bias"):
                        grad_fc1b.zero_()
                        grad_fc2b.zero_()
                    grad_before_l1 = torch.mm(ctx.before_act[i].grad, fc1w_gpu)  

        
                    ctx.allocator.backward_set_grad(0, expert_info[0], expert_info[1], grad_fc1w)
            
                    ctx.allocator.backward_set_grad(1, expert_info[0], expert_info[1], grad_fc2w)
                
                    opt = get_optimizer()
                    if ctx.allocator.early_update(0, self.layer_id, i):
                        opt.early_step(0, self.layer_id, i)
                        ctx.allocator.sync_after_early_update(0, self.layer_id, i)
                    if ctx.allocator.early_update(1, self.layer_id, i):
                        opt.early_step(1, self.layer_id, i)
                        ctx.allocator.sync_after_early_update(1, self.layer_id, i)
                    
                    gradfc1s[i] = grad_fc1b
                    gradfc2s[i] = grad_fc2b
                    inp_base += batch_size
                    outputs.append(grad_before_l1)

                out_expert_grad = torch.cat(outputs, dim=0)
                e_comp = torch.cuda.Event()
                e_comp.record()
                comp1.append(e_comp)

            out_expert_grad_all.append(out_expert_grad)
            expert_base += this_shard_experts
            
        #torch.cuda.synchronize()
        expert_base = 0
        for i in range(shards):
            this_shard_experts = shard_size[i]
            out_expert_grad = out_expert_grad_all[i]
            
            comm_stream().wait_event(comp1[i])
            with torch.cuda.stream(comm_stream()):
                mole_global_gather_all(
                grad_in_all,
                out_expert_grad,
                local_expert_count,
                global_expert_count,
                inp_sz,
                world_size,
                expert_base, 
                expert_base + this_shard_experts,
                )

            expert_base += this_shard_experts

        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()

        all_grads = gradfc1s + gradfc2s

        return grad_in_all, None, None, None, None, None, None, None, None, *all_grads
    
class ExpertForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, fc1b, fc2b, activation, expert_info):
        dtype = fc1b.dtype 
        dev = fc1b.device
        ctx.expert_info = expert_info
        ctx.allocator = get_memory_allocator()

        fc1w_gpu = ctx.allocator.forward_get_param(0, expert_info[0], expert_info[1], inp.dtype) # fc1w.to(dtype = dtype, device= dev)
        fc2w_gpu = ctx.allocator.forward_get_param(1, expert_info[0], expert_info[1], inp.dtype) #fc2w.to(dtype = dtype, device= dev)


        ctx.dtype = dtype
        ctx.dev = dev
        ctx.before_l1 = inp 
        x = torch.addmm(fc1b, inp, fc1w_gpu.t())  #, 

        ctx.before_act = x 
        ctx.before_act.requires_grad_()

        with torch.enable_grad():
            x = activation(x)

        ctx.before_l2 = x
        x = torch.addmm(fc2b, x, fc2w_gpu.t()) #F.linear(x, fc2w.to(dtype = dtype, device= dev), fc2b)

        return x 
    
    @staticmethod
    def backward(ctx, grad_out):
        #fc1w, fc2w = ctx.fc1w, ctx.fc2w
        expert_info = ctx.expert_info
        fc1w_gpu = ctx.allocator.backward_get_param(0, expert_info[0], expert_info[1], grad_out.dtype) #fc1w.to(dtype = ctx.dtype, device= ctx.dev)
        fc2w_gpu = ctx.allocator.backward_get_param(1, expert_info[0], expert_info[1], grad_out.dtype) #fc2w.to(dtype = ctx.dtype, device= ctx.dev)
        grad_fc2w = torch.mm(grad_out.t(), ctx.before_l2)
        grad_before_l2 = torch.mm(grad_out, fc2w_gpu)
        grad_fc2b = torch.sum(grad_out, dim=0)
        ctx.before_l2.backward(grad_before_l2)

        grad_fc1w = torch.mm(ctx.before_act.grad.t(), ctx.before_l1)
        grad_fc1b = torch.sum(ctx.before_act.grad, dim=0)
        grad_before_l1 = torch.mm(ctx.before_act.grad, fc1w_gpu)  
        
        ctx.allocator.backward_set_grad(0, ctx.expert_info[0], ctx.expert_info[1], grad_fc1w)
            
        ctx.allocator.backward_set_grad(1, ctx.expert_info[0], ctx.expert_info[1], grad_fc2w)

        return grad_before_l1,  grad_fc1b, grad_fc2b, None, None, None

class EarlyUpdate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, layer_id, expert_id, alloc):

        ctx.layer_id = layer_id
        ctx.expert_id = expert_id
        ctx.alloc = alloc
        return inp
    @staticmethod
    def backward(ctx, grad):
        opt = get_optimizer()
        if ctx.alloc.early_update(0, ctx.layer_id, ctx.expert_id):
            opt.early_step(0, ctx.layer_id, ctx.expert_id)
            ctx.alloc.sync_after_early_update(0, ctx.layer_id, ctx.expert_id)
        if ctx.alloc.early_update(1, ctx.layer_id, ctx.expert_id):
            opt.early_step(1, ctx.layer_id, ctx.expert_id)
            ctx.alloc.sync_after_early_update(1, ctx.layer_id, ctx.expert_id)
        return grad, None, None, None



class _ExpertTORCH(torch.nn.Module):
    def __init__(self, hidden_size, activation, d_model, device, layer_id, expert_id, bias=True):
        super().__init__()
        self.device = device
        self.memory_allocator = get_memory_allocator()

        self.activation = activation
        self.layer_id = layer_id
        self.expert_id = expert_id

        fc1w = self.memory_allocator.init_fc(0, self.layer_id, self.expert_id, "cpu")
        fc2w = self.memory_allocator.init_fc(1, self.layer_id, self.expert_id, "cpu")

        self.fc1b = torch.nn.Parameter(torch.rand(hidden_size)) #not offload bias as it is too small
        self.fc2b = torch.nn.Parameter(torch.rand(d_model)) #

        htoh4 = torch.nn.Linear(d_model, hidden_size, bias=True)
        h4toh = torch.nn.Linear(hidden_size, d_model, bias=True)
        fc1w.data.copy_(htoh4.weight)
        self.fc1b.data.copy_(htoh4.bias)
        fc2w.data.copy_(h4toh.weight)
        self.fc2b.data.copy_(h4toh.bias)    

    @property
    def fc1w_id(self):
        return self._fc1w_id 
    
    @fc1w_id.setter
    def fc1w_id(self, value):
        if hasattr(self, "_fc1w_id"):
            raise RuntimeError("fc1w_id only set once")
        self._fc1w_id = value

    @property
    def fc2w_id(self):
        return self._fc2w_id 
    
    @fc2w_id.setter
    def fc2w_id(self, value):
        if hasattr(self, "_fc2w_id"):
            raise RuntimeError("fc1w_id only set once")
        self._fc2w_id = value

    def forward(self, inp, fwd_expert_count=None):
        inp = EarlyUpdate.apply(inp, self.layer_id, self.expert_id, self.memory_allocator)
        return ExpertForward.apply(inp, self.fc1b, self.fc2b, self.activation, [self.layer_id, self.expert_id])

    def xforward(self, inp, fwd_expert_count=None):
        x = self.htoh4(inp)
        x = self.activation(x)
        x = self.h4toh(x)
        return x
    

class moleCore(torch.nn.Module):
    r"""
    A general moe implementation that supports an arbitrary module as the
    expert.
    * `num_expert` stands for the number of experts on **each** worker.
    * `world_size` stands for the total number of workers that contains
    different experts.
    * `slice_group` can be a torch's communication group, indicating that
    specific model parallel is applied across the group, and workers in the
    group hold the same copy of input feature, and requires the same copy of
    the output. For each worker, FMoE only computes the output of a certain
    slice of the input batch, and will all-gather the outputs after
    computation.
    * `top_k` stands for the number of experts each token is going to.
    * `gate` is a gate class which can found in `fmoe.gates`.
    * `expert` can be specified as a module class, it is used to generate
    `num_expert` expert modules.
    """

    def __init__(
        self,
        num_expert=32,
        d_model=1024,
        world_size=1,
        mp_group=None,  # being deprecated
        moe_group=None,
        top_k=2,
        gate=TutelGate,
        expert=None,
        gate_hook=None,
        mask=None,
        mask_dict=None,
        layer_id=-1,
        **kwargs
    ):
        super().__init__()
        self.num_expert = num_expert
        self.d_model = d_model
        self.world_size = world_size
        assert layer_id != -1, "layer id must not be -1"
        self.layer_id = layer_id
        
        if mp_group is not None:
            print("[Warning] mp_group is being deprecated")
            self.slice_group = mp_group

        self.top_k = top_k

        self.experts = torch.nn.ModuleList([expert(d_model, idx) for idx in range(num_expert)])

        for i in range(num_expert):
            Moniter.append_expert(self.layer_id, i, self.experts[i])

        self.gate = gate(d_model, num_expert, world_size, top_k)
        self.gate_hook = gate_hook
        self.mask = mask
        self.mask_dict = mask_dict
        self.moe_group = moe_group

        self.tutel_style = (gate == TutelGate)
        if self.tutel_style:
            import tutel
            self.gate_capacity = kwargs["gate_capacity"]
        
        self.bias0 = [None] * self.num_expert
        self.bias1 = [None] * self.num_expert

        self.logtime = {}

    def expert_fn(self, inp, fwd_expert_count):
        r"""
        The default expert function which either calls the experts as a whole
        or as separate experts.
        """

        if isinstance(fwd_expert_count, torch.Tensor):
            fwd_expert_count_cpu = fwd_expert_count.cpu().numpy()
        outputs = []
        base_idx = 0
        
        for i in range(self.num_expert):
            batch_size = fwd_expert_count_cpu[i]
            inp_slice = inp[base_idx : base_idx + batch_size]
            #print("dt: ", inp_slice.dtype)
            outputs.append(self.experts[i](inp_slice, torch.tensor([fwd_expert_count[i]])))
            base_idx += batch_size
        return torch.cat(outputs, dim=0)

    def set_bias(self, grad_fcb, expert_id, bias_id):
        if bias_id == 0:
            self.bias0[expert_id] = grad_fcb
        else:
            self.bias1[expert_id] = grad_fcb

    def get_bias(self, expert_id, bias_id):
        if bias_id == 0:
            ret = self.bias0[expert_id]
            self.bias0[expert_id] = None
        else:
            ret = self.bias1[expert_id]
            self.bias1[expert_id] = None 
        return ret

    def overlap_merged_fn(self, inp, local_expert_count, global_expert_count, fwd_batch_size,fwd_expert_count, world_size, inp_sz, topk):
        r"""
        The default expert function which either calls the experts as a whole
        or as separate experts.
        """
        biases = [expert.fc1b for expert in self.experts]
        biases.extend([expert.fc2b for expert in self.experts])
        #for i in range(self.num_expert):
        #    inp = SetBias.apply(inp,  self.experts[i].fc1b, self.experts[i].fc2b, self, i)
        self.experts_ = self.experts
        return FusedOverlap.apply(inp, local_expert_count, global_expert_count, fwd_batch_size, fwd_expert_count, world_size, inp_sz, topk, self, *biases)
    

    def merged_fn(self, inp, local_expert_count, global_expert_count, fwd_batch_size,fwd_expert_count, world_size, inp_sz, topk):
        r"""
        The default expert function which either calls the experts as a whole
        or as separate experts.
        """
        torch.cuda.synchronize()
        time0 = time.time()
        out_batch_size = inp.shape[0]
        out_batch_size *= topk
        if isinstance(fwd_expert_count, torch.Tensor):
            fwd_expert_count_cpu = fwd_expert_count.cpu().numpy()
        shards = moleConfig.OVERLAP_SHARDS
        shards = min(shards, self.num_expert)
        expert_per_shard = self.num_expert // shards
        expert_base = 0

        out = [None] * self.num_expert * world_size


        shard_size = []
        inp_all = torch.split(inp, local_expert_count.tolist(), dim=0)

        for i in range(shards):
            in_all = []
            if i == shards - 1:
                this_shard_experts = self.num_expert - expert_base
            else:
                this_shard_experts = min(expert_per_shard, self.num_expert - expert_base)
            for each_node in range(world_size):
                for expert_count in range(this_shard_experts):
                    in_all.append(inp_all[each_node * self.num_expert + expert_base + expert_count]) 
            torch.cuda.synchronize()
            cattime = time.time()

            in_all = torch.cat(in_all, dim=0)
            
            shard_size.append(this_shard_experts)
            torch.cuda.synchronize()
            tes = time.time()

            inp_expert = MOEScatterShard.apply(
                in_all,
                None,
                local_expert_count,
                global_expert_count,
                fwd_batch_size,
                world_size,
                True,
                inp_sz,
                expert_base,
                expert_base + this_shard_experts,
                self
                )
            outputs = []

            torch.cuda.synchronize()
            te0 = time.time()

            self.logtime["scatter"] = te0 - tes

            inp_base = 0
            for i in range(expert_base, expert_base + this_shard_experts):
                batch_size = fwd_expert_count_cpu[i]
                inp_slice = inp_expert[inp_base : inp_base + batch_size]
                outputs.append(self.experts[i](inp_slice, torch.tensor([fwd_expert_count_cpu[i]])))
                inp_base += batch_size

            out_expert = torch.cat(outputs, dim=0)

            splits = []

            torch.cuda.synchronize()
            te1 = time.time()

            self.logtime["merged_experts"] = te1 - te0

            out_all = MOEGatherShard.apply(
                out_expert,
                None,
                local_expert_count,
                global_expert_count,
                out_batch_size,
                world_size,
                True,
                inp_sz,
                expert_base,                
                expert_base + this_shard_experts,
                self
                )
            torch.cuda.synchronize()
            teg = time.time()

            self.logtime["gather"] = teg - te1

            for each_node in range(world_size):
                for expert_count in range(this_shard_experts):
                    splits.append(local_expert_count[each_node * self.num_expert + expert_base + expert_count]) 

            out_all = torch.split(out_all, splits)
            
            for each_node in range(world_size):
                for expert_count in range(this_shard_experts):
                    out[each_node * self.num_expert + expert_base + expert_count] = out_all[each_node * this_shard_experts + expert_count]
            
            expert_base += this_shard_experts

        torch.cuda.synchronize()
        cat2 = time.time()

        out_new = []
        for t in out:
            if t is not None:
                out_new.append(t)
        out = torch.cat(out_new, dim=0) 

        assert out.size() == inp.size()
        torch.cuda.synchronize()
        time1 = time.time()

        self.logtime["merged_fn"] = time1 - time0
        self.logtime["cattime"] = tes - cattime 
        self.logtime["cat2time"] = time1 - cat2
        self.logtime["splittime"] = cat2 - teg
        return out 
    
    def forward(self, moe_inp):
        r"""
        The FMoE module first computes gate output, and then conduct MoE forward
        according to the gate.  The score of the selected gate given by the
        expert is multiplied to the experts' output tensors as a weight.
        """


        if self.world_size > 1:

            def ensure_comm_func(tensor):
                ensure_comm(tensor, self.moe_group)

            ensure_comm_func( moe_inp)


        gate_top_k_idx, gate_score = self.gate(moe_inp)

        if self.gate_hook is not None:
            self.gate_hook(gate_top_k_idx, gate_score, None)
        
        assert self.tutel_style, "Only Support Tutel Style"

        
        fwd = _tutel_style_fmoe_general_global_forward(
                moe_inp, self.top_k, gate_score, self.expert_fn, self.overlap_merged_fn if (moleConfig.OVERLAP_SHARDS != -1 and self.world_size > 1) else None,
            self.num_expert, self.world_size,
            experts=self.experts, 
            capacity_factor=self.gate_capacity,
            gate=self.gate,
            layer_id = self._hack_num_layer,
            moe=self
            )
        return fwd
         


class moleLayer(moleCore):
    r"""
    A complete MoE MLP module in a Transformer block.
    * `activation` is the activation function to be used in MLP in each expert.
    * `d_hidden` is the dimension of the MLP layer.
    """

    def __init__(
        self,
        num_expert=32,
        d_model=1024,
        d_hidden=4096,
        activation=torch.nn.GELU(),
        expert_dp_comm="none",
        expert_rank=0,
        **kwargs
    ):
        
        def one_expert(d_model, expert_id):
            return _ExpertTORCH(d_hidden, activation, d_model, kwargs["device"], kwargs["layer_id"], expert_id)
        expert = one_expert
        super().__init__(num_expert=num_expert, d_model=d_model, expert=expert, **kwargs)


    def forward(self, inp: torch.Tensor):
        r"""
        This module wraps up the FMoE module with reshape, residual and layer
        normalization.
        """
        original_shape = inp.shape
        inp = inp.reshape(-1, self.d_model)
        output = super().forward(inp)
        return output.reshape(original_shape)




