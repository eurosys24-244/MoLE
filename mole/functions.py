r"""
The fmoe.functions module contains functions that are directly warped up from
C/CUDA functions to complete distributed communication, computation and gradient
computation.
"""

import torch
from torch.autograd import Function
import fmoe_cuda

from .utils import get_torch_default_comm
import torch.distributed as dist
import logging
import time

_moe_group = None


def ensure_comm(t, comm):
    if comm is None:
        comm = get_torch_default_comm()
    global _moe_group
    _moe_group = comm
    #fmoe_cuda.ensure_nccl(comm, t)
    torch_version_major, torch_version_minor = list(map(int, torch.__version__.split(".")[:2]))
    if torch_version_major > 1 or torch_version_minor >= 13:
        fmoe_cuda.ensure_nccl(comm._get_backend(t.device), t)
    else:
        fmoe_cuda.ensure_nccl(comm, t)


def get_moe_group():
    return _moe_group

def _local_scatter(inp, pos):
    inp_buf = torch.index_select(inp, 0, pos)
    return inp_buf


def _local_gather(inp, pos, out_batch_size, maybe_overlap=True):
    inp_buf = torch.zeros(out_batch_size, inp.shape[-1],
            dtype=inp.dtype, device=inp.device)
    if maybe_overlap:
        inp_buf.index_add_(0, pos, inp)
    else:
        inp_buf.index_copy_(0, pos, inp)
    return inp_buf

def mole_global_scatter(input_buf, local_expert_count, global_expert_count,
                      batch_size, n_workers, shard_start=-1, shard_end=-1):
    ret = fmoe_cuda.mole_global_scatter(input_buf, local_expert_count, global_expert_count,batch_size, n_workers, shard_start, shard_end)
    return ret       
def mole_global_scatter_all(input_buf, local_expert_count, global_expert_count,
                      batch_size, n_workers, shard_start=-1, shard_end=-1):
    ret = fmoe_cuda.mole_global_scatter_all(input_buf, local_expert_count, global_expert_count,batch_size, n_workers, shard_start, shard_end)
    return ret       

def mole_global_gather_all(output_all, input_buf, local_expert_count, global_expert_count,
                      batch_size, n_workers, shard_start=-1, shard_end=-1):
    ret = fmoe_cuda.mole_global_gather_all(output_all, input_buf, local_expert_count, global_expert_count,batch_size, n_workers, shard_start, shard_end)
    return ret       

def mole_global_gather(input_buf, local_expert_count, global_expert_count,
                      batch_size, n_workers, shard_start=-1, shard_end=-1):
    ret = fmoe_cuda.mole_global_gather(input_buf, local_expert_count, global_expert_count,batch_size, n_workers, shard_start, shard_end)
    return ret                   
    
def py_global_scatter(input_buf, local_expert_count, global_expert_count,
                      batch_size, n_workers, shard_start=-1, shard_end=-1, async_flag=False):
    
    t0 = time.time()
    in_feat = input_buf.size(1)
    n_expert = local_expert_count.size(0) // n_workers #local expert numbers
    expert_ptr = [0]
    recv_expert_ptr = [0]
    if shard_start == -1:
        shard_start = 0
        shard_end = n_expert

    cnt = 0

    for j in range( n_workers):
        for i in range(shard_start, shard_end):
            cnt = j * n_expert + i
            expert_ptr.append(expert_ptr[-1] + local_expert_count[cnt].item()) 

    for i in range(shard_start, shard_end):
        for j in range( n_workers):
            cnt = j * n_expert + i
            recv_expert_ptr.append(recv_expert_ptr[-1] + global_expert_count[cnt].item())
    torch.cuda.synchronize()
    t1 = time.time()
    
    global_input_buf = input_buf.new_empty((recv_expert_ptr[-1], in_feat))

    p2p_op_list = []
    for i in range(shard_start, shard_end):
        
        for j in range(n_workers):
            cnt = j * n_expert + i
            recv_table_cnt = j + (i - shard_start) * n_workers
            send_table_cnt = (i - shard_start) + j * (shard_end - shard_start)
            if local_expert_count[cnt].item():
                p2p_op_list.append(dist.P2POp(dist.isend, input_buf[expert_ptr[send_table_cnt]:expert_ptr[send_table_cnt] + local_expert_count[cnt].item()], j))
                
            if global_expert_count[cnt].item():
                p2p_op_list.append(dist.P2POp(dist.irecv, global_input_buf[recv_expert_ptr[recv_table_cnt]:recv_expert_ptr[recv_table_cnt] + global_expert_count[cnt].item()], j))

    torch.cuda.synchronize()
    t2 = time.time()
    

    if async_flag:
        return global_input_buf, p2p_op_list

    if len(p2p_op_list) > 0:
        reqs =  dist.batch_isend_irecv(p2p_op_list)
        for req in reqs:
            req.wait()

    torch.cuda.synchronize()
    t3 = time.time()
    

    return global_input_buf

def py_global_gather(output_buf, local_expert_count, global_expert_count,
                     batch_size, n_workers, shard_start=-1, shard_end=-1, async_flag=False):
    t0 = time.time()
    out_feat = output_buf.size(1)
    
    n_expert = local_expert_count.size(0) // n_workers #local expert numbers
    expert_ptr = [0]
    send_expert_ptr = [0]
    #send_ptr = 0
    if shard_start == -1:
        shard_start = 0
        shard_end = n_expert  

    cnt = 0

    for j in range( n_workers):
        for i in range(shard_start, shard_end):
            cnt = j * n_expert + i
            expert_ptr.append(expert_ptr[-1] + local_expert_count[cnt].item()) 

    for i in range(shard_start, shard_end):
        for j in range( n_workers):
            cnt = j * n_expert + i
            send_expert_ptr.append(send_expert_ptr[-1] + global_expert_count[cnt].item())

    torch.cuda.synchronize()
    t1 = time.time()

    local_output_buf = output_buf.new_empty((expert_ptr[-1], out_feat))

    p2p_op_list = []
    for i in range(shard_start, shard_end):
        for j in range(n_workers):
            cnt = j * n_expert + i
            recv_table_cnt = j + (i - shard_start) * n_workers
            send_table_cnt = (i - shard_start) + j * (shard_end - shard_start)
            
            if global_expert_count[cnt].item():
                p2p_op_list.append(dist.P2POp(dist.isend, output_buf[send_expert_ptr[recv_table_cnt]:send_expert_ptr[recv_table_cnt] + global_expert_count[cnt].item()], j))
            if local_expert_count[cnt].item():
                p2p_op_list.append(dist.P2POp(dist.irecv, local_output_buf[expert_ptr[send_table_cnt]:expert_ptr[send_table_cnt] + local_expert_count[cnt].item()], j))

    torch.cuda.synchronize()
    t2 = time.time()

    if async_flag:
        return local_output_buf, p2p_op_list
    
    if len(p2p_op_list) > 0:
        reqs =  dist.batch_isend_irecv(p2p_op_list)
        for req in reqs:
            req.wait()

    torch.cuda.synchronize()
    t3 = time.time()
    

    return local_output_buf

class MOEScatterShard(Function):
    r"""
    Scatter input samples from [batch x sequences] to contiguous alone experts.
    If `world_size` is greater than 1, the samples will first be locally
    scattered, and then exchanged across workers.
    """

    @staticmethod
    def forward(
        ctx,
        inp,
        pos,
        local_expert_count,
        global_expert_count,
        fwd_batch_size,
        world_size,
        skip_local=False,
        inp_size=None,
        shard_start=-1, shard_end=-1,
        MoE=None
    ):
        ctx.skip_local = skip_local
        ctx.shard_start, ctx.shard_end = shard_start, shard_end
        if not skip_local:
            local_input_buf = _local_scatter(inp, pos)
        else:
            local_input_buf = inp
        if world_size > 1:
            global_input_buf = mole_global_scatter(
                local_input_buf,
                local_expert_count,
                global_expert_count,
                fwd_batch_size,
                world_size,
                shard_start, 
                shard_end
            )
            torch.cuda.synchronize()

        else:
            global_input_buf = local_input_buf
        if not skip_local:
            ctx.moe_args = inp.shape[0], pos.shape[0], world_size
        else:
            ctx.moe_args = inp.shape[0], inp_size, world_size
        variables = (pos, local_expert_count, global_expert_count)
        ctx.save_for_backward(*variables)
        return global_input_buf

    @staticmethod
    def backward(ctx, global_grad_in):
        (pos, local_expert_count, global_expert_count) = ctx.saved_tensors
        (inp_batch_size, buf_batch_size, world_size) = ctx.moe_args

        if world_size > 1:
            local_grad_in = mole_global_gather(
                global_grad_in,
                local_expert_count,
                global_expert_count,
                buf_batch_size,
                world_size,
                ctx.shard_start,
                ctx.shard_end
            )
            torch.cuda.synchronize()
        else:
            local_grad_in = global_grad_in
        if not ctx.skip_local:
            grad_in = _local_gather(local_grad_in, pos, inp_batch_size)
        else:
            grad_in = local_grad_in
        return grad_in, None, None, None, None, None, None, None, None, None, None

class MOEGatherShard(Function):
    r"""
    Gather output samples from contiguous alone experts back to [batch x
    sequences]. Works symmetrically with MOEScatter.
    """

    @staticmethod
    def forward(
        ctx,
        global_output_buf,
        pos,
        local_expert_count,
        global_expert_count,
        local_batch_size,
        world_size,
        skip_local=False,
        inp_size=None,
        shard_start=-1, 
        shard_end=-1,
        MoE=None,
    ):  
        ctx.shard_start, ctx.shard_end = shard_start, shard_end
        ctx.skip_local = skip_local
        if world_size > 1:
            local_output_buf = mole_global_gather(
                global_output_buf,
                local_expert_count,
                global_expert_count,
                pos.shape[0] if not skip_local else inp_size,
                world_size,
                shard_start, 
                shard_end
            )
            torch.cuda.synchronize()
        else:
            local_output_buf = global_output_buf
        if not skip_local:
            output = _local_gather(local_output_buf, pos, local_batch_size,
                maybe_overlap=False)
        else:
            output = local_output_buf

        ctx.moe_args = (global_output_buf.shape[0], world_size)
        variables = (pos, local_expert_count, global_expert_count)
        ctx.save_for_backward(*variables)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        pos, local_expert_count, global_expert_count = ctx.saved_tensors
        fwd_batch_size, world_size = ctx.moe_args
        if not ctx.skip_local:
            grad_out_buf = _local_scatter(grad_out.contiguous(), pos)
        else:
            grad_out_buf = grad_out.contiguous()
        if world_size > 1:
            global_grad_out_buf = mole_global_scatter(
                grad_out_buf,
                local_expert_count,
                global_expert_count,
                fwd_batch_size,
                world_size,
                ctx.shard_start, 
                ctx.shard_end
            )
            torch.cuda.synchronize()
        else:
            global_grad_out_buf = grad_out_buf
        return global_grad_out_buf, None, None, None, None, None, None, None, None, None, None


class MOEScatter(Function):
    r"""
    Scatter input samples from [batch x sequences] to contiguous alone experts.
    If `world_size` is greater than 1, the samples will first be locally
    scattered, and then exchanged across workers.
    """

    @staticmethod
    def forward(
        ctx,
        inp,
        pos,
        local_expert_count,
        global_expert_count,
        fwd_batch_size,
        world_size,
        skip_local=False,
        inp_size=None
    ):
        ctx.skip_local = skip_local
        if not skip_local:
            local_input_buf = _local_scatter(inp, pos)
        else:
            local_input_buf = inp
        if world_size > 1:
            global_input_buf = fmoe_cuda.global_scatter(
                local_input_buf,
                local_expert_count,
                global_expert_count,
                fwd_batch_size,
                world_size,
            )
        else:
            global_input_buf = local_input_buf
        if not skip_local:
            ctx.moe_args = inp.shape[0], pos.shape[0], world_size
        else:
            ctx.moe_args = inp.shape[0], inp_size, world_size
        variables = (pos, local_expert_count, global_expert_count)
        ctx.save_for_backward(*variables)
        return global_input_buf

    @staticmethod
    def backward(ctx, global_grad_in):
        (pos, local_expert_count, global_expert_count) = ctx.saved_tensors
        (inp_batch_size, buf_batch_size, world_size) = ctx.moe_args

        if world_size > 1:
            local_grad_in = fmoe_cuda.global_gather(
                global_grad_in,
                local_expert_count,
                global_expert_count,
                buf_batch_size,
                world_size,
            )
        else:
            local_grad_in = global_grad_in
        if not ctx.skip_local:
            grad_in = _local_gather(local_grad_in, pos, inp_batch_size)
        else:
            grad_in = local_grad_in
        return grad_in, None, None, None, None, None, None, None

class MOEGather(Function):
    r"""
    Gather output samples from contiguous alone experts back to [batch x
    sequences]. Works symmetrically with MOEScatter.
    """

    @staticmethod
    def forward(
        ctx,
        global_output_buf,
        pos,
        local_expert_count,
        global_expert_count,
        local_batch_size,
        world_size,
        skip_local=False,
        inp_size=None,
    ):  
        ctx.skip_local = skip_local
        if world_size > 1:
            local_output_buf = fmoe_cuda.global_gather(
                global_output_buf,
                local_expert_count,
                global_expert_count,
                pos.shape[0] if not skip_local else inp_size,
                world_size,
            )
        else:
            local_output_buf = global_output_buf
        if not skip_local:
            output = _local_gather(local_output_buf, pos, local_batch_size,
                maybe_overlap=False)
        else:
            output = local_output_buf

        ctx.moe_args = (global_output_buf.shape[0], world_size)
        variables = (pos, local_expert_count, global_expert_count)
        ctx.save_for_backward(*variables)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        pos, local_expert_count, global_expert_count = ctx.saved_tensors
        fwd_batch_size, world_size = ctx.moe_args
        if not ctx.skip_local:
            grad_out_buf = _local_scatter(grad_out.contiguous(), pos)
        else:
            grad_out_buf = grad_out.contiguous()
        if world_size > 1:
            global_grad_out_buf = fmoe_cuda.global_scatter(
                grad_out_buf,
                local_expert_count,
                global_expert_count,
                fwd_batch_size,
                world_size,
            )
        else:
            global_grad_out_buf = grad_out_buf
        return global_grad_out_buf, None, None, None, None, None, None, None
