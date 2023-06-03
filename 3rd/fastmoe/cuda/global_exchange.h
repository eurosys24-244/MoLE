#include "stream_manager.h"
#ifdef FMOE_USE_NCCL
#include <iostream>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDACachingAllocator.h>


void fmoe_cuda_expert_exchange_impl(
        const long* local_expert_count,
        long* global_expert_count,
        int n_expert, int world_size,
        CudaStreamManager* smgr);

template<typename scalar_t>
void mole_cuda_global_scatter_impl(
    const scalar_t* local_input_buf,
    const long* local_expert_count,
    const long* global_expert_count,
    scalar_t* input_buf,
    size_t in_feat, size_t n_expert, size_t world_size,
    CudaStreamManager* smgr, size_t shard_start, size_t shard_end, const long* expert_ptr, const long* recv_expert_ptr) {
    // assert world_size > 1
    
    NCCL_SAFE_CALL(ncclGroupStart());
    for (size_t i = shard_start;i < shard_end;i++) {
        
        for (size_t j = 0; j < world_size; ++j) {
            int cnt = i + j * n_expert;
            int recv_table_cnt = j + (i - shard_start) * world_size;
            int send_table_cnt = (i - shard_start) + j * (shard_end - shard_start);

            if (local_expert_count[cnt]) {
                NCCL_SAFE_CALL(ncclSend(
                        local_input_buf + expert_ptr[send_table_cnt] * in_feat,
                        local_expert_count[cnt] * in_feat * sizeof(scalar_t),
                        ncclChar,
                        j,
                        smgr->ncclcomm,
                        at::cuda::getCurrentCUDAStream().stream()));
            }
            if (global_expert_count[cnt]) {
                NCCL_SAFE_CALL(ncclRecv(
                        input_buf + recv_expert_ptr[recv_table_cnt] * in_feat,
                        global_expert_count[cnt] * in_feat * sizeof(scalar_t),
                        ncclChar,
                        j,
                        smgr->ncclcomm,
                        at::cuda::getCurrentCUDAStream().stream()));
            }
        }
        
    }
    NCCL_SAFE_CALL(ncclGroupEnd());
}


template<typename scalar_t>
void mole_cuda_global_scatter_impl_all(
    const scalar_t* local_input_buf,
    const long* local_expert_count,
    const long* global_expert_count,
    scalar_t* input_buf,
    size_t in_feat, size_t n_expert, size_t world_size,
    CudaStreamManager* smgr, size_t shard_start, size_t shard_end, const long* expert_ptr, const long* recv_expert_ptr) {
    // assert world_size > 1
    
    NCCL_SAFE_CALL(ncclGroupStart());
    for (size_t i = shard_start;i < shard_end;i++) {
        
        for (size_t j = 0; j < world_size; ++j) {
            int cnt = i + j * n_expert;
            int recv_table_cnt = j + (i - shard_start) * world_size;
            int send_table_cnt = (i - shard_start) + j * (shard_end - shard_start);

            if (local_expert_count[cnt]) {
                NCCL_SAFE_CALL(ncclSend(
                        local_input_buf + expert_ptr[send_table_cnt] * in_feat,
                        local_expert_count[cnt] * in_feat * sizeof(scalar_t),
                        ncclChar,
                        j,
                        smgr->ncclcomm,
                        at::cuda::getCurrentCUDAStream().stream()));
            }
            if (global_expert_count[cnt]) {
                NCCL_SAFE_CALL(ncclRecv(
                        input_buf + recv_expert_ptr[recv_table_cnt] * in_feat,
                        global_expert_count[cnt] * in_feat * sizeof(scalar_t),
                        ncclChar,
                        j,
                        smgr->ncclcomm,
                        at::cuda::getCurrentCUDAStream().stream()));
            }
        }
    }
    NCCL_SAFE_CALL(ncclGroupEnd());
}


template<typename scalar_t>
void mole_cuda_global_gather_impl(
    const scalar_t* output_buf,
    const long* local_expert_count,
    const long* global_expert_count,
    scalar_t* local_output_buf,
    size_t out_feat, size_t n_expert, size_t world_size,
    CudaStreamManager* smgr, size_t shard_start, size_t shard_end, const long* expert_ptr, const long* send_expert_ptr) {


    for (size_t i = shard_start;i < shard_end;i++) {
        NCCL_SAFE_CALL(ncclGroupStart());
        for (size_t j = 0; j < world_size; ++j) {
            int cnt = j * n_expert + i;
            int recv_table_cnt = j + (i - shard_start) * world_size;
            int send_table_cnt = (i - shard_start) + j * (shard_end - shard_start);
            if (global_expert_count[cnt]) {
                NCCL_SAFE_CALL(ncclSend(
                        output_buf + send_expert_ptr[recv_table_cnt] * out_feat,
                        global_expert_count[cnt] * out_feat * sizeof(scalar_t),
                        ncclChar,
                        j,
                        smgr->ncclcomm,
                        at::cuda::getCurrentCUDAStream().stream()));
            }
            if (local_expert_count[cnt]) {
                NCCL_SAFE_CALL(ncclRecv(
                        local_output_buf + expert_ptr[send_table_cnt] * out_feat,
                        local_expert_count[cnt] * out_feat * sizeof(scalar_t),
                        ncclChar,
                        j,
                        smgr->ncclcomm,
                        at::cuda::getCurrentCUDAStream().stream()));
            }
        }
        NCCL_SAFE_CALL(ncclGroupEnd());
    }
}


template<typename scalar_t>
void fmoe_cuda_global_scatter_impl(
    const scalar_t* local_input_buf,
    const long* local_expert_count,
    const long* global_expert_count,
    scalar_t* input_buf,
    size_t in_feat, size_t n_expert, size_t world_size,
    CudaStreamManager* smgr) {
    // assert world_size > 1
    int recv_ptr = 0;
    /* TODO: may save for backward */
    long*expert_ptr = new long[n_expert * world_size];
    expert_ptr[0] = 0;
    for (size_t i = 1; i < n_expert * world_size; ++i) {
        expert_ptr[i] = expert_ptr[i - 1] + local_expert_count[i - 1];
    }

    for (size_t i = 0; i < n_expert; ++i) {
        NCCL_SAFE_CALL(ncclGroupStart());
        for (size_t j = 0; j < world_size; ++j) {
            int idx = i + j * n_expert;
            if (local_expert_count[idx]) {
                NCCL_SAFE_CALL(ncclSend(
                        local_input_buf + expert_ptr[idx] * in_feat,
                        local_expert_count[idx] * in_feat * sizeof(scalar_t),
                        ncclChar,
                        j,
                        smgr->ncclcomm,
                        smgr->stream(0)));
            }
            if (global_expert_count[idx]) {
                NCCL_SAFE_CALL(ncclRecv(
                        input_buf + recv_ptr * in_feat,
                        global_expert_count[idx] * in_feat * sizeof(scalar_t),
                        ncclChar,
                        j,
                        smgr->ncclcomm,
                        smgr->stream(0)));
                recv_ptr += global_expert_count[idx];
            }
        }
        NCCL_SAFE_CALL(ncclGroupEnd());
    }
    delete [] expert_ptr;
    smgr->sync(1);
}

template<typename scalar_t>
void fmoe_cuda_global_gather_impl(
    const scalar_t* output_buf,
    const long* local_expert_count,
    const long* global_expert_count,
    scalar_t* local_output_buf,
    size_t out_feat, size_t n_expert, size_t world_size,
    CudaStreamManager* smgr) {
    long send_ptr = 0;
    /* TODO: may save for backward */
    long *expert_ptr = new long[n_expert * world_size];
    expert_ptr[0] = 0;
    for (size_t i = 1; i < n_expert * world_size; ++i) {
        expert_ptr[i] = expert_ptr[i - 1] + local_expert_count[i - 1];
    }

    for (size_t i = 0; i < n_expert; ++i) {
        NCCL_SAFE_CALL(ncclGroupStart());
        for (size_t j = 0; j < world_size; ++j) {
            int idx = i + j * n_expert;
            if (global_expert_count[idx]) {
                NCCL_SAFE_CALL(ncclSend(
                        output_buf + send_ptr * out_feat,
                        global_expert_count[idx] * out_feat * sizeof(scalar_t),
                        ncclChar,
                        j,
                        smgr->ncclcomm,
                        smgr->stream(0)));
                send_ptr += global_expert_count[idx];
            }
            if (local_expert_count[idx]) {
                NCCL_SAFE_CALL(ncclRecv(
                        local_output_buf + expert_ptr[idx] * out_feat,
                        local_expert_count[idx] * out_feat * sizeof(scalar_t),
                        ncclChar,
                        j,
                        smgr->ncclcomm,
                        smgr->stream(0)));
            }
        }
        NCCL_SAFE_CALL(ncclGroupEnd());
    }
    delete [] expert_ptr;
    smgr->sync(1);
}


#endif  // FMOE_USE_NCCL
