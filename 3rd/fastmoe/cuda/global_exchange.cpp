#include "global_exchange.h"
#include "utils/fmoe_utils.h"
#include <torch/extension.h>

#ifdef FMOE_USE_NCCL
#include <nccl.h>


void fmoe_cuda_expert_exchange_impl(
        const long* local_expert_count,
        long* global_expert_count,
        int n_expert, int world_size,
        CudaStreamManager* smgr) {
    NCCL_SAFE_CALL(ncclGroupStart());
    for (int i = 0; i < world_size; ++i) {
        NCCL_SAFE_CALL(ncclSend(
                local_expert_count + n_expert * i,
                n_expert,
                ncclInt64,
                i,
                smgr->ncclcomm,
                smgr->stream(0)));
        NCCL_SAFE_CALL(ncclRecv(
                global_expert_count + n_expert * i,
                n_expert,
                ncclInt64,
                i,
                smgr->ncclcomm,
                smgr->stream(0)));
    }
    NCCL_SAFE_CALL(ncclGroupEnd());
    smgr->sync(1);
}

torch::Tensor _expert_exchange(
        torch::Tensor local_expert_count,
        long n_expert, long n_workers) {
    auto global_expert_count = torch::empty_like(local_expert_count);
    auto smgr = getCudaStreamManager(local_expert_count.device().index());

    fmoe_cuda_expert_exchange_impl(
            local_expert_count.data_ptr<long>(),
            global_expert_count.data_ptr<long>(),
            n_expert, n_workers,
            smgr);
    return global_expert_count;
}

torch::Tensor _global_scatter(
        torch::Tensor input_buf,
        torch::Tensor local_expert_count,
        torch::Tensor global_expert_count,
        long batch_size, long n_workers) {
    CHECK_INPUT(input_buf);

    auto n_expert = local_expert_count.size(0) / n_workers;
    auto in_feat = input_buf.size(1);
    auto global_input_buf = input_buf.new_empty({batch_size, in_feat});
    auto smgr = getCudaStreamManager(input_buf.device().index());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_buf.scalar_type(),
            "fmoe_cuda_global_scatter", ([&] {
        fmoe_cuda_global_scatter_impl<scalar_t>(
            input_buf.data_ptr<scalar_t>(),
            local_expert_count.data_ptr<long>(),
            global_expert_count.data_ptr<long>(),
            global_input_buf.data_ptr<scalar_t>(),
            in_feat, n_expert, n_workers,
            smgr
        );
    }));
    return global_input_buf;
}

torch::Tensor _global_gather(
        torch::Tensor output_buf,
        torch::Tensor local_expert_count,
        torch::Tensor global_expert_count,
        long batch_size, long n_workers) {
    CHECK_INPUT(output_buf);

    auto n_expert = local_expert_count.size(0) / n_workers;
    auto out_feat = output_buf.size(1);
    auto local_output_buf = output_buf.new_empty({batch_size, out_feat});
    auto smgr = getCudaStreamManager(output_buf.device().index());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(output_buf.scalar_type(),
            "fmoe_cuda_global_gather", ([&] {
        fmoe_cuda_global_gather_impl<scalar_t>(
            output_buf.data_ptr<scalar_t>(),
            local_expert_count.data_ptr<long>(),
            global_expert_count.data_ptr<long>(),
            local_output_buf.data_ptr<scalar_t>(),
            out_feat, n_expert, n_workers,
            smgr
        );
    }));
    return local_output_buf;
}

torch::Tensor _mole_global_scatter_all(
        torch::Tensor input_buf,
        torch::Tensor local_expert_count,
        torch::Tensor global_expert_count,
        long batch_size, long n_workers, long shard_start, long shard_end) {
    CHECK_INPUT(input_buf);

    auto n_expert = local_expert_count.size(0) / n_workers;
    auto in_feat = input_buf.size(1);
    
    auto smgr = getCudaStreamManager(input_buf.device().index());
    if(shard_start == -1){
        shard_start = 0;
        shard_end = n_expert;
    }
    std::vector<long> expert_ptr;
    std::vector<long> recv_expert_ptr;
    std::vector<long> bases;

    recv_expert_ptr.emplace_back(0);
    bases.emplace_back(0);

    for (size_t i = 0;i < local_expert_count.size(0);++i){
        bases.emplace_back(bases.back() + local_expert_count[i].item<long>());
    }
    //base[i] is the start pointer of the i-th expert
    
    size_t this_shard_experts = shard_end - shard_start;
    for (size_t each_node = 0;each_node < n_workers;++each_node){
        for (size_t expert_count = 0;expert_count < this_shard_experts;++expert_count){
            int cnt = each_node * n_expert + shard_start + expert_count;
            expert_ptr.emplace_back(bases[cnt]);
        }
    }

    for (size_t i = shard_start;i < shard_end;i++){
        for (size_t j = 0;j < n_workers;j++){
            size_t cnt = j * n_expert + i;
            recv_expert_ptr.emplace_back(recv_expert_ptr.back() + global_expert_count[cnt].item<long>());
        }
    }

    auto global_input_buf = input_buf.new_empty({recv_expert_ptr.back(), in_feat});
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_buf.scalar_type(),
            "mole_cuda_global_scatter", ([&] {
        mole_cuda_global_scatter_impl_all<scalar_t>(
            input_buf.data_ptr<scalar_t>(),
            local_expert_count.data_ptr<long>(),
            global_expert_count.data_ptr<long>(),
            global_input_buf.data_ptr<scalar_t>(),
            in_feat, n_expert, n_workers,
            smgr, shard_start, shard_end, expert_ptr.data(), recv_expert_ptr.data()
        );
    }));
    return global_input_buf;
}

torch::Tensor _mole_global_gather_all(
        torch::Tensor output_all,
        torch::Tensor output_buf,
        torch::Tensor local_expert_count,
        torch::Tensor global_expert_count,
        long batch_size, long n_workers, long shard_start, long shard_end) {
    CHECK_INPUT(output_buf);

    auto n_expert = local_expert_count.size(0) / n_workers;
    auto out_feat = output_buf.size(1);
    
    auto smgr = getCudaStreamManager(output_buf.device().index());
    if(shard_start == -1){
        shard_start = 0;
        shard_end = n_expert;
    }
    std::vector<long> expert_ptr;
    std::vector<long> send_expert_ptr;
    std::vector<long> bases;

    send_expert_ptr.emplace_back(0);
    bases.emplace_back(0);

    for (size_t i = 0;i < local_expert_count.size(0);++i){
        bases.emplace_back(bases.back() + local_expert_count[i].item<long>());
    }

    //size_t this_shard_experts = shard_end - shard_start;

    size_t this_shard_experts = shard_end - shard_start;
    for (size_t each_node = 0;each_node < n_workers;++each_node){
        for (size_t expert_count = 0;expert_count < this_shard_experts;++expert_count){
            int cnt = each_node * n_expert + shard_start + expert_count;
            expert_ptr.emplace_back(bases[cnt]);
        }
    }
    /*
    for (size_t j = 0;j < n_workers;++j){
        for (size_t i = shard_start;i < shard_end;i++){
            size_t cnt = j * n_expert + i;
            expert_ptr.emplace_back(expert_ptr.back() + local_expert_count[cnt].item<long>());
        }
    }*/

    for (size_t i = shard_start;i < shard_end;i++){
        for (size_t j = 0;j < n_workers;j++){
            size_t cnt = j * n_expert + i;
            send_expert_ptr.emplace_back(send_expert_ptr.back() + global_expert_count[cnt].item<long>());
        }
    }
    //std::cout << "Sending: " << output_buf.size(0) << " " << send_expert_ptr << std::endl;
    //std::cout << "Recving: " << output_all.size(0) << " " << expert_ptr << std::endl;
    //auto local_output_buf = output_buf.new_empty({expert_ptr.back(), out_feat});
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(output_buf.scalar_type(),
            "mole_cuda_global_gather", ([&] {
        mole_cuda_global_gather_impl<scalar_t>(
            output_buf.data_ptr<scalar_t>(),
            local_expert_count.data_ptr<long>(),
            global_expert_count.data_ptr<long>(),
            output_all.data_ptr<scalar_t>(),
            out_feat, n_expert, n_workers,
            smgr, shard_start, shard_end, expert_ptr.data(), send_expert_ptr.data()
        );
    }));
    return output_all;
}


torch::Tensor _mole_global_scatter(
        torch::Tensor input_buf,
        torch::Tensor local_expert_count,
        torch::Tensor global_expert_count,
        long batch_size, long n_workers, long shard_start, long shard_end) {
    CHECK_INPUT(input_buf);

    auto n_expert = local_expert_count.size(0) / n_workers;
    auto in_feat = input_buf.size(1);
    
    auto smgr = getCudaStreamManager(input_buf.device().index());
    if(shard_start == -1){
        shard_start = 0;
        shard_end = n_expert;
    }
    std::vector<long> expert_ptr;
    std::vector<long> recv_expert_ptr;

    expert_ptr.emplace_back(0);
    recv_expert_ptr.emplace_back(0);

    for (size_t j = 0;j < n_workers;++j){
        for (size_t i = shard_start;i < shard_end;i++){
            size_t cnt = j * n_expert + i;
            expert_ptr.emplace_back(expert_ptr.back() + local_expert_count[cnt].item<long>());
        }
    }

    for (size_t i = shard_start;i < shard_end;i++){
        for (size_t j = 0;j < n_workers;j++){
            size_t cnt = j * n_expert + i;
            recv_expert_ptr.emplace_back(recv_expert_ptr.back() + global_expert_count[cnt].item<long>());
        }
    }

    auto global_input_buf = input_buf.new_empty({recv_expert_ptr.back(), in_feat});
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_buf.scalar_type(),
            "mole_cuda_global_scatter", ([&] {
        mole_cuda_global_scatter_impl<scalar_t>(
            input_buf.data_ptr<scalar_t>(),
            local_expert_count.data_ptr<long>(),
            global_expert_count.data_ptr<long>(),
            global_input_buf.data_ptr<scalar_t>(),
            in_feat, n_expert, n_workers,
            smgr, shard_start, shard_end, expert_ptr.data(), recv_expert_ptr.data()
        );
    }));
    return global_input_buf;
}

torch::Tensor _mole_global_gather(
        torch::Tensor output_buf,
        torch::Tensor local_expert_count,
        torch::Tensor global_expert_count,
        long batch_size, long n_workers, long shard_start, long shard_end) {
    CHECK_INPUT(output_buf);

    auto n_expert = local_expert_count.size(0) / n_workers;
    auto out_feat = output_buf.size(1);
    
    auto smgr = getCudaStreamManager(output_buf.device().index());
    if(shard_start == -1){
        shard_start = 0;
        shard_end = n_expert;
    }
    std::vector<long> expert_ptr;
    std::vector<long> send_expert_ptr;

    expert_ptr.emplace_back(0);
    send_expert_ptr.emplace_back(0);

    for (size_t j = 0;j < n_workers;++j){
        for (size_t i = shard_start;i < shard_end;i++){
            size_t cnt = j * n_expert + i;
            expert_ptr.emplace_back(expert_ptr.back() + local_expert_count[cnt].item<long>());
        }
    }

    for (size_t i = shard_start;i < shard_end;i++){
        for (size_t j = 0;j < n_workers;j++){
            size_t cnt = j * n_expert + i;
            send_expert_ptr.emplace_back(send_expert_ptr.back() + global_expert_count[cnt].item<long>());
        }
    }

    auto local_output_buf = output_buf.new_empty({expert_ptr.back(), out_feat});
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(output_buf.scalar_type(),
            "mole_cuda_global_gather", ([&] {
        mole_cuda_global_gather_impl<scalar_t>(
            output_buf.data_ptr<scalar_t>(),
            local_expert_count.data_ptr<long>(),
            global_expert_count.data_ptr<long>(),
            local_output_buf.data_ptr<scalar_t>(),
            out_feat, n_expert, n_workers,
            smgr, shard_start, shard_end, expert_ptr.data(), send_expert_ptr.data()
        );
    }));
    return local_output_buf;
}


#if defined(TORCH_VERSION_MAJOR) && (TORCH_VERSION_MAJOR > 1 || \
        (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 13))
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#else
#include <c10d/ProcessGroupNCCL.hpp>
#endif

class HackNCCLGroup: public c10d::ProcessGroupNCCL {
public:
    ncclComm_t getcomm(at::Device dev) {
        ncclUniqueId ncclID;
        int rank = getRank();
        if (rank == 0) {
            ncclGetUniqueId(&ncclID);
        }
#if defined(TORCH_VERSION_MAJOR) && (TORCH_VERSION_MAJOR > 1 || \
        (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 12))
        broadcastUniqueNCCLID(&ncclID,
                false,
                "fastmoe_nccl_comm",
                rank);
#elif defined(TORCH_VERSION_MAJOR) && (TORCH_VERSION_MAJOR > 1 || \
        (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 8))
        broadcastUniqueNCCLID(&ncclID,
                c10d::OpType::SEND,
                "fastmoe_nccl_comm",
                rank);
#else
        broadcastUniqueNCCLID(&ncclID);
#endif
        ncclComm_t comm;
        NCCL_SAFE_CALL(ncclCommInitRank(&comm, getSize(), ncclID, rank));
        return comm;
    }
};

void _ensure_nccl(c10d::ProcessGroupNCCL& p, torch::Tensor t) {
    auto smgr = getCudaStreamManager(t.device().index());
    if (smgr->ncclgood) {
        return;
    }
    HackNCCLGroup* h = (HackNCCLGroup*)(void*)&p;
    smgr->ncclcomm = h->getcomm(t.device());
    if (smgr->ncclcomm != 0) {
        smgr->ncclgood = 1;
    } else {
        std::cerr << "Nccl initialization failed\n";
    }
}

#endif  // FMOE_USE_NCCL
