#include "nn/decoders/edge/distmult_selected_neg_cuda.h"

#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>

#include "pytorch_scatter/atomics.cuh"

namespace {

constexpr double kSelectedNegL2Tol = 1e-8;

template <typename scalar_t>
__global__ void selected_neg_dot_scores_kernel(const scalar_t* __restrict__ chunked_adjusted_embeddings,
                                               const scalar_t* __restrict__ negative_embeddings,
                                               const int64_t* __restrict__ selected_neg_indices,
                                               scalar_t* __restrict__ output,
                                               int64_t chunk_num,
                                               int64_t num_per_chunk,
                                               int64_t negatives_num,
                                               int64_t selected_negatives_num,
                                               int64_t embedding_dim) {
    using acc_t = typename at::acc_type<scalar_t, true>;

    int64_t linear_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total_scores = chunk_num * num_per_chunk * selected_negatives_num;
    if (linear_idx >= total_scores) {
        return;
    }

    int64_t selected_offset = linear_idx % selected_negatives_num;
    int64_t row_idx = linear_idx / selected_negatives_num;
    int64_t chunk_idx = row_idx / num_per_chunk;
    int64_t row_in_chunk = row_idx % num_per_chunk;
    int64_t selected_index_offset = chunk_idx * (num_per_chunk * selected_negatives_num) + row_in_chunk * selected_negatives_num + selected_offset;
    int64_t negative_idx = selected_neg_indices[selected_index_offset];

    const scalar_t* src_ptr = chunked_adjusted_embeddings + ((chunk_idx * num_per_chunk + row_in_chunk) * embedding_dim);
    const scalar_t* neg_ptr = negative_embeddings + ((chunk_idx * negatives_num + negative_idx) * embedding_dim);

    acc_t acc = 0;
    for (int64_t d = 0; d < embedding_dim; d++) {
        acc += static_cast<acc_t>(src_ptr[d]) * static_cast<acc_t>(neg_ptr[d]);
    }

    output[linear_idx] = static_cast<scalar_t>(acc);
}

template <typename scalar_t>
__global__ void selected_neg_l2_scores_kernel(const scalar_t* __restrict__ chunked_adjusted_embeddings,
                                              const scalar_t* __restrict__ negative_embeddings,
                                              const int64_t* __restrict__ selected_neg_indices,
                                              scalar_t* __restrict__ output,
                                              int64_t chunk_num,
                                              int64_t num_per_chunk,
                                              int64_t negatives_num,
                                              int64_t selected_negatives_num,
                                              int64_t embedding_dim) {
    using acc_t = typename at::acc_type<scalar_t, true>;

    int64_t linear_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total_scores = chunk_num * num_per_chunk * selected_negatives_num;
    if (linear_idx >= total_scores) {
        return;
    }

    int64_t selected_offset = linear_idx % selected_negatives_num;
    int64_t row_idx = linear_idx / selected_negatives_num;
    int64_t chunk_idx = row_idx / num_per_chunk;
    int64_t row_in_chunk = row_idx % num_per_chunk;
    int64_t selected_index_offset = chunk_idx * (num_per_chunk * selected_negatives_num) + row_in_chunk * selected_negatives_num + selected_offset;
    int64_t negative_idx = selected_neg_indices[selected_index_offset];

    const scalar_t* src_ptr = chunked_adjusted_embeddings + ((chunk_idx * num_per_chunk + row_in_chunk) * embedding_dim);
    const scalar_t* neg_ptr = negative_embeddings + ((chunk_idx * negatives_num + negative_idx) * embedding_dim);

    acc_t sq = 0;
    for (int64_t d = 0; d < embedding_dim; d++) {
        acc_t diff = static_cast<acc_t>(src_ptr[d]) - static_cast<acc_t>(neg_ptr[d]);
        sq += diff * diff;
    }

    output[linear_idx] = static_cast<scalar_t>(sqrt(max(sq, static_cast<acc_t>(kSelectedNegL2Tol))));
}

template <typename scalar_t>
__global__ void selected_neg_dot_grad_chunked_kernel(const scalar_t* __restrict__ grad_out,
                                                     const scalar_t* __restrict__ negative_embeddings,
                                                     const int64_t* __restrict__ selected_neg_indices,
                                                     scalar_t* __restrict__ grad_chunked_adjusted_embeddings,
                                                     int64_t chunk_num,
                                                     int64_t num_per_chunk,
                                                     int64_t negatives_num,
                                                     int64_t selected_negatives_num,
                                                     int64_t embedding_dim) {
    using acc_t = typename at::acc_type<scalar_t, true>;

    int64_t linear_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = chunk_num * num_per_chunk * embedding_dim;
    if (linear_idx >= total) {
        return;
    }

    int64_t dim_idx = linear_idx % embedding_dim;
    int64_t row_idx = linear_idx / embedding_dim;
    int64_t chunk_idx = row_idx / num_per_chunk;
    int64_t row_in_chunk = row_idx % num_per_chunk;
    int64_t selected_base = chunk_idx * (num_per_chunk * selected_negatives_num) + row_in_chunk * selected_negatives_num;
    int64_t grad_base = row_idx * selected_negatives_num;

    acc_t acc = 0;
    for (int64_t selected_offset = 0; selected_offset < selected_negatives_num; selected_offset++) {
        int64_t negative_idx = selected_neg_indices[selected_base + selected_offset];
        int64_t negative_offset = ((chunk_idx * negatives_num + negative_idx) * embedding_dim) + dim_idx;
        acc += static_cast<acc_t>(grad_out[grad_base + selected_offset]) * static_cast<acc_t>(negative_embeddings[negative_offset]);
    }

    grad_chunked_adjusted_embeddings[linear_idx] = static_cast<scalar_t>(acc);
}

template <typename scalar_t>
__global__ void selected_neg_dot_grad_negative_kernel(const scalar_t* __restrict__ grad_out,
                                                      const scalar_t* __restrict__ chunked_adjusted_embeddings,
                                                      const int64_t* __restrict__ selected_neg_indices,
                                                      scalar_t* __restrict__ grad_negative_embeddings,
                                                      int64_t chunk_num,
                                                      int64_t num_per_chunk,
                                                      int64_t negatives_num,
                                                      int64_t selected_negatives_num,
                                                      int64_t embedding_dim) {
    using acc_t = typename at::acc_type<scalar_t, true>;

    int64_t linear_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = chunk_num * num_per_chunk * selected_negatives_num * embedding_dim;
    if (linear_idx >= total) {
        return;
    }

    int64_t dim_idx = linear_idx % embedding_dim;
    int64_t selected_linear = linear_idx / embedding_dim;
    int64_t selected_offset = selected_linear % selected_negatives_num;
    int64_t row_idx = selected_linear / selected_negatives_num;
    int64_t chunk_idx = row_idx / num_per_chunk;
    int64_t row_in_chunk = row_idx % num_per_chunk;
    int64_t selected_index_offset = chunk_idx * (num_per_chunk * selected_negatives_num) + row_in_chunk * selected_negatives_num + selected_offset;
    int64_t negative_idx = selected_neg_indices[selected_index_offset];

    int64_t grad_out_offset = row_idx * selected_negatives_num + selected_offset;
    int64_t chunked_offset = (row_idx * embedding_dim) + dim_idx;
    int64_t grad_negative_offset = ((chunk_idx * negatives_num + negative_idx) * embedding_dim) + dim_idx;
    acc_t grad_value = static_cast<acc_t>(grad_out[grad_out_offset]) * static_cast<acc_t>(chunked_adjusted_embeddings[chunked_offset]);
    atomAdd(grad_negative_embeddings + grad_negative_offset, static_cast<scalar_t>(grad_value));
}

template <typename scalar_t>
__global__ void selected_neg_l2_backward_pair_kernel(const scalar_t* __restrict__ grad_out,
                                                     const scalar_t* __restrict__ chunked_adjusted_embeddings,
                                                     const scalar_t* __restrict__ negative_embeddings,
                                                     const int64_t* __restrict__ selected_neg_indices,
                                                     scalar_t* __restrict__ grad_chunked_adjusted_embeddings,
                                                     scalar_t* __restrict__ grad_negative_embeddings,
                                                     int64_t chunk_num,
                                                     int64_t num_per_chunk,
                                                     int64_t negatives_num,
                                                     int64_t selected_negatives_num,
                                                     int64_t embedding_dim) {
    using acc_t = typename at::acc_type<scalar_t, true>;

    int64_t linear_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total_pairs = chunk_num * num_per_chunk * selected_negatives_num;
    if (linear_idx >= total_pairs) {
        return;
    }

    int64_t selected_offset = linear_idx % selected_negatives_num;
    int64_t row_idx = linear_idx / selected_negatives_num;
    int64_t chunk_idx = row_idx / num_per_chunk;
    int64_t row_in_chunk = row_idx % num_per_chunk;
    int64_t selected_index_offset = chunk_idx * (num_per_chunk * selected_negatives_num) + row_in_chunk * selected_negatives_num + selected_offset;
    int64_t negative_idx = selected_neg_indices[selected_index_offset];
    const scalar_t* src_ptr = chunked_adjusted_embeddings + (row_idx * embedding_dim);
    const scalar_t* neg_ptr = negative_embeddings + ((chunk_idx * negatives_num + negative_idx) * embedding_dim);

    acc_t sq = 0;
    for (int64_t d = 0; d < embedding_dim; d++) {
        acc_t diff = static_cast<acc_t>(src_ptr[d]) - static_cast<acc_t>(neg_ptr[d]);
        sq += diff * diff;
    }

    if (sq <= static_cast<acc_t>(kSelectedNegL2Tol)) {
        return;
    }

    acc_t coeff = static_cast<acc_t>(grad_out[linear_idx]) / sqrt(sq);
    for (int64_t d = 0; d < embedding_dim; d++) {
        acc_t diff = static_cast<acc_t>(src_ptr[d]) - static_cast<acc_t>(neg_ptr[d]);
        scalar_t grad_value = static_cast<scalar_t>(coeff * diff);
        atomAdd(grad_chunked_adjusted_embeddings + (row_idx * embedding_dim) + d, grad_value);
        atomAdd(grad_negative_embeddings + ((chunk_idx * negatives_num + negative_idx) * embedding_dim) + d, static_cast<scalar_t>(-coeff * diff));
    }
}

}  // namespace

torch::Tensor selected_neg_scores_cuda_forward(torch::Tensor chunked_adjusted_embeddings,
                                               torch::Tensor negative_embeddings,
                                               torch::Tensor selected_neg_indices,
                                               int64_t score_kind) {
    TORCH_CHECK(chunked_adjusted_embeddings.is_cuda(), "chunked_adjusted_embeddings must be CUDA");
    TORCH_CHECK(negative_embeddings.is_cuda(), "negative_embeddings must be CUDA");
    TORCH_CHECK(selected_neg_indices.is_cuda(), "selected_neg_indices must be CUDA");
    TORCH_CHECK(chunked_adjusted_embeddings.dim() == 3, "chunked_adjusted_embeddings must be rank 3");
    TORCH_CHECK(negative_embeddings.dim() == 3, "negative_embeddings must be rank 3");
    TORCH_CHECK(selected_neg_indices.dim() == 2, "selected_neg_indices must be rank 2");
    TORCH_CHECK(chunked_adjusted_embeddings.size(0) == negative_embeddings.size(0), "chunk and negative chunk counts must match");
    TORCH_CHECK(chunked_adjusted_embeddings.size(0) == selected_neg_indices.size(0), "chunk and selected index chunk counts must match");
    TORCH_CHECK(chunked_adjusted_embeddings.size(2) == negative_embeddings.size(2), "embedding dimensions must match");
    TORCH_CHECK(selected_neg_indices.scalar_type() == torch::kInt64, "selected_neg_indices must be int64");
    TORCH_CHECK(score_kind == static_cast<int64_t>(SelectedNegScoreKind::DOT) || score_kind == static_cast<int64_t>(SelectedNegScoreKind::L2),
                "score_kind must be DOT or L2");

    c10::cuda::CUDAGuard device_guard(chunked_adjusted_embeddings.device());

    auto chunked = chunked_adjusted_embeddings.contiguous();
    auto negatives = negative_embeddings.contiguous();
    auto selected = selected_neg_indices.contiguous();

    int64_t chunk_num = chunked.size(0);
    int64_t num_per_chunk = chunked.size(1);
    int64_t embedding_dim = chunked.size(2);
    int64_t negatives_num = negatives.size(1);
    int64_t selected_negatives_num = selected.size(1) / std::max<int64_t>(num_per_chunk, 1);
    TORCH_CHECK(selected.size(1) == num_per_chunk * selected_negatives_num, "selected_neg_indices width must be num_per_chunk * selected_negatives_num");

    auto output = torch::empty({chunk_num * num_per_chunk, selected_negatives_num}, chunked.options());

    int64_t total_scores = output.numel();
    constexpr int threads = 256;
    int blocks = static_cast<int>((total_scores + threads - 1) / threads);
    if (total_scores == 0) {
        return output;
    }

    auto stream = at::cuda::getCurrentCUDAStream(chunked.device().index()).stream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(chunked.scalar_type(), "selected_neg_scores_cuda_forward", [&] {
        if (score_kind == static_cast<int64_t>(SelectedNegScoreKind::DOT)) {
            selected_neg_dot_scores_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                chunked.data_ptr<scalar_t>(),
                negatives.data_ptr<scalar_t>(),
                selected.data_ptr<int64_t>(),
                output.data_ptr<scalar_t>(),
                chunk_num,
                num_per_chunk,
                negatives_num,
                selected_negatives_num,
                embedding_dim);
        } else {
            selected_neg_l2_scores_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                chunked.data_ptr<scalar_t>(),
                negatives.data_ptr<scalar_t>(),
                selected.data_ptr<int64_t>(),
                output.data_ptr<scalar_t>(),
                chunk_num,
                num_per_chunk,
                negatives_num,
                selected_negatives_num,
                embedding_dim);
        }
    });

    AT_CUDA_CHECK(cudaGetLastError());
    return output;
}

std::tuple<torch::Tensor, torch::Tensor> selected_neg_scores_cuda_backward(torch::Tensor grad_out,
                                                                           torch::Tensor chunked_adjusted_embeddings,
                                                                           torch::Tensor negative_embeddings,
                                                                           torch::Tensor selected_neg_indices,
                                                                           int64_t score_kind) {
    TORCH_CHECK(grad_out.is_cuda(), "grad_out must be CUDA");
    TORCH_CHECK(chunked_adjusted_embeddings.is_cuda(), "chunked_adjusted_embeddings must be CUDA");
    TORCH_CHECK(negative_embeddings.is_cuda(), "negative_embeddings must be CUDA");
    TORCH_CHECK(selected_neg_indices.is_cuda(), "selected_neg_indices must be CUDA");
    TORCH_CHECK(grad_out.dim() == 2, "grad_out must be rank 2");
    TORCH_CHECK(chunked_adjusted_embeddings.dim() == 3, "chunked_adjusted_embeddings must be rank 3");
    TORCH_CHECK(negative_embeddings.dim() == 3, "negative_embeddings must be rank 3");
    TORCH_CHECK(selected_neg_indices.dim() == 2, "selected_neg_indices must be rank 2");
    TORCH_CHECK(selected_neg_indices.scalar_type() == torch::kInt64, "selected_neg_indices must be int64");
    TORCH_CHECK(score_kind == static_cast<int64_t>(SelectedNegScoreKind::DOT) || score_kind == static_cast<int64_t>(SelectedNegScoreKind::L2),
                "score_kind must be DOT or L2");

    c10::cuda::CUDAGuard device_guard(chunked_adjusted_embeddings.device());

    auto grad = grad_out.contiguous();
    auto chunked = chunked_adjusted_embeddings.contiguous();
    auto negatives = negative_embeddings.contiguous();
    auto selected = selected_neg_indices.contiguous();

    int64_t chunk_num = chunked.size(0);
    int64_t num_per_chunk = chunked.size(1);
    int64_t embedding_dim = chunked.size(2);
    int64_t negatives_num = negatives.size(1);
    int64_t selected_negatives_num = grad.size(1);
    TORCH_CHECK(grad.size(0) == chunk_num * num_per_chunk, "grad_out leading dimension must equal chunk_num * num_per_chunk");
    TORCH_CHECK(selected.size(1) == num_per_chunk * selected_negatives_num, "selected_neg_indices width must be num_per_chunk * selected_negatives_num");

    auto grad_chunked = torch::zeros_like(chunked);
    auto grad_negatives = torch::zeros_like(negatives);
    if (grad.numel() == 0) {
        return std::forward_as_tuple(grad_chunked, grad_negatives);
    }

    auto stream = at::cuda::getCurrentCUDAStream(chunked.device().index()).stream();
    constexpr int threads = 256;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(chunked.scalar_type(), "selected_neg_scores_cuda_backward", [&] {
        if (score_kind == static_cast<int64_t>(SelectedNegScoreKind::DOT)) {
            int grad_chunked_blocks = static_cast<int>((grad_chunked.numel() + threads - 1) / threads);
            int grad_negative_blocks = static_cast<int>(((chunk_num * num_per_chunk * selected_negatives_num * embedding_dim) + threads - 1) / threads);

            selected_neg_dot_grad_chunked_kernel<scalar_t><<<grad_chunked_blocks, threads, 0, stream>>>(
                grad.data_ptr<scalar_t>(),
                negatives.data_ptr<scalar_t>(),
                selected.data_ptr<int64_t>(),
                grad_chunked.data_ptr<scalar_t>(),
                chunk_num,
                num_per_chunk,
                negatives_num,
                selected_negatives_num,
                embedding_dim);

            selected_neg_dot_grad_negative_kernel<scalar_t><<<grad_negative_blocks, threads, 0, stream>>>(
                grad.data_ptr<scalar_t>(),
                chunked.data_ptr<scalar_t>(),
                selected.data_ptr<int64_t>(),
                grad_negatives.data_ptr<scalar_t>(),
                chunk_num,
                num_per_chunk,
                negatives_num,
                selected_negatives_num,
                embedding_dim);
        } else {
            int grad_pair_blocks = static_cast<int>(((chunk_num * num_per_chunk * selected_negatives_num) + threads - 1) / threads);
            selected_neg_l2_backward_pair_kernel<scalar_t><<<grad_pair_blocks, threads, 0, stream>>>(
                grad.data_ptr<scalar_t>(),
                chunked.data_ptr<scalar_t>(),
                negatives.data_ptr<scalar_t>(),
                selected.data_ptr<int64_t>(),
                grad_chunked.data_ptr<scalar_t>(),
                grad_negatives.data_ptr<scalar_t>(),
                chunk_num,
                num_per_chunk,
                negatives_num,
                selected_negatives_num,
                embedding_dim);
        }
    });

    AT_CUDA_CHECK(cudaGetLastError());
    return std::forward_as_tuple(grad_chunked, grad_negatives);
}
