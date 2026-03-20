#include <torch/script.h>

#include "segment_csr_cuda.h"

inline std::vector<int64_t> list2vec(const c10::List<int64_t> list) {
    std::vector<int64_t> result;
    result.reserve(list.size());
    for (size_t i = 0; i < list.size(); i++) {
        result.push_back(list[i]);
    }
    return result;
}

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class SegmentSumCSR : public torch::autograd::Function<SegmentSumCSR> {
   public:
    static variable_list forward(AutogradContext *ctx, Variable src, Variable indptr, torch::optional<Variable> optional_out) {
        ctx->saved_data["src_shape"] = src.sizes();
        auto result = segment_csr_cuda(src, indptr, optional_out, "sum");
        auto out = std::get<0>(result);
        ctx->save_for_backward({indptr});
        if (optional_out.has_value()) {
            ctx->mark_dirty({optional_out.value()});
        }
        return {out};
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
        auto grad_out = grad_outs[0];
        auto saved = ctx->get_saved_variables();
        auto indptr = saved[0];
        auto src_shape = list2vec(ctx->saved_data["src_shape"].toIntList());
        auto grad_in = gather_csr_cuda(grad_out, indptr, torch::nullopt);
        if (grad_in.sizes().vec() != src_shape) {
            grad_in = grad_in.view(src_shape);
        }
        return {grad_in, Variable(), Variable()};
    }
};

class GatherCSR : public torch::autograd::Function<GatherCSR> {
   public:
    static variable_list forward(AutogradContext *ctx, Variable src, Variable indptr, torch::optional<Variable> optional_out) {
        ctx->saved_data["src_shape"] = src.sizes();
        auto out = gather_csr_cuda(src, indptr, optional_out);
        ctx->save_for_backward({indptr});
        if (optional_out.has_value()) {
            ctx->mark_dirty({optional_out.value()});
        }
        return {out};
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
        auto grad_out = grad_outs[0];
        auto saved = ctx->get_saved_variables();
        auto indptr = saved[0];
        auto src_shape = list2vec(ctx->saved_data["src_shape"].toIntList());
        auto result = segment_csr_cuda(grad_out, indptr, torch::nullopt, "sum");
        auto grad_in = std::get<0>(result);
        if (grad_in.sizes().vec() != src_shape) {
            grad_in = grad_in.view(src_shape);
        }
        return {grad_in, Variable(), Variable()};
    }
};

torch::Tensor segment_sum_csr(torch::Tensor src, torch::Tensor indptr, torch::optional<torch::Tensor> optional_out) {
    auto result = SegmentSumCSR::apply(src, indptr, optional_out);
    return result[0];
}

torch::Tensor gather_csr(torch::Tensor src, torch::Tensor indptr, torch::optional<torch::Tensor> optional_out) {
    auto result = GatherCSR::apply(src, indptr, optional_out);
    return result[0];
}
