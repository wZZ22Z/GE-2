#include <torch/script.h>

#include <algorithm>

#include "nn/decoders/edge/distmult_selected_neg_cuda.h"

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

namespace {

class SelectedNegScores : public torch::autograd::Function<SelectedNegScores> {
   public:
    static variable_list forward(AutogradContext *ctx,
                                 Variable chunked_adjusted_embeddings,
                                 Variable negative_embeddings,
                                 Variable selected_neg_indices,
                                 int64_t score_kind) {
        auto out = selected_neg_scores_cuda_forward(chunked_adjusted_embeddings, negative_embeddings, selected_neg_indices, score_kind);
        ctx->save_for_backward({chunked_adjusted_embeddings, negative_embeddings, selected_neg_indices});
        ctx->saved_data["score_kind"] = score_kind;
        return {out};
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
        auto grad_out = grad_outs[0];
        auto saved = ctx->get_saved_variables();
        auto score_kind = ctx->saved_data["score_kind"].toInt();
        auto grads = selected_neg_scores_cuda_backward(grad_out, saved[0], saved[1], saved[2], score_kind);
        return {std::get<0>(grads), std::get<1>(grads), Variable(), Variable()};
    }
};

}  // namespace

torch::Tensor selected_neg_scores(torch::Tensor chunked_adjusted_embeddings,
                                  torch::Tensor negative_embeddings,
                                  torch::Tensor selected_neg_indices,
                                  SelectedNegScoreKind score_kind) {
    auto result = SelectedNegScores::apply(
        chunked_adjusted_embeddings, negative_embeddings, selected_neg_indices, static_cast<int64_t>(score_kind));
    return result[0];
}

std::tuple<torch::Tensor, torch::Tensor> distmult_tiled_tournament_selected_scores(torch::Tensor selector_chunked_embeddings,
                                                                                    torch::Tensor selector_negative_embeddings,
                                                                                    torch::Tensor score_chunked_embeddings,
                                                                                    torch::Tensor score_negative_embeddings,
                                                                                    int64_t selected_negatives_num,
                                                                                    int64_t groups_per_tile) {
    TORCH_CHECK(selector_chunked_embeddings.defined(), "selector_chunked_embeddings must be defined");
    TORCH_CHECK(selector_negative_embeddings.defined(), "selector_negative_embeddings must be defined");
    TORCH_CHECK(score_chunked_embeddings.defined(), "score_chunked_embeddings must be defined");
    TORCH_CHECK(score_negative_embeddings.defined(), "score_negative_embeddings must be defined");
    TORCH_CHECK(selector_chunked_embeddings.dim() == 3, "selector_chunked_embeddings must be rank 3");
    TORCH_CHECK(selector_negative_embeddings.dim() == 3, "selector_negative_embeddings must be rank 3");
    TORCH_CHECK(score_chunked_embeddings.dim() == 3, "score_chunked_embeddings must be rank 3");
    TORCH_CHECK(score_negative_embeddings.dim() == 3, "score_negative_embeddings must be rank 3");
    TORCH_CHECK(selector_chunked_embeddings.sizes() == score_chunked_embeddings.sizes(), "selector and score chunked embeddings must match");
    TORCH_CHECK(selector_negative_embeddings.sizes() == score_negative_embeddings.sizes(), "selector and score negative embeddings must match");
    TORCH_CHECK(selector_chunked_embeddings.size(0) == selector_negative_embeddings.size(0), "chunk counts must match");
    TORCH_CHECK(selector_chunked_embeddings.size(2) == selector_negative_embeddings.size(2), "embedding dimensions must match");
    TORCH_CHECK(selected_negatives_num > 0, "selected_negatives_num must be positive");

    auto selector_chunked = selector_chunked_embeddings.contiguous();
    auto selector_negatives = selector_negative_embeddings.contiguous();
    auto score_chunked = score_chunked_embeddings.contiguous();
    auto score_negatives = score_negative_embeddings.contiguous();

    int64_t chunk_num = score_chunked.size(0);
    int64_t num_per_chunk = score_chunked.size(1);
    int64_t negatives_num = score_negatives.size(1);
    TORCH_CHECK(negatives_num % selected_negatives_num == 0, "negatives_num must be divisible by selected_negatives_num");

    int64_t tournament_size = negatives_num / selected_negatives_num;
    int64_t tile_groups = std::max<int64_t>(1, std::min<int64_t>(groups_per_tile, selected_negatives_num));
    auto index_opts = torch::TensorOptions().dtype(torch::kInt64).device(score_chunked.device());

    std::vector<torch::Tensor> score_tiles;
    std::vector<torch::Tensor> index_tiles;
    score_tiles.reserve((selected_negatives_num + tile_groups - 1) / tile_groups);
    index_tiles.reserve(score_tiles.capacity());

    bool same_query = selector_chunked.data_ptr() == score_chunked.data_ptr();
    bool same_neg = selector_negatives.data_ptr() == score_negatives.data_ptr();

    for (int64_t group_start = 0; group_start < selected_negatives_num; group_start += tile_groups) {
        int64_t groups = std::min<int64_t>(tile_groups, selected_negatives_num - group_start);
        int64_t neg_start = group_start * tournament_size;
        int64_t neg_count = groups * tournament_size;

        torch::Tensor score_neg_tile = score_negatives.narrow(1, neg_start, neg_count);
        torch::Tensor score_tile = score_chunked.bmm(score_neg_tile.transpose(1, 2));

        torch::Tensor selector_tile;
        if (same_query && same_neg) {
            selector_tile = score_tile.detach();
        } else {
            torch::Tensor selector_neg_tile = selector_negatives.narrow(1, neg_start, neg_count);
            selector_tile = selector_chunked.bmm(selector_neg_tile.transpose(1, 2)).detach();
        }

        torch::Tensor selector_view = selector_tile.view({chunk_num, num_per_chunk, groups, tournament_size});
        auto max_result = selector_view.max(3);
        torch::Tensor local_indices = std::get<1>(max_result).to(torch::kInt64);

        torch::Tensor score_view = score_tile.view({chunk_num, num_per_chunk, groups, tournament_size});
        torch::Tensor selected_scores = score_view.gather(3, local_indices.unsqueeze(3)).squeeze(3);

        torch::Tensor offsets = torch::arange(groups, index_opts).view({1, 1, groups}) * tournament_size + neg_start;
        torch::Tensor selected_indices = local_indices + offsets;

        score_tiles.push_back(selected_scores);
        index_tiles.push_back(selected_indices);
    }

    torch::Tensor all_scores = torch::cat(score_tiles, 2).reshape({chunk_num * num_per_chunk, selected_negatives_num});
    torch::Tensor all_indices = torch::cat(index_tiles, 2).reshape({chunk_num, num_per_chunk * selected_negatives_num});
    return std::forward_as_tuple(all_scores, all_indices);
}
