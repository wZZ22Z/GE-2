#pragma once

#include "common/datatypes.h"

enum class SelectedNegScoreKind : int64_t { NONE = 0, DOT = 1, L2 = 2 };

torch::Tensor selected_neg_scores(torch::Tensor chunked_adjusted_embeddings,
                                  torch::Tensor negative_embeddings,
                                  torch::Tensor selected_neg_indices,
                                  SelectedNegScoreKind score_kind);

std::tuple<torch::Tensor, torch::Tensor> distmult_tiled_tournament_selected_scores(torch::Tensor selector_chunked_embeddings,
                                                                                    torch::Tensor selector_negative_embeddings,
                                                                                    torch::Tensor score_chunked_embeddings,
                                                                                    torch::Tensor score_negative_embeddings,
                                                                                    int64_t selected_negatives_num,
                                                                                    int64_t groups_per_tile);

torch::Tensor selected_neg_scores_cuda_forward(torch::Tensor chunked_adjusted_embeddings,
                                               torch::Tensor negative_embeddings,
                                               torch::Tensor selected_neg_indices,
                                               int64_t score_kind);

std::tuple<torch::Tensor, torch::Tensor> selected_neg_scores_cuda_backward(torch::Tensor grad_out,
                                                                           torch::Tensor chunked_adjusted_embeddings,
                                                                           torch::Tensor negative_embeddings,
                                                                           torch::Tensor selected_neg_indices,
                                                                           int64_t score_kind);
