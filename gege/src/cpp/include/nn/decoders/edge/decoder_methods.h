#pragma once

#include "common/datatypes.h"
#include "configuration/options.h"
#include "data/samplers/negative.h"
#include "nn/decoders/edge/edge_decoder.h"

std::tuple<torch::Tensor, torch::Tensor> only_pos_forward(shared_ptr<EdgeDecoder> decoder, torch::Tensor positive_edges, torch::Tensor node_embeddings);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> neg_and_pos_forward(shared_ptr<EdgeDecoder> decoder, torch::Tensor positive_edges,
                                                                                           torch::Tensor negative_edges, torch::Tensor node_embeddings);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> node_corrupt_forward(shared_ptr<EdgeDecoder> decoder, torch::Tensor positive_edges,
                                                                                            torch::Tensor node_embeddings, torch::Tensor dst_negs,
                                                                                            torch::Tensor src_negs = torch::Tensor());

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> rel_corrupt_forward(shared_ptr<EdgeDecoder> decoder, torch::Tensor positive_edges,
                                                                                           torch::Tensor node_embeddings, torch::Tensor neg_rel_ids);


std::tuple<torch::Tensor, torch::Tensor> prepare_pos_embeddings(shared_ptr<EdgeDecoder> decoder, torch::Tensor positive_edges, torch::Tensor src_embeddings, torch::Tensor dst_embeddings, bool has_relations);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> mod_node_corrupt_forward(NegativeSamplingMethod negative_sampling_method, float negative_sampling_selected_ratio,
                                                                                                shared_ptr<NegativeSampler> negative_sampler, shared_ptr<EdgeDecoder> decoder, torch::Tensor positive_edges,
                                                                                                torch::Tensor node_embeddings, torch::Tensor dst_negs, torch::Tensor src_negs,
                                                                                                torch::Tensor node_embeddings_g);

std::tuple<torch::Tensor, torch::Tensor> get_rewards(shared_ptr<EdgeDecoder> decoder, torch::Tensor positive_edges, torch::Tensor node_embeddings, torch::Tensor dst_negs, torch::Tensor src_negs);

torch::Tensor forward_g(shared_ptr<EdgeDecoder> decoder, torch::Tensor positive_edges, torch::Tensor node_embeddings_g, torch::Tensor dst_negs, torch::Tensor src_negs, torch::Tensor reward, torch::Tensor inv_reward);
