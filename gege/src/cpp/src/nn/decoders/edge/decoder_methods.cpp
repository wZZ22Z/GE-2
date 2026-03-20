#include "nn/decoders/edge/decoder_methods.h"

#include "common/util.h"
#include "configuration/options.h"
#include "nn/decoders/edge/comparators.h"
#include "nn/decoders/edge/distmult.h"
#include "nn/decoders/edge/distmult_selected_neg_cuda.h"
#include "reporting/logger.h"

#ifdef GEGE_CUDA
#include "pytorch_scatter/segment_sum.h"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

namespace {

std::string tensor_shape_string(const torch::Tensor &tensor) {
    std::ostringstream out;
    out << "[";
    for (int64_t i = 0; i < tensor.dim(); i++) {
        if (i > 0) {
            out << ", ";
        }
        out << tensor.size(i);
    }
    out << "]";
    return out.str();
}

bool parse_env_flag(const char *name, bool default_value) {
    const char *raw = std::getenv(name);
    if (raw == nullptr) {
        return default_value;
    }

    std::string value(raw);
    if (value == "0" || value == "false" || value == "False" || value == "FALSE") {
        return false;
    }

    if (value == "1" || value == "true" || value == "True" || value == "TRUE") {
        return true;
    }

    return default_value;
}

int64_t parse_env_int(const char *name, int64_t default_value) {
    const char *raw = std::getenv(name);
    if (raw == nullptr) {
        return default_value;
    }

    try {
        return std::stoll(std::string(raw));
    } catch (...) {
        return default_value;
    }
}

std::tuple<torch::Tensor, torch::Tensor> unique_with_inverse_compat(torch::Tensor ids) {
    torch::Tensor ids64 = ids.to(torch::kInt64);
    auto sort_tup = torch::sort(ids64, 0, false);
    torch::Tensor sorted_ids = std::get<0>(sort_tup);
    torch::Tensor perm = std::get<1>(sort_tup).to(torch::kInt64);
    auto unique_tup = torch::unique_consecutive(sorted_ids, false, true);
    torch::Tensor unique_ids = std::get<0>(unique_tup);
    torch::Tensor inverse_sorted = std::get<1>(unique_tup).to(torch::kInt64);
    torch::Tensor inverse = torch::empty_like(inverse_sorted);
    inverse.scatter_(0, perm, inverse_sorted);
    return std::forward_as_tuple(unique_ids, inverse);
}

bool csr_debug_enabled() {
    static bool enabled = parse_env_flag("GEGE_CSR_DEBUG", true);
    return enabled;
}

bool csr_gather_enabled() {
    static bool enabled = parse_env_flag("GEGE_CSR_GATHER", true);
    return enabled;
}

bool stage_debug_enabled() {
    static bool enabled = parse_env_flag("GEGE_STAGE_DEBUG", false);
    return enabled;
}

bool selected_neg_cuda_enabled() {
    const char *new_name = std::getenv("GEGE_SELECTED_NEG_CUDA");
    if (new_name != nullptr) {
        return parse_env_flag("GEGE_SELECTED_NEG_CUDA", true);
    }

    static bool enabled = parse_env_flag("GEGE_DISTMULT_SELECTED_CUDA", true);
    return enabled;
}

bool tiled_tournament_scores_validate_enabled() {
    static bool enabled = parse_env_flag("GEGE_TILED_TOURNAMENT_SCORES_VALIDATE", false);
    return enabled;
}

int64_t tiled_tournament_scores_validate_max_batches() {
    static int64_t max_batches = parse_env_int("GEGE_TILED_TOURNAMENT_SCORES_VALIDATE_MAX_BATCHES", 5);
    return std::max<int64_t>(max_batches, 0);
}

std::atomic<int64_t> &tiled_tournament_scores_validate_counter() {
    static std::atomic<int64_t> counter{0};
    return counter;
}

bool transe_squared_sampling_enabled() {
    static bool enabled = parse_env_flag("GEGE_TRANSE_SQUARED_SAMPLING", true);
    return enabled;
}

int64_t stage_debug_max_batches() {
    static int64_t max_batches = parse_env_int("GEGE_STAGE_DEBUG_MAX_BATCHES", 20);
    return std::max<int64_t>(max_batches, 0);
}

std::atomic<int64_t> &stage_debug_counter() {
    static std::atomic<int64_t> counter{0};
    return counter;
}

bool should_run_stage_debug(int64_t &debug_batch_id) {
    if (!stage_debug_enabled()) {
        return false;
    }
    debug_batch_id = stage_debug_counter().fetch_add(1);
    return debug_batch_id < stage_debug_max_batches();
}

double elapsed_ms(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int64_t csr_debug_max_batches() {
    static int64_t max_batches = parse_env_int("GEGE_CSR_DEBUG_MAX_BATCHES", 5);
    return std::max<int64_t>(max_batches, 0);
}

std::atomic<int64_t> &csr_debug_counter() {
    static std::atomic<int64_t> counter{0};
    return counter;
}

bool should_run_csr_debug(int64_t &debug_batch_id) {
    if (!csr_debug_enabled()) {
        return false;
    }

    debug_batch_id = csr_debug_counter().fetch_add(1);
    return debug_batch_id < csr_debug_max_batches();
}

torch::Tensor materialize_selected_negative_embeddings(torch::Tensor negative_embeddings, torch::Tensor selected_neg_indices,
                                                       int chunk_num, int num_per_chunk, int selected_negatives_num) {
    int64_t embedding_dim = negative_embeddings.size(2);
    if (selected_neg_indices.defined()) {
        auto index_opts = torch::TensorOptions().dtype(torch::kInt64).device(selected_neg_indices.device());
        torch::Tensor indices_3d = selected_neg_indices.view({chunk_num, num_per_chunk, selected_negatives_num}).to(torch::kInt64);
        torch::Tensor chunk_offsets = torch::arange(chunk_num, index_opts).view({chunk_num, 1, 1}) * negative_embeddings.size(1);
        torch::Tensor linear_indices = (indices_3d + chunk_offsets).reshape({-1});
        return negative_embeddings.flatten(0, 1)
            .index_select(0, linear_indices)
            .view({chunk_num, num_per_chunk, selected_negatives_num, embedding_dim});
    }

    return negative_embeddings.reshape({chunk_num, num_per_chunk, selected_negatives_num, embedding_dim});
}

torch::Tensor gather_selected_negative_ids(torch::Tensor negative_ids, torch::Tensor selected_neg_indices) {
    if (!selected_neg_indices.defined()) {
        return negative_ids;
    }

    return negative_ids.gather(1, selected_neg_indices);
}

SelectedNegScoreKind selected_neg_score_kind(const shared_ptr<EdgeDecoder> &decoder) {
#ifdef GEGE_CUDA
    if (instance_of<Comparator, DotCompare>(decoder->comparator_)) {
        return SelectedNegScoreKind::DOT;
    }
    if (instance_of<Comparator, L2Compare>(decoder->comparator_)) {
        return SelectedNegScoreKind::L2;
    }
#else
    (void)decoder;
#endif
    return SelectedNegScoreKind::NONE;
}

bool sampling_prefers_higher_scores(const shared_ptr<EdgeDecoder> &decoder) {
#ifdef GEGE_CUDA
    return !instance_of<Comparator, L2Compare>(decoder->comparator_);
#else
    (void)decoder;
    return true;
#endif
}

bool negative_sampler_uses_tournament(const shared_ptr<NegativeSampler> &negative_sampler) {
    auto sampling_base = std::dynamic_pointer_cast<NegativeSamplingBase>(negative_sampler);
    return sampling_base != nullptr && (sampling_base->tournament_selection_ || negative_tournament_env_enabled());
}

bool tiled_tournament_scores_enabled(const shared_ptr<NegativeSampler> &negative_sampler) {
    const char *env_override = std::getenv("GEGE_TILED_TOURNAMENT_SCORES");
    if (env_override != nullptr) {
        return parse_env_flag("GEGE_TILED_TOURNAMENT_SCORES", false);
    }

    auto sampling_base = std::dynamic_pointer_cast<NegativeSamplingBase>(negative_sampler);
    return sampling_base != nullptr && sampling_base->tiled_tournament_scores_;
}

int64_t tiled_tournament_groups_per_tile(const shared_ptr<NegativeSampler> &negative_sampler) {
    const char *env_override = std::getenv("GEGE_TILED_TOURNAMENT_GROUPS_PER_TILE");
    if (env_override != nullptr) {
        return std::max<int64_t>(parse_env_int("GEGE_TILED_TOURNAMENT_GROUPS_PER_TILE", 64), 1);
    }

    auto sampling_base = std::dynamic_pointer_cast<NegativeSamplingBase>(negative_sampler);
    if (sampling_base == nullptr) {
        return 64;
    }
    return std::max<int64_t>(sampling_base->tiled_tournament_groups_per_tile_, 1);
}

bool can_use_tiled_distmult_tournament_scores(const shared_ptr<EdgeDecoder> &decoder,
                                              const shared_ptr<NegativeSampler> &negative_sampler,
                                              bool run_csr_debug,
                                              const torch::Tensor &score_embeddings,
                                              const torch::Tensor &negative_embeddings,
                                              int64_t selected_negatives_num) {
#ifdef GEGE_CUDA
    return tiled_tournament_scores_enabled(negative_sampler) && !run_csr_debug && std::dynamic_pointer_cast<DistMult>(decoder) != nullptr &&
           negative_sampler_uses_tournament(negative_sampler) && score_embeddings.defined() && negative_embeddings.defined() &&
           score_embeddings.is_cuda() && negative_embeddings.is_cuda() && score_embeddings.dim() == 2 && negative_embeddings.dim() == 3 &&
           selected_negatives_num > 0 && negative_embeddings.size(1) % selected_negatives_num == 0;
#else
    (void)decoder;
    (void)negative_sampler;
    (void)run_csr_debug;
    (void)score_embeddings;
    (void)negative_embeddings;
    (void)selected_negatives_num;
    return false;
#endif
}

void validate_tiled_tournament_scores_once(const torch::Tensor &selector_chunked_embeddings,
                                           const torch::Tensor &selector_negative_embeddings,
                                           const torch::Tensor &score_chunked_embeddings,
                                           const torch::Tensor &score_negative_embeddings,
                                           int64_t selected_negatives_num,
                                           const torch::Tensor &tiled_scores,
                                           const torch::Tensor &tiled_indices,
                                           const char *tag) {
#ifdef GEGE_CUDA
    if (!tiled_tournament_scores_validate_enabled()) {
        return;
    }

    int64_t validate_batch = tiled_tournament_scores_validate_counter().fetch_add(1);
    if (validate_batch >= tiled_tournament_scores_validate_max_batches()) {
        return;
    }

    int64_t chunk_num = selector_chunked_embeddings.size(0);
    int64_t num_per_chunk = selector_chunked_embeddings.size(1);
    torch::NoGradGuard no_grad;
    torch::Tensor selector_scores = selector_chunked_embeddings.bmm(selector_negative_embeddings.transpose(1, 2));
    torch::Tensor reference_indices = tournament_select_indices(selector_scores, selected_negatives_num, true).view({chunk_num, num_per_chunk * selected_negatives_num});
    torch::Tensor reference_scores =
        selected_neg_scores(score_chunked_embeddings, score_negative_embeddings, reference_indices, SelectedNegScoreKind::DOT);

    if (!torch::equal(reference_indices, tiled_indices)) {
        SPDLOG_ERROR("[tiled-tournament-scores] index validation failed for {} on validation batch {}", tag, validate_batch);
        throw GegeRuntimeException("Tiled tournament indices mismatch reference");
    }

    if (!torch::allclose(reference_scores, tiled_scores, 1e-5, 1e-5)) {
        SPDLOG_ERROR("[tiled-tournament-scores] score validation failed for {} on validation batch {}", tag, validate_batch);
        throw GegeRuntimeException("Tiled tournament scores mismatch reference");
    }

    SPDLOG_INFO("[tiled-tournament-scores] validation passed for {} on validation batch {}", tag, validate_batch);
#else
    (void)selector_chunked_embeddings;
    (void)selector_negative_embeddings;
    (void)score_chunked_embeddings;
    (void)score_negative_embeddings;
    (void)selected_negatives_num;
    (void)tiled_scores;
    (void)tiled_indices;
    (void)tag;
#endif
}

torch::Tensor reshape_sampling_scores(torch::Tensor scores, int64_t chunk_num, int64_t num_per_chunk, int64_t negatives_num) {
    if (!scores.defined()) {
        return scores;
    }

    return scores.reshape({chunk_num, num_per_chunk, negatives_num});
}

torch::Tensor l2_candidate_sampling_scores(torch::Tensor adjusted_embeddings, torch::Tensor negative_embeddings, int64_t chunk_num, int64_t num_per_chunk) {
    torch::Tensor padded_embeddings = adjusted_embeddings;
    if (num_per_chunk != adjusted_embeddings.size(0) / chunk_num) {
        int64_t new_size = num_per_chunk * chunk_num;
        torch::nn::functional::PadFuncOptions options({0, 0, 0, new_size - adjusted_embeddings.size(0)});
        padded_embeddings = torch::nn::functional::pad(adjusted_embeddings, options);
    }

    torch::Tensor src = padded_embeddings.view({chunk_num, num_per_chunk, adjusted_embeddings.size(1)});
    torch::Tensor x2 = src.pow(2).sum(2).unsqueeze(2);
    torch::Tensor y2 = negative_embeddings.pow(2).sum(2).unsqueeze(1);
    torch::Tensor xy = torch::matmul(src, negative_embeddings.transpose(1, 2));
    torch::Tensor distance2 = torch::clamp_min(x2 + y2 - 2 * xy, 1e-8);
    return -distance2;
}

torch::Tensor candidate_sampling_scores(const shared_ptr<EdgeDecoder> &decoder,
                                        torch::Tensor adjusted_embeddings,
                                        torch::Tensor negative_embeddings,
                                        int64_t chunk_num,
                                        int64_t num_per_chunk) {
    torch::Tensor scores;

    if (instance_of<Comparator, L2Compare>(decoder->comparator_)) {
        if (transe_squared_sampling_enabled()) {
            return l2_candidate_sampling_scores(adjusted_embeddings, negative_embeddings, chunk_num, num_per_chunk);
        }
        scores = reshape_sampling_scores(decoder->compute_scores(adjusted_embeddings, negative_embeddings), chunk_num, num_per_chunk, negative_embeddings.size(1));
        return -scores;
    }

    scores = reshape_sampling_scores(decoder->compute_scores(adjusted_embeddings, negative_embeddings), chunk_num, num_per_chunk, negative_embeddings.size(1));

    if (!sampling_prefers_higher_scores(decoder)) {
        scores = -scores;
    }

    return scores;
}

bool can_use_selected_neg_cuda(const shared_ptr<EdgeDecoder> &decoder, bool run_csr_debug, const torch::Tensor &adjusted_embeddings,
                               const torch::Tensor &negative_embeddings, const torch::Tensor &selected_neg_indices) {
#ifdef GEGE_CUDA
    return selected_neg_cuda_enabled() && !run_csr_debug && selected_neg_score_kind(decoder) != SelectedNegScoreKind::NONE && adjusted_embeddings.defined() &&
           negative_embeddings.defined() && selected_neg_indices.defined() && adjusted_embeddings.is_cuda() && negative_embeddings.is_cuda() &&
           selected_neg_indices.is_cuda() && adjusted_embeddings.dim() == 2 && negative_embeddings.dim() == 3 &&
           selected_neg_indices.dim() == 2 && adjusted_embeddings.scalar_type() == negative_embeddings.scalar_type() &&
           selected_neg_indices.scalar_type() == torch::kInt64;
#else
    (void)decoder;
    (void)run_csr_debug;
    (void)adjusted_embeddings;
    (void)negative_embeddings;
    (void)selected_neg_indices;
    return false;
#endif
}

#ifdef GEGE_CUDA
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> build_csr_from_local_ids(torch::Tensor local_ids, int64_t num_segments) {
    torch::Tensor ids64 = local_ids.to(torch::kInt64);
    torch::Tensor sort_ids = ids64;
    if (num_segments <= std::numeric_limits<int32_t>::max()) {
        sort_ids = ids64.to(torch::kInt32);
    }

    torch::Tensor perm = torch::argsort(sort_ids);
    torch::Tensor sorted_ids = ids64.index_select(0, perm);
    auto unique_tup = torch::unique_consecutive(sorted_ids, false, true);
    torch::Tensor unique_ids = std::get<0>(unique_tup);
    torch::Tensor counts = std::get<2>(unique_tup).to(torch::kInt64);

    auto count_opts = torch::TensorOptions().dtype(torch::kInt64).device(local_ids.device());
    torch::Tensor indptr = torch::zeros({unique_ids.numel() + 1}, count_opts);
    if (counts.numel() > 0) {
        indptr.narrow(0, 1, unique_ids.numel()).copy_(counts.cumsum(0));
    }

    return std::forward_as_tuple(unique_ids, perm, indptr);
}

torch::Tensor gather_negative_embeddings_csr(torch::Tensor node_embeddings, torch::Tensor negative_ids) {
    torch::Tensor flat_ids = negative_ids.flatten(0, 1).to(torch::kInt64);
    auto unique_tup = unique_with_inverse_compat(flat_ids);
    torch::Tensor unique_ids = std::get<0>(unique_tup);
    torch::Tensor inverse = std::get<1>(unique_tup).to(torch::kInt64);
    torch::Tensor unique_embeddings = node_embeddings.index_select(0, unique_ids);
    return unique_embeddings.index_select(0, inverse).reshape({negative_ids.size(0), negative_ids.size(1), -1});
}

void run_csr_reduce_debug(const std::string &tag, int64_t debug_batch_id, torch::Tensor local_ids, torch::Tensor values, int64_t num_segments) {
    if (!local_ids.defined() || !values.defined() || local_ids.numel() == 0 || values.numel() == 0 || num_segments <= 0) {
        SPDLOG_INFO("[csr-debug][batch {}][{}][step 0] skipped: local_ids defined={} values defined={} ids_numel={} values_numel={} num_segments={}",
                    debug_batch_id, tag, local_ids.defined(), values.defined(), local_ids.numel(), values.numel(), num_segments);
        return;
    }

    SPDLOG_INFO("[csr-debug][batch {}][{}][step 1] build local CSR input: ids_shape={} values_shape={} num_segments={}",
                debug_batch_id, tag, tensor_shape_string(local_ids), tensor_shape_string(values), num_segments);

    auto csr_tensors = build_csr_from_local_ids(local_ids, num_segments);
    torch::Tensor unique_ids = std::get<0>(csr_tensors);
    torch::Tensor perm = std::get<1>(csr_tensors);
    torch::Tensor indptr = std::get<2>(csr_tensors);

    int64_t indptr_last = indptr[-1].item<int64_t>();
    int64_t unique_count = unique_ids.numel();
    SPDLOG_INFO("[csr-debug][batch {}][{}][step 2] csr ready: perm_shape={} indptr_shape={} indptr_last={} unique_ids={}",
                debug_batch_id, tag, tensor_shape_string(perm), tensor_shape_string(indptr), indptr_last, unique_count);

    torch::Tensor sorted_values = values.index_select(0, perm);
    SPDLOG_INFO("[csr-debug][batch {}][{}][step 3] sorted values: shape={}", debug_batch_id, tag, tensor_shape_string(sorted_values));

    torch::Tensor reduced = segment_sum_csr(sorted_values, indptr, torch::nullopt);
    int64_t non_empty_segments = unique_count;
    double max_abs_diff = (values.sum(0) - reduced.sum(0)).abs().max().item<double>();
    SPDLOG_INFO("[csr-debug][batch {}][{}][step 4] reduced output: shape={} non_empty_segments={} max_abs_sum_diff={:.6e}",
                debug_batch_id, tag, tensor_shape_string(reduced), non_empty_segments, max_abs_diff);
}
#endif

struct PositiveGatherPlan {
    bool use_csr = false;
    torch::Tensor unique_ids;
    torch::Tensor src_inverse;
    torch::Tensor dst_inverse;
};

struct NegativeGatherPlan {
    bool use_csr = false;
    int64_t dim0 = 0;
    int64_t dim1 = 0;
    torch::Tensor unique_ids;
    torch::Tensor inverse;
};

PositiveGatherPlan build_positive_gather_plan(torch::Tensor src_ids, torch::Tensor dst_ids, bool use_csr_gather) {
    PositiveGatherPlan plan;
#ifdef GEGE_CUDA
    if (use_csr_gather && src_ids.device().is_cuda()) {
        torch::Tensor flat_ids = torch::cat({src_ids, dst_ids}, 0).to(torch::kInt64);
        auto unique_tup = unique_with_inverse_compat(flat_ids);
        torch::Tensor inverse = std::get<1>(unique_tup).to(torch::kInt64);

        int64_t src_count = src_ids.numel();
        plan.use_csr = true;
        plan.unique_ids = std::get<0>(unique_tup);
        plan.src_inverse = inverse.narrow(0, 0, src_count);
        plan.dst_inverse = inverse.narrow(0, src_count, dst_ids.numel());
    }
#endif
    return plan;
}

NegativeGatherPlan build_negative_gather_plan(torch::Tensor negative_ids, bool use_csr_gather) {
    NegativeGatherPlan plan;
    plan.dim0 = negative_ids.size(0);
    plan.dim1 = negative_ids.size(1);
#ifdef GEGE_CUDA
    if (use_csr_gather && negative_ids.device().is_cuda()) {
        torch::Tensor flat_ids = negative_ids.flatten(0, 1).to(torch::kInt64);
        auto unique_tup = unique_with_inverse_compat(flat_ids);
        plan.use_csr = true;
        plan.unique_ids = std::get<0>(unique_tup);
        plan.inverse = std::get<1>(unique_tup).to(torch::kInt64);
    }
#endif
    return plan;
}

std::tuple<torch::Tensor, torch::Tensor> gather_positive_embeddings(torch::Tensor node_embeddings, torch::Tensor src_ids, torch::Tensor dst_ids,
                                                                     bool use_csr_gather, const PositiveGatherPlan *plan_ptr = nullptr) {
#ifdef GEGE_CUDA
    if (use_csr_gather && node_embeddings.device().is_cuda()) {
        PositiveGatherPlan local_plan;
        const PositiveGatherPlan *active_plan = plan_ptr;
        if (active_plan == nullptr) {
            local_plan = build_positive_gather_plan(src_ids, dst_ids, use_csr_gather);
            active_plan = &local_plan;
        }

        if (active_plan->use_csr) {
            torch::Tensor unique_embeddings = node_embeddings.index_select(0, active_plan->unique_ids);
            torch::Tensor src_embeddings = unique_embeddings.index_select(0, active_plan->src_inverse);
            torch::Tensor dst_embeddings = unique_embeddings.index_select(0, active_plan->dst_inverse);
            return std::forward_as_tuple(src_embeddings, dst_embeddings);
        }
    }
#endif

    torch::Tensor src_embeddings = node_embeddings.index_select(0, src_ids);
    torch::Tensor dst_embeddings = node_embeddings.index_select(0, dst_ids);
    return std::forward_as_tuple(src_embeddings, dst_embeddings);
}

torch::Tensor gather_negative_embeddings(torch::Tensor node_embeddings, torch::Tensor negative_ids, bool use_csr_gather,
                                         const NegativeGatherPlan *plan_ptr = nullptr) {
#ifdef GEGE_CUDA
    if (use_csr_gather && node_embeddings.device().is_cuda()) {
        NegativeGatherPlan local_plan;
        const NegativeGatherPlan *active_plan = plan_ptr;
        if (active_plan == nullptr) {
            local_plan = build_negative_gather_plan(negative_ids, use_csr_gather);
            active_plan = &local_plan;
        }

        if (active_plan->use_csr) {
            torch::Tensor unique_embeddings = node_embeddings.index_select(0, active_plan->unique_ids);
            return unique_embeddings.index_select(0, active_plan->inverse).reshape({active_plan->dim0, active_plan->dim1, -1});
        }
    }
#endif
    return node_embeddings.index_select(0, negative_ids.flatten(0, 1)).reshape({negative_ids.size(0), negative_ids.size(1), -1});
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> only_pos_forward(shared_ptr<EdgeDecoder> decoder, torch::Tensor edges, torch::Tensor node_embeddings) {
    torch::Tensor pos_scores;
    torch::Tensor inv_pos_scores;

    bool has_relations;
    if (edges.size(1) == 3) {
        has_relations = true;
    } else if (edges.size(1) == 2) {
        has_relations = false;
    } else {
        throw TensorSizeMismatchException(edges, "Edge list must be a 3 or 2 column tensor");
    }

    bool use_csr_gather = false;
#ifdef GEGE_CUDA
    use_csr_gather = csr_gather_enabled() && node_embeddings.device().is_cuda();
#endif

    torch::Tensor src_ids = edges.select(1, 0);
    torch::Tensor dst_ids = edges.select(1, -1);
    auto src_dst = gather_positive_embeddings(node_embeddings, src_ids, dst_ids, use_csr_gather);
    torch::Tensor src = std::get<0>(src_dst);
    torch::Tensor dst = std::get<1>(src_dst);

    torch::Tensor rel_ids;

    if (has_relations) {
        rel_ids = edges.select(1, 1);

        torch::Tensor rels = decoder->select_relations(rel_ids);

        pos_scores = decoder->compute_scores(decoder->apply_relation(src, rels), dst);

        if (decoder->use_inverse_relations_) {
            torch::Tensor inv_rels = decoder->select_relations(rel_ids, true);

            inv_pos_scores = decoder->compute_scores(decoder->apply_relation(dst, inv_rels), src);
        }
    } else {
        pos_scores = decoder->compute_scores(src, dst);
    }

    return std::forward_as_tuple(pos_scores, inv_pos_scores);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> neg_and_pos_forward(shared_ptr<EdgeDecoder> decoder, torch::Tensor positive_edges,
                                                                                           torch::Tensor negative_edges, torch::Tensor node_embeddings) {
    torch::Tensor pos_scores;
    torch::Tensor inv_pos_scores;
    torch::Tensor neg_scores;
    torch::Tensor inv_neg_scores;

    std::tie(pos_scores, inv_pos_scores) = only_pos_forward(decoder, positive_edges, node_embeddings);
    std::tie(neg_scores, inv_neg_scores) = only_pos_forward(decoder, negative_edges, node_embeddings);

    return std::forward_as_tuple(pos_scores, neg_scores, inv_pos_scores, inv_neg_scores);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> node_corrupt_forward(shared_ptr<EdgeDecoder> decoder, torch::Tensor positive_edges,
                                                                                            torch::Tensor node_embeddings, torch::Tensor dst_negs,
                                                                                            torch::Tensor src_negs) {
    torch::Tensor pos_scores;
    torch::Tensor inv_pos_scores;
    torch::Tensor neg_scores;
    torch::Tensor inv_neg_scores;
    bool has_relations;
    if (positive_edges.size(1) == 3) {
        has_relations = true;
    } else if (positive_edges.size(1) == 2) {
        has_relations = false;
    } else {
        throw TensorSizeMismatchException(positive_edges, "Edge list must be a 3 or 2 column tensor");
    }

    bool use_csr_gather = false;
#ifdef GEGE_CUDA
    use_csr_gather = csr_gather_enabled() && node_embeddings.device().is_cuda();
#endif

    torch::Tensor src_ids = positive_edges.select(1, 0);
    torch::Tensor dst_ids = positive_edges.select(1, -1);
    auto src_dst = gather_positive_embeddings(node_embeddings, src_ids, dst_ids, use_csr_gather);
    torch::Tensor src = std::get<0>(src_dst);
    torch::Tensor dst = std::get<1>(src_dst);
    torch::Tensor rel_ids;

    torch::Tensor dst_neg_embs = gather_negative_embeddings(node_embeddings, dst_negs, use_csr_gather);

    if (has_relations) {
        rel_ids = positive_edges.select(1, 1);
        torch::Tensor rels = decoder->select_relations(rel_ids);
        torch::Tensor adjusted_src = decoder->apply_relation(src, rels);
        pos_scores = decoder->compute_scores(adjusted_src, dst);
        neg_scores = decoder->compute_scores(adjusted_src, dst_neg_embs);

        if (decoder->use_inverse_relations_) {
            torch::Tensor inv_rels = decoder->select_relations(rel_ids, true);
            torch::Tensor adjusted_dst = decoder->apply_relation(dst, inv_rels);

            torch::Tensor src_neg_embs = gather_negative_embeddings(node_embeddings, src_negs, use_csr_gather);

            inv_pos_scores = decoder->compute_scores(adjusted_dst, src);
            inv_neg_scores = decoder->compute_scores(adjusted_dst, src_neg_embs);
        }
    } else {
        pos_scores = decoder->compute_scores(src, dst);
        neg_scores = decoder->compute_scores(src, dst_neg_embs);
    }

    if (pos_scores.size(0) != neg_scores.size(0)) {
        int64_t new_size = neg_scores.size(0) - pos_scores.size(0);
        torch::nn::functional::PadFuncOptions options({0, new_size});
        pos_scores = torch::nn::functional::pad(pos_scores, options);

        if (inv_pos_scores.defined()) {
            inv_pos_scores = torch::nn::functional::pad(inv_pos_scores, options);
        }
    }

    return std::forward_as_tuple(pos_scores, neg_scores, inv_pos_scores, inv_neg_scores);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> rel_corrupt_forward(shared_ptr<EdgeDecoder> decoder, torch::Tensor positive_edges,
                                                                                           torch::Tensor node_embeddings, torch::Tensor neg_rel_ids) {
    torch::Tensor pos_scores;
    torch::Tensor inv_pos_scores;
    torch::Tensor neg_scores;
    torch::Tensor inv_neg_scores;

    if (positive_edges.size(1) != 3) {
        throw TensorSizeMismatchException(positive_edges, "Edge list must be a 3 column tensor");
    }

    bool use_csr_gather = false;
#ifdef GEGE_CUDA
    use_csr_gather = csr_gather_enabled() && node_embeddings.device().is_cuda();
#endif
    torch::Tensor src_ids = positive_edges.select(1, 0);
    torch::Tensor dst_ids = positive_edges.select(1, -1);
    auto src_dst = gather_positive_embeddings(node_embeddings, src_ids, dst_ids, use_csr_gather);
    torch::Tensor src = std::get<0>(src_dst);
    torch::Tensor dst = std::get<1>(src_dst);

    torch::Tensor rel_ids = positive_edges.select(1, 1);

    torch::Tensor rels = decoder->select_relations(rel_ids);
    torch::Tensor neg_rels = decoder->select_relations(neg_rel_ids);

    pos_scores = decoder->compute_scores(decoder->apply_relation(src, rels), dst);
    neg_scores = decoder->compute_scores(decoder->apply_relation(src, neg_rels), dst);

    if (decoder->use_inverse_relations_) {
        torch::Tensor inv_rels = decoder->select_relations(rel_ids, true);
        torch::Tensor inv_neg_rels = decoder->select_relations(neg_rel_ids, true);

        inv_pos_scores = decoder->compute_scores(decoder->apply_relation(dst, inv_rels), src);
        inv_neg_scores = decoder->compute_scores(decoder->apply_relation(dst, inv_neg_rels), src);
    }

    return std::forward_as_tuple(pos_scores, neg_scores, inv_pos_scores, inv_neg_scores);
}

std::tuple<torch::Tensor, torch::Tensor> prepare_pos_embeddings(shared_ptr<EdgeDecoder> decoder, torch::Tensor positive_edges, torch::Tensor src_embeddings, torch::Tensor dst_embeddings, bool has_relations) {
    torch::Tensor adjusted_src_embeddings;
    torch::Tensor adjusted_dst_embeddings;

    torch::Tensor rel_ids;
    torch::Tensor rels;
    torch::Tensor inv_rels;

    if (has_relations) {
        rel_ids = positive_edges.select(1, 1);
        rels = decoder->select_relations(rel_ids);
        adjusted_src_embeddings = decoder->apply_relation(src_embeddings, rels);
        if (decoder->use_inverse_relations_) {
            inv_rels = decoder->select_relations(rel_ids, true);
            adjusted_dst_embeddings = decoder->apply_relation(dst_embeddings, inv_rels);
        }
    } else {  // no relations
        adjusted_src_embeddings = src_embeddings;
    }

    return std::forward_as_tuple(adjusted_src_embeddings, adjusted_dst_embeddings);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> mod_node_corrupt_forward(NegativeSamplingMethod negative_sampling_method, float negative_sampling_selected_ratio, shared_ptr<NegativeSampler> negative_sampler,
                                                                                                shared_ptr<EdgeDecoder> decoder, torch::Tensor positive_edges, torch::Tensor node_embeddings, torch::Tensor dst_negs, torch::Tensor src_negs,
                                                                                                torch::Tensor node_embeddings_g) {
    int64_t stage_debug_batch_id = -1;
    bool run_stage_debug = should_run_stage_debug(stage_debug_batch_id);
    auto decoder_total_start = std::chrono::high_resolution_clock::now();
    auto step_start = decoder_total_start;

    bool has_relations;
    if (positive_edges.size(1) == 3) {
        has_relations = true;
    } else if (positive_edges.size(1) == 2) {
        has_relations = false;
    } else {
        throw TensorSizeMismatchException(positive_edges, "Edge list must be a 3 or 2 column tensor");
    }

    torch::Tensor src_ids = positive_edges.select(1, 0);
    torch::Tensor dst_ids = positive_edges.select(1, -1);

    bool use_csr_gather = false;
#ifdef GEGE_CUDA
    use_csr_gather = csr_gather_enabled() && node_embeddings.device().is_cuda();
#endif
    PositiveGatherPlan positive_gather_plan = build_positive_gather_plan(src_ids, dst_ids, use_csr_gather);
    NegativeGatherPlan dst_neg_gather_plan = build_negative_gather_plan(dst_negs, use_csr_gather);
    NegativeGatherPlan src_neg_gather_plan;
    if (has_relations && decoder->use_inverse_relations_) {
        src_neg_gather_plan = build_negative_gather_plan(src_negs, use_csr_gather);
    }
    auto src_dst = gather_positive_embeddings(node_embeddings, src_ids, dst_ids, use_csr_gather, &positive_gather_plan);
    torch::Tensor src_embeddings = std::get<0>(src_dst);
    torch::Tensor dst_embeddings = std::get<1>(src_dst);
    if (run_stage_debug) {
        auto now = std::chrono::high_resolution_clock::now();
        int64_t src_dst_total = src_ids.numel() + dst_ids.numel();
        int64_t src_dst_unique = positive_gather_plan.use_csr ? positive_gather_plan.unique_ids.numel() : src_dst_total;
        int64_t src_dst_duplicates = std::max<int64_t>(src_dst_total - src_dst_unique, 0);
        SPDLOG_INFO(
            "[stage-debug][decoder][batch {}][step 1] positive_gather ms={:.3f} use_csr_gather={} src_dst_total={} src_dst_unique={} src_dst_duplicates={}",
            stage_debug_batch_id, elapsed_ms(step_start, now), use_csr_gather, src_dst_total, src_dst_unique, src_dst_duplicates);
        step_start = now;
    }
    int batch_num = src_embeddings.sizes()[0];
    int embedding_size = src_embeddings.sizes()[1];
    int chunk_num = dst_negs.sizes()[0];
    int negatives_num = dst_negs.sizes()[1];
    int selected_negatives_num = int(negatives_num * negative_sampling_selected_ratio);
    int num_per_chunk = (int)ceil((float) batch_num / chunk_num);

    // SPDLOG_INFO("batch_num : {}", batch_num);
    // SPDLOG_INFO("embedding_size: {}", embedding_size);
    // SPDLOG_INFO("chunk_num: {}", chunk_num);
    // SPDLOG_INFO("negatives_num: {}", negatives_num);
    // SPDLOG_INFO("selected_negatives_num: {}", selected_negatives_num);

    torch::Tensor adjusted_src_embeddings;
    torch::Tensor adjusted_dst_embeddings;
    torch::Tensor dst_neg_embeddings;
    torch::Tensor src_neg_embeddings;
    torch::Tensor selected_dst_neg_indices;
    torch::Tensor selected_src_neg_indices;
    torch::Tensor dst_neg_candidates = dst_negs;
    torch::Tensor src_neg_candidates = src_negs;
    torch::Tensor selector_dst_chunked_embeddings;
    torch::Tensor selector_src_chunked_embeddings;
    torch::Tensor selector_dst_neg_embeddings;
    torch::Tensor selector_src_neg_embeddings;
    bool use_tiled_tournament_scores = false;

    auto all_pos_embeddings = prepare_pos_embeddings(decoder, positive_edges, src_embeddings, dst_embeddings, has_relations);

    if (has_relations) {
        adjusted_src_embeddings = std::get<0>(all_pos_embeddings);
        dst_neg_embeddings = gather_negative_embeddings(node_embeddings, dst_negs, use_csr_gather, &dst_neg_gather_plan);
        if (decoder->use_inverse_relations_) {
            adjusted_dst_embeddings = std::get<1>(all_pos_embeddings);
            src_neg_embeddings = gather_negative_embeddings(node_embeddings, src_negs, use_csr_gather, &src_neg_gather_plan);
        }
    } else {
        adjusted_src_embeddings = std::get<0>(all_pos_embeddings);
        dst_neg_embeddings = gather_negative_embeddings(node_embeddings, dst_negs, use_csr_gather, &dst_neg_gather_plan);
    }
    if (run_stage_debug) {
        auto now = std::chrono::high_resolution_clock::now();
        int64_t dst_total = dst_negs.numel();
        int64_t dst_unique = dst_neg_gather_plan.use_csr ? dst_neg_gather_plan.unique_ids.numel() : dst_total;
        int64_t dst_duplicates = std::max<int64_t>(dst_total - dst_unique, 0);
        int64_t src_total = src_negs.defined() ? src_negs.numel() : 0;
        int64_t src_unique = (src_neg_gather_plan.use_csr && src_neg_gather_plan.unique_ids.defined()) ? src_neg_gather_plan.unique_ids.numel() : src_total;
        int64_t src_duplicates = std::max<int64_t>(src_total - src_unique, 0);
        SPDLOG_INFO(
            "[stage-debug][decoder][batch {}][step 2] neg_gather ms={:.3f} dst_total={} dst_unique={} dst_dup={} src_total={} src_unique={} src_dup={}",
            stage_debug_batch_id, elapsed_ms(step_start, now), dst_total, dst_unique, dst_duplicates, src_total, src_unique, src_duplicates);
        step_start = now;
    }

    int64_t csr_debug_batch_id = -1;
    bool run_csr_debug = should_run_csr_debug(csr_debug_batch_id);

    {
        torch::NoGradGuard no_grad;

        torch::Tensor adjusted_src_embeddings_g;
        torch::Tensor adjusted_dst_embeddings_g;
        torch::Tensor dst_neg_embeddings_g;
        torch::Tensor src_neg_embeddings_g;
        torch::Tensor dst_negs_scores;
        torch::Tensor src_negs_scores;

        if (negative_sampling_method == NegativeSamplingMethod::DNS || negative_sampling_method == NegativeSamplingMethod::GAN) {
            if (negative_sampling_method == NegativeSamplingMethod::GAN) {
                auto src_dst_g = gather_positive_embeddings(node_embeddings_g, src_ids, dst_ids, use_csr_gather, &positive_gather_plan);
                auto all_pos_embeddings_g = prepare_pos_embeddings(decoder, positive_edges, std::get<0>(src_dst_g), std::get<1>(src_dst_g), has_relations);
                adjusted_src_embeddings_g = std::get<0>(all_pos_embeddings_g);
                adjusted_dst_embeddings_g = std::get<1>(all_pos_embeddings_g);
                dst_neg_embeddings_g = gather_negative_embeddings(node_embeddings_g, dst_negs, use_csr_gather, &dst_neg_gather_plan);
                if (has_relations && decoder->use_inverse_relations_) {
                    src_neg_embeddings_g = gather_negative_embeddings(node_embeddings_g, src_negs, use_csr_gather, &src_neg_gather_plan);
                }
            }

            if (negative_sampling_method == NegativeSamplingMethod::GAN) {
                selector_dst_chunked_embeddings = pad_and_reshape(adjusted_src_embeddings_g.detach(), chunk_num);
                selector_dst_neg_embeddings = dst_neg_embeddings_g.detach();
                if (has_relations && decoder->use_inverse_relations_) {
                    selector_src_chunked_embeddings = pad_and_reshape(adjusted_dst_embeddings_g.detach(), chunk_num);
                    selector_src_neg_embeddings = src_neg_embeddings_g.detach();
                }
            } else {
                selector_dst_chunked_embeddings = pad_and_reshape(adjusted_src_embeddings.detach(), chunk_num);
                selector_dst_neg_embeddings = dst_neg_embeddings.detach();
                if (has_relations && decoder->use_inverse_relations_) {
                    selector_src_chunked_embeddings = pad_and_reshape(adjusted_dst_embeddings.detach(), chunk_num);
                    selector_src_neg_embeddings = src_neg_embeddings.detach();
                }
            }

            use_tiled_tournament_scores =
                can_use_tiled_distmult_tournament_scores(decoder, negative_sampler, run_csr_debug, adjusted_src_embeddings, dst_neg_embeddings,
                                                         selected_negatives_num) &&
                (!has_relations || !decoder->use_inverse_relations_ ||
                 can_use_tiled_distmult_tournament_scores(decoder, negative_sampler, run_csr_debug, adjusted_dst_embeddings, src_neg_embeddings,
                                                         selected_negatives_num));

            if (!use_tiled_tournament_scores) {
                if (negative_sampling_method == NegativeSamplingMethod::GAN) {
                    dst_negs_scores = candidate_sampling_scores(decoder, adjusted_src_embeddings_g, dst_neg_embeddings_g, chunk_num, num_per_chunk);
                    if (has_relations && decoder->use_inverse_relations_) {
                        src_negs_scores = candidate_sampling_scores(decoder, adjusted_dst_embeddings_g, src_neg_embeddings_g, chunk_num, num_per_chunk);
                    }
                } else {
                    dst_negs_scores = candidate_sampling_scores(decoder, adjusted_src_embeddings, dst_neg_embeddings, chunk_num, num_per_chunk);
                    if (has_relations && decoder->use_inverse_relations_) {
                        src_negs_scores = candidate_sampling_scores(decoder, adjusted_dst_embeddings, src_neg_embeddings, chunk_num, num_per_chunk);
                    }
                }

                auto all_selected_negs = negative_sampler->sample(dst_negs, src_negs, dst_negs_scores, src_negs_scores, chunk_num, num_per_chunk,
                                                                  selected_negatives_num, has_relations, decoder->use_inverse_relations_);

                torch::Tensor sampled_dst_negs = std::get<0>(all_selected_negs);
                torch::Tensor sampled_src_negs = std::get<1>(all_selected_negs);
                selected_dst_neg_indices = std::get<2>(all_selected_negs);
                selected_src_neg_indices = std::get<3>(all_selected_negs);

                if (sampled_dst_negs.defined()) {
                    dst_negs = sampled_dst_negs;
                }
                if (sampled_src_negs.defined()) {
                    src_negs = sampled_src_negs;
                }
            }
        }
    }
    if (run_stage_debug) {
        auto now = std::chrono::high_resolution_clock::now();
        int64_t tiled_selected_count = static_cast<int64_t>(chunk_num) * num_per_chunk * selected_negatives_num;
        int64_t selected_dst = selected_dst_neg_indices.defined() ? selected_dst_neg_indices.numel() : (use_tiled_tournament_scores ? tiled_selected_count : dst_negs.numel());
        int64_t selected_src = selected_src_neg_indices.defined() ? selected_src_neg_indices.numel() : (use_tiled_tournament_scores ? tiled_selected_count : src_negs.numel());
        SPDLOG_INFO("[stage-debug][decoder][batch {}][step 3] neg_sampler ms={:.3f} method={} selected_dst={} selected_src={}",
                    stage_debug_batch_id, elapsed_ms(step_start, now), (int)negative_sampling_method, selected_dst, selected_src);
        step_start = now;
    }

    torch::Tensor score_dst_chunked_embeddings;
    torch::Tensor score_src_chunked_embeddings;
    if (use_tiled_tournament_scores) {
        score_dst_chunked_embeddings = pad_and_reshape(adjusted_src_embeddings, chunk_num);
        if (has_relations && decoder->use_inverse_relations_) {
            score_src_chunked_embeddings = pad_and_reshape(adjusted_dst_embeddings, chunk_num);
        }
        if (negative_sampling_method == NegativeSamplingMethod::DNS) {
            selector_dst_chunked_embeddings = score_dst_chunked_embeddings.detach();
            if (has_relations && decoder->use_inverse_relations_) {
                selector_src_chunked_embeddings = score_src_chunked_embeddings.detach();
            }
        }
    }

    torch::Tensor pos_scores;
    torch::Tensor inv_pos_scores;
    torch::Tensor neg_scores;
    torch::Tensor inv_neg_scores;
    if (run_csr_debug) {
        SPDLOG_INFO("[csr-debug][batch {}] csr_gather_enabled={} use_csr_gather={}", csr_debug_batch_id, csr_gather_enabled(), use_csr_gather);
    }

    switch (negative_sampling_method) {
        case NegativeSamplingMethod::RNS : {
            if (has_relations) {
                pos_scores = decoder->compute_scores(adjusted_src_embeddings, dst_embeddings);
                neg_scores = decoder->compute_scores(adjusted_src_embeddings, dst_neg_embeddings);
                if (decoder->use_inverse_relations_) {
                    inv_pos_scores = decoder->compute_scores(adjusted_dst_embeddings, src_embeddings);
                    inv_neg_scores = decoder->compute_scores(adjusted_dst_embeddings, src_neg_embeddings);
                }
            } else {
                pos_scores = decoder->compute_scores(adjusted_src_embeddings, dst_embeddings);
                neg_scores = decoder->compute_scores(adjusted_src_embeddings, dst_neg_embeddings);
            }
            break;
        }
        case NegativeSamplingMethod::DNS :
        case NegativeSamplingMethod::GAN : {
            SelectedNegScoreKind score_kind = selected_neg_score_kind(decoder);
            bool use_fused_dst_scores = !use_tiled_tournament_scores &&
                                        can_use_selected_neg_cuda(decoder, run_csr_debug, adjusted_src_embeddings, dst_neg_embeddings, selected_dst_neg_indices);
            torch::Tensor selected_dst_negs_embeddings;
            if (!use_fused_dst_scores) {
                if (!use_tiled_tournament_scores) {
                    selected_dst_negs_embeddings =
                        materialize_selected_negative_embeddings(dst_neg_embeddings, selected_dst_neg_indices, chunk_num, num_per_chunk, selected_negatives_num);
                }
                // SPDLOG_INFO("selected_dst_negs_embeddings dim: {}", selected_dst_negs_embeddings.dim());
                // SPDLOG_INFO("selected_dst_negs_embeddings size[0]: {}", selected_dst_negs_embeddings.sizes()[0]);  // chunk num
                // SPDLOG_INFO("selected_dst_negs_embeddings size[1]: {}", selected_dst_negs_embeddings.sizes()[1]);  // num_per_chunk
                // SPDLOG_INFO("selected_dst_negs_embeddings size[2]: {}", selected_dst_negs_embeddings.sizes()[2]);  // selected negatives num
                // SPDLOG_INFO("selected_dst_negs_embeddings size[3]: {}", selected_dst_negs_embeddings.sizes()[3]);  // embedding size

#ifdef GEGE_CUDA
                if (!use_tiled_tournament_scores && run_csr_debug && node_embeddings.device().is_cuda()) {
                    SPDLOG_INFO("[csr-debug][batch {}][step -1] DNS/GAN branch: start dst reduce validation", csr_debug_batch_id);
                    torch::NoGradGuard no_grad;
                    torch::Tensor dst_selected_ids = gather_selected_negative_ids(dst_neg_candidates, selected_dst_neg_indices);
                    torch::Tensor dst_local_ids = dst_selected_ids.flatten(0, 1).reshape({-1}).to(torch::kInt64);
                    torch::Tensor dst_flat_values = selected_dst_negs_embeddings.reshape({-1, selected_dst_negs_embeddings.size(-1)});
                    run_csr_reduce_debug("dst_neg", csr_debug_batch_id, dst_local_ids, dst_flat_values, node_embeddings.size(0));
                }
#endif
            }

            if (has_relations) {
                pos_scores = decoder->compute_scores(adjusted_src_embeddings, dst_embeddings);

                if (use_tiled_tournament_scores) {
#ifdef GEGE_CUDA
                    std::tie(neg_scores, selected_dst_neg_indices) = distmult_tiled_tournament_selected_scores(
                        selector_dst_chunked_embeddings,
                        selector_dst_neg_embeddings,
                        score_dst_chunked_embeddings,
                        dst_neg_embeddings,
                        selected_negatives_num,
                        tiled_tournament_groups_per_tile(negative_sampler));
                    validate_tiled_tournament_scores_once(selector_dst_chunked_embeddings,
                                                          selector_dst_neg_embeddings,
                                                          score_dst_chunked_embeddings,
                                                          dst_neg_embeddings,
                                                          selected_negatives_num,
                                                          neg_scores,
                                                          selected_dst_neg_indices,
                                                          "dst");
#endif
                } else if (use_fused_dst_scores) {
#ifdef GEGE_CUDA
                    neg_scores = selected_neg_scores(pad_and_reshape(adjusted_src_embeddings, chunk_num), dst_neg_embeddings, selected_dst_neg_indices, score_kind);
#endif
                } else {
                    neg_scores = decoder->compute_scores(adjusted_src_embeddings, selected_dst_negs_embeddings);
                }
                if (decoder->use_inverse_relations_) {
                    torch::Tensor selected_src_negs_embeddings;
                    bool use_fused_src_scores = !use_tiled_tournament_scores &&
                                                can_use_selected_neg_cuda(decoder, run_csr_debug, adjusted_dst_embeddings, src_neg_embeddings, selected_src_neg_indices);
                    if (!use_fused_src_scores) {
                        if (!use_tiled_tournament_scores) {
                            selected_src_negs_embeddings =
                                materialize_selected_negative_embeddings(src_neg_embeddings, selected_src_neg_indices, chunk_num, num_per_chunk, selected_negatives_num);
                        }
#ifdef GEGE_CUDA
                        if (!use_tiled_tournament_scores && run_csr_debug && node_embeddings.device().is_cuda()) {
                            SPDLOG_INFO("[csr-debug][batch {}][step -1] DNS/GAN branch: start src reduce validation", csr_debug_batch_id);
                            torch::NoGradGuard no_grad;
                            torch::Tensor src_selected_ids = gather_selected_negative_ids(src_neg_candidates, selected_src_neg_indices);
                            torch::Tensor src_local_ids = src_selected_ids.flatten(0, 1).reshape({-1}).to(torch::kInt64);
                            torch::Tensor src_flat_values = selected_src_negs_embeddings.reshape({-1, selected_src_negs_embeddings.size(-1)});
                            run_csr_reduce_debug("src_neg", csr_debug_batch_id, src_local_ids, src_flat_values, node_embeddings.size(0));
                        }
#endif
                    }
                    inv_pos_scores = decoder->compute_scores(adjusted_dst_embeddings, src_embeddings);

                    if (use_tiled_tournament_scores) {
#ifdef GEGE_CUDA
                        std::tie(inv_neg_scores, selected_src_neg_indices) = distmult_tiled_tournament_selected_scores(
                            selector_src_chunked_embeddings,
                            selector_src_neg_embeddings,
                            score_src_chunked_embeddings,
                            src_neg_embeddings,
                            selected_negatives_num,
                            tiled_tournament_groups_per_tile(negative_sampler));
                        validate_tiled_tournament_scores_once(selector_src_chunked_embeddings,
                                                              selector_src_neg_embeddings,
                                                              score_src_chunked_embeddings,
                                                              src_neg_embeddings,
                                                              selected_negatives_num,
                                                              inv_neg_scores,
                                                              selected_src_neg_indices,
                                                              "src");
#endif
                    } else if (use_fused_src_scores) {
#ifdef GEGE_CUDA
                        inv_neg_scores = selected_neg_scores(
                            pad_and_reshape(adjusted_dst_embeddings, chunk_num), src_neg_embeddings, selected_src_neg_indices, score_kind);
#endif
                    } else {
                        inv_neg_scores = decoder->compute_scores(adjusted_dst_embeddings, selected_src_negs_embeddings);
                    }
                }
            } else {
                pos_scores = decoder->compute_scores(adjusted_src_embeddings, dst_embeddings);

                if (use_tiled_tournament_scores) {
#ifdef GEGE_CUDA
                    std::tie(neg_scores, selected_dst_neg_indices) = distmult_tiled_tournament_selected_scores(
                        selector_dst_chunked_embeddings,
                        selector_dst_neg_embeddings,
                        score_dst_chunked_embeddings,
                        dst_neg_embeddings,
                        selected_negatives_num,
                        tiled_tournament_groups_per_tile(negative_sampler));
                    validate_tiled_tournament_scores_once(selector_dst_chunked_embeddings,
                                                          selector_dst_neg_embeddings,
                                                          score_dst_chunked_embeddings,
                                                          dst_neg_embeddings,
                                                          selected_negatives_num,
                                                          neg_scores,
                                                          selected_dst_neg_indices,
                                                          "dst");
#endif
                } else if (use_fused_dst_scores) {
#ifdef GEGE_CUDA
                    neg_scores = selected_neg_scores(pad_and_reshape(adjusted_src_embeddings, chunk_num), dst_neg_embeddings, selected_dst_neg_indices, score_kind);
#endif
                } else {
                    neg_scores = decoder->compute_scores(adjusted_src_embeddings, selected_dst_negs_embeddings);
                }
            }
            break;
        }
        default : {
            throw GegeRuntimeException("Unsupported negative_sampling_method in scoreNegatives");
        }
    }

    if (pos_scores.size(0) != neg_scores.size(0)) {
        int64_t new_size = neg_scores.size(0) - pos_scores.size(0);
        torch::nn::functional::PadFuncOptions options({0, new_size});
        pos_scores = torch::nn::functional::pad(pos_scores, options);

        if (inv_pos_scores.defined()) {
            inv_pos_scores = torch::nn::functional::pad(inv_pos_scores, options);
        }
    }
    if (run_stage_debug) {
        auto now = std::chrono::high_resolution_clock::now();
        int64_t pos_numel = pos_scores.defined() ? pos_scores.numel() : 0;
        int64_t neg_numel = neg_scores.defined() ? neg_scores.numel() : 0;
        SPDLOG_INFO("[stage-debug][decoder][batch {}][step 4] score+pad ms={:.3f} pos_numel={} neg_numel={} total_ms={:.3f}",
                    stage_debug_batch_id, elapsed_ms(step_start, now), pos_numel, neg_numel, elapsed_ms(decoder_total_start, now));
    }
    // SPDLOG_INFO("pos_scores : {}", pos_scores.dim());
    // SPDLOG_INFO("pos_scores size[0]: {}", pos_scores.sizes()[0]);  // chunk num
    // SPDLOG_INFO("pos_scores size[1]: {}", pos_scores.sizes()[1]);  // num_per_chunk
    // SPDLOG_INFO("pos_scores size[2]: {}", pos_scores.sizes()[2]);  // embedding size
    // SPDLOG_INFO("neg_scores: {}", neg_scores.dim());
    // SPDLOG_INFO("neg_scores size[0]: {}", neg_scores.sizes()[0]);  // chunk num
    // SPDLOG_INFO("neg_scores size[1]: {}", neg_scores.sizes()[1]);  // num_per_chunk
    // SPDLOG_INFO("neg_scores size[2]: {}", neg_scores.sizes()[2]);  // embedding size

    return std::forward_as_tuple(pos_scores, neg_scores, inv_pos_scores, inv_neg_scores);
}

std::tuple<torch::Tensor, torch::Tensor> get_rewards(shared_ptr<EdgeDecoder> decoder, torch::Tensor positive_edges, torch::Tensor node_embeddings, torch::Tensor dst_negs, torch::Tensor src_negs) {
    bool has_relations;
    if (positive_edges.size(1) == 3) {
        has_relations = true;
    } else if (positive_edges.size(1) == 2) {
        has_relations = false;
    } else {
        throw TensorSizeMismatchException(positive_edges, "Edge list must be a 3 or 2 column tensor");
    }
    torch::Tensor reward;
    torch::Tensor inv_reward;
    bool use_csr_gather = false;
#ifdef GEGE_CUDA
    use_csr_gather = csr_gather_enabled() && node_embeddings.device().is_cuda();
#endif
    {
        torch::NoGradGuard no_grad;

        torch::Tensor adjusted_src_embeddings;
        torch::Tensor adjusted_dst_embeddings;
        torch::Tensor rel_ids;
        torch::Tensor rels;
        torch::Tensor inv_rels;

        torch::Tensor src_ids = positive_edges.select(1, 0);
        torch::Tensor dst_ids = positive_edges.select(1, -1);
        auto src_dst = gather_positive_embeddings(node_embeddings, src_ids, dst_ids, use_csr_gather);
        torch::Tensor src_embeddings = std::get<0>(src_dst);
        torch::Tensor dst_embeddings = std::get<1>(src_dst);

        torch::Tensor dst_neg_embs = gather_negative_embeddings(node_embeddings, dst_negs, use_csr_gather);
        if (has_relations) {
            rel_ids = positive_edges.select(1, 1);
            rels = decoder->select_relations(rel_ids);
            adjusted_src_embeddings = decoder->apply_relation(src_embeddings, rels);
            // reward = decoder->compute_scores(adjusted_src_embeddings, dst_neg_embs);
            torch::Tensor logits = decoder->compute_scores(adjusted_src_embeddings, dst_neg_embs);
            reward = logits.sigmoid().sub(0.5).mul(2);
            if (decoder->use_inverse_relations_) {
                inv_rels = decoder->select_relations(rel_ids, true);
                adjusted_dst_embeddings = decoder->apply_relation(dst_embeddings, inv_rels);
                torch::Tensor src_neg_embs = gather_negative_embeddings(node_embeddings, src_negs, use_csr_gather);
                // inv_reward = decoder->compute_scores(adjusted_dst_embeddings, src_neg_embs);
                torch::Tensor inv_logits = decoder->compute_scores(adjusted_dst_embeddings, src_neg_embs);
                inv_reward = inv_logits.sigmoid().sub(0.5).mul(2);
            }
        } else {
            adjusted_src_embeddings = src_embeddings;
            // reward = decoder->compute_scores(adjusted_src_embeddings, dst_neg_embs);
            torch::Tensor logits = decoder->compute_scores(adjusted_src_embeddings, dst_neg_embs);
            reward = logits.sigmoid().sub(0.5).mul(2);
        }
    }
    return std::forward_as_tuple(reward, inv_reward);
}

torch::Tensor forward_g(shared_ptr<EdgeDecoder> decoder, torch::Tensor positive_edges, torch::Tensor node_embeddings_g, torch::Tensor dst_negs, torch::Tensor src_negs, torch::Tensor reward, torch::Tensor inv_reward) {
    bool has_relations;
    if (positive_edges.size(1) == 3) {
        has_relations = true;
    } else if (positive_edges.size(1) == 2) {
        has_relations = false;
    } else {
        throw TensorSizeMismatchException(positive_edges, "Edge list must be a 3 or 2 column tensor");
    }

    bool use_csr_gather = false;
#ifdef GEGE_CUDA
    use_csr_gather = csr_gather_enabled() && node_embeddings_g.device().is_cuda();
#endif
    torch::Tensor src_ids = positive_edges.select(1, 0);
    torch::Tensor dst_ids = positive_edges.select(1, -1);
    auto src_dst_g = gather_positive_embeddings(node_embeddings_g, src_ids, dst_ids, use_csr_gather);
    torch::Tensor src_embeddings_g = std::get<0>(src_dst_g);
    torch::Tensor dst_neg_embs_g = gather_negative_embeddings(node_embeddings_g, dst_negs, use_csr_gather);

    torch::Tensor logits = decoder->compute_scores(src_embeddings_g, dst_neg_embs_g);
    torch::Tensor probs = logits.softmax(1);
    reward = reward.mul(probs);
    torch::Tensor log_probs = logits.log_softmax(1);
    torch::Tensor loss_g = -(log_probs.mul(reward).mean());
    // torch::Tensor loss_g = -(log_probs.mul(reward).sum());

    if (has_relations && decoder->use_inverse_relations_) {
        torch::Tensor dst_embeddings_g = std::get<1>(src_dst_g);
        torch::Tensor src_neg_embs_g = gather_negative_embeddings(node_embeddings_g, src_negs, use_csr_gather);

        torch::Tensor inv_logits = decoder->compute_scores(dst_embeddings_g, src_neg_embs_g);
        torch::Tensor inv_probs = inv_logits.softmax(1);
        inv_reward = inv_reward.mul(inv_probs);
        torch::Tensor inv_log_probs = inv_logits.log_softmax(1);
        torch::Tensor inv_loss_g = -(inv_log_probs.mul(inv_reward).mean());
        // torch::Tensor inv_loss_g = -(inv_log_probs.mul(inv_reward).sum());

        return loss_g + inv_loss_g;
    }

    return loss_g;
}
