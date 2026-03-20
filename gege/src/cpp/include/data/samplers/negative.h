#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <limits>
#include <map>
#include <mutex>
#include <string>

#include "storage/graph_storage.h"

inline bool parse_negative_env_flag(const char *name, bool default_value) {
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

inline int parse_negative_env_int(const char *name, int default_value) {
    const char *raw = std::getenv(name);
    if (raw == nullptr) {
        return default_value;
    }

    try {
        return std::stoi(std::string(raw));
    } catch (...) {
        return default_value;
    }
}

inline bool negative_tournament_env_enabled() {
    static bool enabled = parse_negative_env_flag("GEGE_NEGATIVE_TOURNAMENT", false);
    return enabled;
}

inline torch::Tensor tournament_select_indices(torch::Tensor scores, int selected_negatives_num, bool use_tournament_selection) {
    if (!scores.defined()) {
        return torch::Tensor();
    }

    int64_t negatives_num = scores.size(2);
    bool enabled = use_tournament_selection || negative_tournament_env_enabled();
    if (!enabled || selected_negatives_num <= 0 || selected_negatives_num >= negatives_num ||
        negatives_num % selected_negatives_num != 0) {
        auto results = scores.topk(selected_negatives_num, 2, true, false);
        return std::get<1>(results);
    }

    int64_t tournament_size = negatives_num / selected_negatives_num;
    torch::Tensor grouped_scores = scores.reshape({scores.size(0), scores.size(1), selected_negatives_num, tournament_size});
    auto grouped_results = grouped_scores.max(3);
    torch::Tensor local_indices = std::get<1>(grouped_results).to(torch::kInt64);
    auto idx_opts = torch::TensorOptions().dtype(torch::kInt64).device(scores.device());
    torch::Tensor offsets = torch::arange(selected_negatives_num, idx_opts).view({1, 1, selected_negatives_num}) * tournament_size;
    return local_indices + offsets;
}

std::tuple<torch::Tensor, torch::Tensor> batch_sample(torch::Tensor edges, int num_negatives, bool inverse = false);

torch::Tensor deg_negative_local_filter(torch::Tensor deg_sample_indices, torch::Tensor edges);

torch::Tensor compute_filter_corruption_cpu(shared_ptr<GegeGraph> graph, torch::Tensor edges, torch::Tensor corruption_nodes, bool inverse = false,
                                            bool global = false, LocalFilterMode local_filter_mode = LocalFilterMode::ALL,
                                            torch::Tensor deg_sample_indices = torch::Tensor());

torch::Tensor compute_filter_corruption_gpu(shared_ptr<GegeGraph> graph, torch::Tensor edges, torch::Tensor corruption_nodes, bool inverse = false,
                                            bool global = false, LocalFilterMode local_filter_mode = LocalFilterMode::ALL,
                                            torch::Tensor deg_sample_indices = torch::Tensor());

torch::Tensor compute_filter_corruption(shared_ptr<GegeGraph> graph, torch::Tensor edges, torch::Tensor corruption_nodes, bool inverse = false,
                                        bool global = false, LocalFilterMode local_filter_mode = LocalFilterMode::ALL,
                                        torch::Tensor deg_sample_indices = torch::Tensor());

torch::Tensor apply_score_filter(torch::Tensor scores, torch::Tensor filter);

struct NegativeSamplerPerfStats {
    int64_t get_negatives_total_ns = 0;
    int64_t get_negatives_call_count = 0;
    int64_t plan_lock_wait_ns = 0;
    int64_t plan_lock_wait_count = 0;
    int64_t state_pool_hit_count = 0;
    int64_t planned_uniform_fetch_count = 0;
    int64_t cuda_call_count = 0;
    int64_t cpu_call_count = 0;
    int64_t uniform_randint_ns = 0;
    int64_t sample_edge_randint_ns = 0;
    int64_t materialize_ns = 0;
    int64_t filter_ns = 0;
    int64_t filter_deg_chunk_ids_ns = 0;
    int64_t filter_deg_mask_ns = 0;
    int64_t filter_deg_nonzero_ns = 0;
    int64_t filter_deg_gather_ns = 0;
    int64_t filter_deg_finalize_ns = 0;
    int64_t filter_gpu_prepare_ns = 0;
    int64_t filter_gpu_searchsorted_ns = 0;
    int64_t filter_gpu_offsets_ns = 0;
    int64_t filter_gpu_repeat_interleave_ns = 0;
    int64_t filter_gpu_neighbor_gather_ns = 0;
    int64_t filter_gpu_relation_filter_ns = 0;
    int64_t filter_gpu_finalize_ns = 0;
    std::vector<int64_t> device_get_negatives_total_ns;
    std::vector<int64_t> device_get_negatives_call_count;
    std::vector<int64_t> device_plan_lock_wait_ns;
    std::vector<int64_t> device_plan_lock_wait_count;
    std::vector<int64_t> device_state_pool_hit_count;
    std::vector<int64_t> device_planned_uniform_fetch_count;
    std::vector<int64_t> device_cuda_call_count;
    std::vector<int64_t> device_cpu_call_count;
    std::vector<int64_t> device_uniform_randint_ns;
    std::vector<int64_t> device_sample_edge_randint_ns;
    std::vector<int64_t> device_materialize_ns;
    std::vector<int64_t> device_filter_ns;
    std::vector<int64_t> device_filter_deg_chunk_ids_ns;
    std::vector<int64_t> device_filter_deg_mask_ns;
    std::vector<int64_t> device_filter_deg_nonzero_ns;
    std::vector<int64_t> device_filter_deg_gather_ns;
    std::vector<int64_t> device_filter_deg_finalize_ns;
    std::vector<int64_t> device_filter_gpu_prepare_ns;
    std::vector<int64_t> device_filter_gpu_searchsorted_ns;
    std::vector<int64_t> device_filter_gpu_offsets_ns;
    std::vector<int64_t> device_filter_gpu_repeat_interleave_ns;
    std::vector<int64_t> device_filter_gpu_neighbor_gather_ns;
    std::vector<int64_t> device_filter_gpu_relation_filter_ns;
    std::vector<int64_t> device_filter_gpu_finalize_ns;
    std::vector<std::vector<int64_t>> device_get_negatives_samples_ns;
    std::vector<std::vector<int64_t>> device_plan_lock_wait_samples_ns;
};

/**
 * Samples the negative edges from a given batch.
 */
class NegativeSampler {
   public:
    using SampleResult = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;
    using NodeCorruptResult = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;

    virtual ~NegativeSampler() {}

    /**
     * Get negative edges from the given batch.
     * Return a tensor of node ids of shape [num_negs] or a [num_negs, 3] shaped tensor of negative edges.
     * @param inverse Sample for inverse edges
     * @return The negative nodes/edges sampled
     */
    virtual std::tuple<torch::Tensor, torch::Tensor> getNegatives(shared_ptr<GegeGraph> graph, torch::Tensor edges = torch::Tensor(),
                                                                  bool inverse = false, int32_t device_idx = 0) = 0;
    virtual NodeCorruptResult getNodeCorruptNegatives(shared_ptr<GegeGraph> graph, torch::Tensor edges = torch::Tensor(),
                                                      bool need_src_negatives = true, int32_t device_idx = 0) {
        torch::Tensor src_negatives;
        torch::Tensor src_filter;
        if (need_src_negatives) {
            std::tie(src_negatives, src_filter) = getNegatives(graph, edges, true, device_idx);
        }

        torch::Tensor dst_negatives;
        torch::Tensor dst_filter;
        std::tie(dst_negatives, dst_filter) = getNegatives(graph, edges, false, device_idx);
        return std::forward_as_tuple(src_negatives, src_filter, dst_negatives, dst_filter);
    }
    virtual void resetPlanCache() {}
    virtual void resetPerfStats() {}
    virtual void initializePerfStats(std::size_t num_devices) {}
    virtual NegativeSamplerPerfStats getPerfStats() const { return NegativeSamplerPerfStats(); }
    // serve as `select` function.

    virtual std::tuple<torch::Tensor, torch::Tensor> compute(torch::Tensor src_embeddings, torch::Tensor dst_embeddings, torch::Tensor dst_neg_embeddings, torch::Tensor src_neg_embeddings,
                                                             torch::Tensor src_embeddings_g, torch::Tensor dst_embeddings_g, torch::Tensor dst_neg_embeddings_g, torch::Tensor src_neg_embeddings_g,
                                                             int batch_num, int embedding_size, int chunk_num, int num_per_chunk, bool has_relations, bool use_inverse_relations) {
        SPDLOG_INFO("NegativeSampling: compute needs override");
        return std::forward_as_tuple(torch::Tensor(), torch::Tensor());
    }

    virtual SampleResult sample(torch::Tensor dst_negs, torch::Tensor src_negs, torch::Tensor dst_negs_scores, torch::Tensor src_negs_scores,
                                int chunk_num, int num_per_chunk, int selected_negatives_num, bool has_relations, bool use_inverse_relations) {
        SPDLOG_INFO("NegativeSampling: sample needs override");
        return std::forward_as_tuple(torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor());
    }

};

class CorruptNodeNegativeSampler : public NegativeSampler {
   public:
    int num_chunks_;
    int num_negatives_;
    float degree_fraction_;
    bool filtered_;
    LocalFilterMode local_filter_mode_;

    CorruptNodeNegativeSampler(int num_chunks, int num_negatives, float degree_fraction, bool filtered = false,
                               LocalFilterMode local_filter_mode = LocalFilterMode::DEG);

    std::tuple<torch::Tensor, torch::Tensor> getNegatives(shared_ptr<GegeGraph> graph, torch::Tensor edges = torch::Tensor(),
                                                          bool inverse = false, int32_t device_idx = 0) override;
    NodeCorruptResult getNodeCorruptNegatives(shared_ptr<GegeGraph> graph, torch::Tensor edges = torch::Tensor(),
                                              bool need_src_negatives = true, int32_t device_idx = 0) override;
};

class CorruptRelNegativeSampler : public NegativeSampler {
   public:
    int num_chunks_;
    int num_negatives_;
    bool filtered_;

    CorruptRelNegativeSampler(int num_chunks, int num_negatives, bool filtered = false);

    std::tuple<torch::Tensor, torch::Tensor> getNegatives(shared_ptr<GegeGraph> graph, torch::Tensor edges = torch::Tensor(),
                                                          bool inverse = false, int32_t device_idx = 0) override;
};

class NegativeEdgeSampler : public NegativeSampler {
   public:
    int num_chunks_;
    int num_negatives_;

    NegativeEdgeSampler(int num_chunks, int num_negatives, bool filtered = false);

    std::tuple<torch::Tensor, torch::Tensor> getNegatives(shared_ptr<GegeGraph> graph, torch::Tensor edges = torch::Tensor(),
                                                          bool inverse = false, int32_t device_idx = 0) override;
};

// APIs.
class NegativeSamplingBase : public NegativeSampler {
   public:
    int num_chunks_;
    int num_negatives_;
    float degree_fraction_;
    int superbatch_negative_plan_batches_;
    bool filtered_;
    LocalFilterMode local_filter_mode_;
    bool tournament_selection_;
    bool tiled_tournament_scores_;
    int tiled_tournament_groups_per_tile_;
    int state_negative_pool_refresh_batches_;
    std::mutex plan_mutex_;
    std::map<std::string, std::deque<torch::Tensor>> planned_uniform_negatives_[2];

    NegativeSamplingBase(int num_chunks, int num_negatives, float degree_fraction, bool filtered = false,
                         int superbatch_negative_plan_batches = 0, LocalFilterMode local_filter_mode = LocalFilterMode::DEG, bool tournament_selection = false,
                         bool tiled_tournament_scores = false, int tiled_tournament_groups_per_tile = 64);

    std::tuple<torch::Tensor, torch::Tensor> getNegatives(shared_ptr<GegeGraph> graph, torch::Tensor edges = torch::Tensor(),
                                                          bool inverse = false, int32_t device_idx = 0) override;
    NodeCorruptResult getNodeCorruptNegatives(shared_ptr<GegeGraph> graph, torch::Tensor edges = torch::Tensor(),
                                              bool need_src_negatives = true, int32_t device_idx = 0) override;
    void resetPlanCache() override;
    void resetPerfStats() override;
    void initializePerfStats(std::size_t num_devices) override;
    NegativeSamplerPerfStats getPerfStats() const override;

   private:
    struct NegativePoolPlanCacheEntry {
        const void *graph_key = nullptr;
        int64_t num_nodes = -1;
        int64_t batch_size = -1;
        int remaining_uses = 0;
        torch::Tensor uniform_ids;
        torch::Tensor sample_edge_ids;
    };

    std::atomic<int64_t> get_negatives_total_ns_{0};
    std::atomic<int64_t> get_negatives_call_count_{0};
    std::atomic<int64_t> plan_lock_wait_ns_{0};
    std::atomic<int64_t> plan_lock_wait_count_{0};
    std::vector<int64_t> device_get_negatives_total_ns_;
    std::vector<int64_t> device_get_negatives_call_count_;
    std::vector<int64_t> device_plan_lock_wait_ns_;
    std::vector<int64_t> device_plan_lock_wait_count_;
    std::vector<int64_t> device_state_pool_hit_count_;
    std::vector<int64_t> device_planned_uniform_fetch_count_;
    std::vector<int64_t> device_cuda_call_count_;
    std::vector<int64_t> device_cpu_call_count_;
    std::vector<int64_t> device_uniform_randint_ns_;
    std::vector<int64_t> device_sample_edge_randint_ns_;
    std::vector<int64_t> device_materialize_ns_;
    std::vector<int64_t> device_filter_ns_;
    std::vector<int64_t> device_filter_deg_chunk_ids_ns_;
    std::vector<int64_t> device_filter_deg_mask_ns_;
    std::vector<int64_t> device_filter_deg_nonzero_ns_;
    std::vector<int64_t> device_filter_deg_gather_ns_;
    std::vector<int64_t> device_filter_deg_finalize_ns_;
    std::vector<int64_t> device_filter_gpu_prepare_ns_;
    std::vector<int64_t> device_filter_gpu_searchsorted_ns_;
    std::vector<int64_t> device_filter_gpu_offsets_ns_;
    std::vector<int64_t> device_filter_gpu_repeat_interleave_ns_;
    std::vector<int64_t> device_filter_gpu_neighbor_gather_ns_;
    std::vector<int64_t> device_filter_gpu_relation_filter_ns_;
    std::vector<int64_t> device_filter_gpu_finalize_ns_;
    std::vector<std::vector<int64_t>> device_get_negatives_samples_ns_;
    std::vector<std::vector<int64_t>> device_plan_lock_wait_samples_ns_;
    std::array<std::vector<NegativePoolPlanCacheEntry>, 2> state_negative_pool_plan_cache_;

    std::atomic<int64_t> state_pool_hit_count_{0};
    std::atomic<int64_t> planned_uniform_fetch_count_{0};
    std::atomic<int64_t> cuda_call_count_{0};
    std::atomic<int64_t> cpu_call_count_{0};
    std::atomic<int64_t> uniform_randint_ns_{0};
    std::atomic<int64_t> sample_edge_randint_ns_{0};
    std::atomic<int64_t> materialize_ns_{0};
    std::atomic<int64_t> filter_ns_{0};
    std::atomic<int64_t> filter_deg_chunk_ids_ns_{0};
    std::atomic<int64_t> filter_deg_mask_ns_{0};
    std::atomic<int64_t> filter_deg_nonzero_ns_{0};
    std::atomic<int64_t> filter_deg_gather_ns_{0};
    std::atomic<int64_t> filter_deg_finalize_ns_{0};
    std::atomic<int64_t> filter_gpu_prepare_ns_{0};
    std::atomic<int64_t> filter_gpu_searchsorted_ns_{0};
    std::atomic<int64_t> filter_gpu_offsets_ns_{0};
    std::atomic<int64_t> filter_gpu_repeat_interleave_ns_{0};
    std::atomic<int64_t> filter_gpu_neighbor_gather_ns_{0};
    std::atomic<int64_t> filter_gpu_relation_filter_ns_{0};
    std::atomic<int64_t> filter_gpu_finalize_ns_{0};
};

class RNS : public NegativeSamplingBase {
   public:
    RNS(int num_chunks, int num_negatives, float degree_fraction, bool filtered = false, int superbatch_negative_plan_batches = 0,
        LocalFilterMode local_filter_mode = LocalFilterMode::DEG, bool tournament_selection = false, bool tiled_tournament_scores = false,
        int tiled_tournament_groups_per_tile = 64)
       : NegativeSamplingBase(
             num_chunks, num_negatives, degree_fraction, filtered, superbatch_negative_plan_batches, local_filter_mode, tournament_selection,
             tiled_tournament_scores,
             tiled_tournament_groups_per_tile) {
           SPDLOG_INFO("NegativeSampling: Used RNS");
    }

    std::tuple<torch::Tensor, torch::Tensor> compute(torch::Tensor src_embeddings, torch::Tensor dst_embeddings, torch::Tensor dst_neg_embeddings, torch::Tensor src_neg_embeddings,
                                                             torch::Tensor src_embeddings_g, torch::Tensor dst_embeddings_g, torch::Tensor dst_neg_embeddings_g, torch::Tensor src_neg_embeddings_g,
                                                             int batch_num, int embedding_size, int chunk_num, int num_per_chunk, bool has_relations, bool use_inverse_relations) override {
        torch::Tensor dst_negs_scores;
        torch::Tensor src_negs_scores;
        return std::forward_as_tuple(dst_negs_scores, src_negs_scores);
    }

    SampleResult sample(torch::Tensor dst_negs, torch::Tensor src_negs, torch::Tensor dst_negs_scores, torch::Tensor src_negs_scores,
                        int chunk_num, int num_per_chunk, int selected_negatives_num, bool has_relations, bool use_inverse_relations) override {
        return std::forward_as_tuple(dst_negs, src_negs, torch::Tensor(), torch::Tensor());
    }

};

class DNS : public NegativeSamplingBase {
   public:
    DNS(int num_chunks, int num_negatives, float degree_fraction, bool filtered = false, int superbatch_negative_plan_batches = 0,
        LocalFilterMode local_filter_mode = LocalFilterMode::DEG, bool tournament_selection = false, bool tiled_tournament_scores = false,
        int tiled_tournament_groups_per_tile = 64)
       : NegativeSamplingBase(
             num_chunks, num_negatives, degree_fraction, filtered, superbatch_negative_plan_batches, local_filter_mode, tournament_selection,
             tiled_tournament_scores,
             tiled_tournament_groups_per_tile) {
           SPDLOG_INFO("NegativeSampling: Used DNS");
    }

    std::tuple<torch::Tensor, torch::Tensor> compute(torch::Tensor src_embeddings, torch::Tensor dst_embeddings, torch::Tensor dst_neg_embeddings, torch::Tensor src_neg_embeddings,
                                                             torch::Tensor src_embeddings_g, torch::Tensor dst_embeddings_g, torch::Tensor dst_neg_embeddings_g, torch::Tensor src_neg_embeddings_g,
                                                             int batch_num, int embedding_size, int chunk_num, int num_per_chunk, bool has_relations, bool use_inverse_relations) override {
        torch::Tensor dst_negs_scores;
        torch::Tensor src_negs_scores;
        // could have problem if batch_num < chunk_num when padding.
        torch::Tensor padded_src_embeddings = src_embeddings;
        torch::Tensor padded_dst_embeddings = dst_embeddings;
        if (num_per_chunk != batch_num / chunk_num) {
            int64_t new_size = num_per_chunk * chunk_num;
            torch::nn::functional::PadFuncOptions options({0, 0, 0, new_size - batch_num});
            padded_src_embeddings = torch::nn::functional::pad(src_embeddings, options);
            if (has_relations && use_inverse_relations) {
                padded_dst_embeddings = torch::nn::functional::pad(dst_embeddings, options);
            }
        }
        padded_src_embeddings = padded_src_embeddings.view({chunk_num, num_per_chunk, embedding_size});
        dst_negs_scores = padded_src_embeddings.bmm(dst_neg_embeddings.transpose(1, 2));
        if (has_relations && use_inverse_relations) {
            padded_dst_embeddings = padded_dst_embeddings.view({chunk_num, num_per_chunk, embedding_size});
            src_negs_scores = padded_dst_embeddings.bmm(src_neg_embeddings.transpose(1, 2));
        }
        return std::forward_as_tuple(dst_negs_scores, src_negs_scores);
    }

    SampleResult sample(torch::Tensor dst_negs, torch::Tensor src_negs, torch::Tensor dst_negs_scores, torch::Tensor src_negs_scores,
                        int chunk_num, int num_per_chunk, int selected_negatives_num, bool has_relations, bool use_inverse_relations) override {
        torch::Tensor dst_results_indices;
        torch::Tensor src_results_indices;

        dst_results_indices = tournament_select_indices(dst_negs_scores, selected_negatives_num, tournament_selection_);
        dst_results_indices = dst_results_indices.view({chunk_num, num_per_chunk * selected_negatives_num});

        if (has_relations && use_inverse_relations) {
            src_results_indices = tournament_select_indices(src_negs_scores, selected_negatives_num, tournament_selection_);
            src_results_indices = src_results_indices.view({chunk_num, num_per_chunk * selected_negatives_num});
        }

        return std::forward_as_tuple(torch::Tensor(), torch::Tensor(), dst_results_indices, src_results_indices);
    }
};

class KBGAN : public NegativeSamplingBase {
   public:
    KBGAN(int num_chunks, int num_negatives, float degree_fraction, bool filtered = false, int superbatch_negative_plan_batches = 0,
          LocalFilterMode local_filter_mode = LocalFilterMode::DEG, bool tournament_selection = false, bool tiled_tournament_scores = false,
          int tiled_tournament_groups_per_tile = 64)
       : NegativeSamplingBase(
             num_chunks, num_negatives, degree_fraction, filtered, superbatch_negative_plan_batches, local_filter_mode, tournament_selection,
             tiled_tournament_scores,
             tiled_tournament_groups_per_tile) {
           SPDLOG_INFO("NegativeSampling: Used KBGAN");
    }

    std::tuple<torch::Tensor, torch::Tensor> compute(torch::Tensor src_embeddings, torch::Tensor dst_embeddings, torch::Tensor dst_neg_embeddings, torch::Tensor src_neg_embeddings,
                                                             torch::Tensor src_embeddings_g, torch::Tensor dst_embeddings_g, torch::Tensor dst_neg_embeddings_g, torch::Tensor src_neg_embeddings_g,
                                                             int batch_num, int embedding_size, int chunk_num, int num_per_chunk, bool has_relations, bool use_inverse_relations) override {
        torch::Tensor dst_negs_scores;
        torch::Tensor src_negs_scores;
        // could have problem if batch_num < chunk_num when padding.
        torch::Tensor padded_src_embeddings_g = src_embeddings_g;
        torch::Tensor padded_dst_embeddings_g = dst_embeddings_g;
        if (num_per_chunk != batch_num / chunk_num) {
            int64_t new_size = num_per_chunk * chunk_num;
            torch::nn::functional::PadFuncOptions options({0, 0, 0, new_size - batch_num});
            padded_src_embeddings_g = torch::nn::functional::pad(src_embeddings_g, options);
            if (has_relations && use_inverse_relations) {
                padded_dst_embeddings_g = torch::nn::functional::pad(dst_embeddings_g, options);
            }
        }
        padded_src_embeddings_g = padded_src_embeddings_g.view({chunk_num, num_per_chunk, embedding_size});
        dst_negs_scores = padded_src_embeddings_g.bmm(dst_neg_embeddings_g.transpose(1, 2));
        if (has_relations && use_inverse_relations) {
            padded_dst_embeddings_g = padded_dst_embeddings_g.view({chunk_num, num_per_chunk, embedding_size});
            src_negs_scores = padded_dst_embeddings_g.bmm(src_neg_embeddings_g.transpose(1, 2));
        }
        return std::forward_as_tuple(dst_negs_scores, src_negs_scores);
    }

    SampleResult sample(torch::Tensor dst_negs, torch::Tensor src_negs, torch::Tensor dst_negs_scores, torch::Tensor src_negs_scores,
                        int chunk_num, int num_per_chunk, int selected_negatives_num, bool has_relations, bool use_inverse_relations) override {
        torch::Tensor dst_results_indices;
        torch::Tensor src_results_indices;

        dst_results_indices = tournament_select_indices(dst_negs_scores, selected_negatives_num, tournament_selection_);
        dst_results_indices = dst_results_indices.view({chunk_num, num_per_chunk * selected_negatives_num});

        if (has_relations && use_inverse_relations) {
            src_results_indices = tournament_select_indices(src_negs_scores, selected_negatives_num, tournament_selection_);
            src_results_indices = src_results_indices.view({chunk_num, num_per_chunk * selected_negatives_num});
        }

        return std::forward_as_tuple(torch::Tensor(), torch::Tensor(), dst_results_indices, src_results_indices);
    }
};
