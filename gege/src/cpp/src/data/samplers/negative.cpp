#include "data/samplers/negative.h"

#include <algorithm>
#include <chrono>

namespace {

std::string planned_uniform_cache_key(const torch::Device &device, int64_t num_nodes, int num_chunks, int num_uniform) {
    std::string key = device.str();
    key += "|";
    key += std::to_string(num_nodes);
    key += "|";
    key += std::to_string(num_chunks);
    key += "|";
    key += std::to_string(num_uniform);
    return key;
}

void initialize_negative_perf_vector(std::vector<int64_t> &values, std::size_t size) {
    values.assign(size, 0);
}

void initialize_negative_perf_samples(std::vector<std::vector<int64_t>> &values, std::size_t size) {
    values.assign(size, {});
}

void add_negative_perf_stat(std::atomic<int64_t> &aggregate, std::vector<int64_t> &per_device, int32_t device_idx, int64_t elapsed_ns) {
    aggregate.fetch_add(elapsed_ns);
    if (device_idx >= 0 && static_cast<std::size_t>(device_idx) < per_device.size()) {
        per_device[device_idx] += elapsed_ns;
    }
}

void add_negative_perf_count(std::atomic<int64_t> &aggregate, std::vector<int64_t> &per_device, int32_t device_idx) {
    aggregate.fetch_add(1);
    if (device_idx >= 0 && static_cast<std::size_t>(device_idx) < per_device.size()) {
        per_device[device_idx] += 1;
    }
}

void add_negative_perf_count(std::atomic<int64_t> &aggregate, std::vector<int64_t> &per_device, int32_t device_idx, int64_t amount) {
    aggregate.fetch_add(amount);
    if (device_idx >= 0 && static_cast<std::size_t>(device_idx) < per_device.size()) {
        per_device[device_idx] += amount;
    }
}

void add_negative_perf_sample(std::vector<std::vector<int64_t>> &per_device, int32_t device_idx, int64_t value) {
    if (device_idx >= 0 && static_cast<std::size_t>(device_idx) < per_device.size()) {
        per_device[device_idx].push_back(value);
    }
}

struct ChunkNegativePlan {
    torch::Tensor uniform_ids;
    torch::Tensor sample_edge_ids;
    int64_t uniform_randint_ns = 0;
    int64_t sample_edge_randint_ns = 0;
};

struct MaterializedNegativeOutput {
    torch::Tensor ids;
    torch::Tensor deg_sample_indices;
};

struct NegativeFilterBreakdown {
    int64_t deg_chunk_ids_ns = 0;
    int64_t deg_mask_ns = 0;
    int64_t deg_nonzero_ns = 0;
    int64_t deg_gather_ns = 0;
    int64_t deg_finalize_ns = 0;
    int64_t gpu_prepare_ns = 0;
    int64_t gpu_searchsorted_ns = 0;
    int64_t gpu_offsets_ns = 0;
    int64_t gpu_repeat_interleave_ns = 0;
    int64_t gpu_neighbor_gather_ns = 0;
    int64_t gpu_relation_filter_ns = 0;
    int64_t gpu_finalize_ns = 0;
};

thread_local NegativeFilterBreakdown *active_negative_filter_breakdown = nullptr;

class ScopedNegativeFilterBreakdownCapture {
   public:
    explicit ScopedNegativeFilterBreakdownCapture(NegativeFilterBreakdown *target) : previous_(active_negative_filter_breakdown) {
        active_negative_filter_breakdown = target;
    }

    ~ScopedNegativeFilterBreakdownCapture() { active_negative_filter_breakdown = previous_; }

   private:
    NegativeFilterBreakdown *previous_;
};

torch::Tensor materialize_degree_samples(torch::Tensor edges, torch::Tensor sample_edge_ids, bool inverse) {
    if (!sample_edge_ids.defined() || sample_edge_ids.numel() == 0) {
        return torch::Tensor();
    }

    torch::Tensor edge_nodes = edges.select(1, inverse ? 0 : -1);
    return edge_nodes.index_select(0, sample_edge_ids.reshape({-1})).view_as(sample_edge_ids);
}

ChunkNegativePlan build_chunk_negative_plan(torch::Tensor edges,
                                            int64_t num_nodes,
                                            int num_chunks,
                                            int num_uniform,
                                            int num_degree,
                                            torch::Tensor planned_uniform_ids) {
    ChunkNegativePlan plan;
    auto ind_opts = torch::TensorOptions().dtype(torch::kInt64).device(edges.device());
    int64_t batch_size = edges.size(0);

    if (num_uniform > 0) {
        if (planned_uniform_ids.defined()) {
            plan.uniform_ids = planned_uniform_ids;
        } else {
            auto uniform_start = std::chrono::high_resolution_clock::now();
            plan.uniform_ids = torch::randint(num_nodes, {num_chunks, num_uniform}, ind_opts);
            plan.uniform_randint_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          std::chrono::high_resolution_clock::now() - uniform_start)
                                          .count();
        }
    }

    if (num_degree > 0) {
        auto degree_start = std::chrono::high_resolution_clock::now();
        plan.sample_edge_ids = torch::randint(0, batch_size, {num_chunks, num_degree}, ind_opts);
        plan.sample_edge_randint_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          std::chrono::high_resolution_clock::now() - degree_start)
                                          .count();
    }

    return plan;
}

MaterializedNegativeOutput materialize_negative_output(torch::Tensor edges,
                                                       int64_t num_nodes,
                                                       int num_chunks,
                                                       int num_negatives,
                                                       float degree_fraction,
                                                       LocalFilterMode local_filter_mode,
                                                       torch::Tensor uniform_ids,
                                                       torch::Tensor sample_edge_ids,
                                                       bool inverse,
                                                       const torch::TensorOptions &ind_opts) {
    MaterializedNegativeOutput output;
    if (num_negatives == -1) {
        output.ids = torch::arange(num_nodes, ind_opts).view({1, num_nodes});
        return output;
    }

    torch::Tensor deg_sample = materialize_degree_samples(edges, sample_edge_ids, inverse);
    if (deg_sample.defined() && uniform_ids.defined()) {
        output.ids = torch::cat({deg_sample, uniform_ids}, 1);
    } else if (deg_sample.defined()) {
        output.ids = deg_sample;
    } else if (uniform_ids.defined()) {
        output.ids = uniform_ids;
    } else {
        output.ids = torch::empty({num_chunks, 0}, ind_opts);
    }

    if (degree_fraction > 0 && local_filter_mode == LocalFilterMode::DEG && sample_edge_ids.defined()) {
        output.deg_sample_indices = sample_edge_ids;
    }

    return output;
}

torch::Tensor build_negative_filter(shared_ptr<GegeGraph> graph,
                                    torch::Tensor edges,
                                    const MaterializedNegativeOutput &output,
                                    bool inverse,
                                    bool filtered,
                                    LocalFilterMode local_filter_mode,
                                    NegativeFilterBreakdown *breakdown = nullptr) {
    ScopedNegativeFilterBreakdownCapture capture(breakdown);
    return compute_filter_corruption(graph, edges, output.ids, inverse, filtered, local_filter_mode, output.deg_sample_indices);
}

void record_negative_perf_call(std::atomic<int64_t> &aggregate_total_ns,
                               std::atomic<int64_t> &aggregate_call_count,
                               std::vector<int64_t> &device_total_ns,
                               std::vector<int64_t> &device_call_count,
                               std::vector<std::vector<int64_t>> &device_samples_ns,
                               int32_t device_idx,
                               int64_t elapsed_ns,
                               int64_t logical_call_count) {
    add_negative_perf_stat(aggregate_total_ns, device_total_ns, device_idx, elapsed_ns);
    add_negative_perf_count(aggregate_call_count, device_call_count, device_idx, logical_call_count);
    int64_t sample_value = logical_call_count > 0 ? elapsed_ns / logical_call_count : elapsed_ns;
    for (int64_t i = 0; i < logical_call_count; i++) {
        add_negative_perf_sample(device_samples_ns, device_idx, sample_value);
    }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> batch_sample(torch::Tensor edges, int num_negatives, bool inverse) {
    auto device = edges.device();
    int64_t batch_size = edges.size(0);
    Indices sample_edge_id;
    sample_edge_id = torch::randint(0, batch_size, {num_negatives}, device).to(torch::kInt64);
    torch::Tensor edge_sample;

    if (inverse) {
        edge_sample = edges.index_select(0, sample_edge_id).select(1, 0);
    } else {
        edge_sample = edges.index_select(0, sample_edge_id).select(1, -1);
    }
    return std::forward_as_tuple(edge_sample, sample_edge_id);
}

torch::Tensor deg_negative_local_filter(torch::Tensor deg_sample_indices, torch::Tensor edges) {

    if (!deg_sample_indices.defined()) {
        torch::TensorOptions ind_opts = torch::TensorOptions().dtype(torch::kInt64).device(edges.device());
        return torch::empty({0, 2}, ind_opts);
    }

    int64_t num_chunks = deg_sample_indices.size(0);
    int64_t chunk_size = ceil((double)edges.size(0) / num_chunks);
    int64_t num_deg_negs = deg_sample_indices.size(1);

    auto chunk_ids_start = std::chrono::high_resolution_clock::now();
    torch::Tensor chunk_ids =
        torch::floor(deg_sample_indices.to(torch::kFloat64).div(static_cast<double>(chunk_size))).to(torch::kInt64);
    int64_t chunk_ids_elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - chunk_ids_start).count();

    auto mask_start = std::chrono::high_resolution_clock::now();
    torch::Tensor inv_mask = chunk_ids - torch::arange(0, num_chunks, deg_sample_indices.device()).view({num_chunks, -1});
    torch::Tensor mask = (inv_mask == 0);
    int64_t mask_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - mask_start).count();

    /*
        @zizhong
        Major bottleneck: torch::nonzero
    */
    auto nonzero_start = std::chrono::high_resolution_clock::now();
    torch::Tensor temp_idx = torch::nonzero(mask);
    int64_t nonzero_elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - nonzero_start).count();

    auto gather_start = std::chrono::high_resolution_clock::now();
    torch::Tensor id_offsets = deg_sample_indices.flatten(0, 1).index_select(0, temp_idx.select(1, 0) * num_deg_negs + temp_idx.select(1, 1));
    int64_t gather_elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - gather_start).count();

    auto finalize_start = std::chrono::high_resolution_clock::now();
    torch::Tensor filter = torch::stack({id_offsets, temp_idx.select(1, 1)}).transpose(0, 1);
    int64_t finalize_elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - finalize_start).count();
    if (active_negative_filter_breakdown != nullptr) {
        active_negative_filter_breakdown->deg_chunk_ids_ns += chunk_ids_elapsed;
        active_negative_filter_breakdown->deg_mask_ns += mask_elapsed;
        active_negative_filter_breakdown->deg_nonzero_ns += nonzero_elapsed;
        active_negative_filter_breakdown->deg_gather_ns += gather_elapsed;
        active_negative_filter_breakdown->deg_finalize_ns += finalize_elapsed;
    }
    return filter;
}

torch::Tensor compute_filter_corruption(shared_ptr<GegeGraph> graph, torch::Tensor edges, torch::Tensor corruption_nodes, bool inverse, bool global,
                                        LocalFilterMode local_filter_mode, torch::Tensor deg_sample_indices) {
    if (edges.is_cuda()) {
        return compute_filter_corruption_gpu(graph, edges, corruption_nodes, inverse, global, local_filter_mode, deg_sample_indices);
    } else {
        return compute_filter_corruption_cpu(graph, edges, corruption_nodes, inverse, global, local_filter_mode, deg_sample_indices);
    }
}

torch::Tensor compute_filter_corruption_cpu(shared_ptr<GegeGraph> graph, torch::Tensor edges, torch::Tensor corruption_nodes, bool inverse, bool global,
                                            LocalFilterMode local_filter_mode, torch::Tensor deg_sample_indices) {
    if (local_filter_mode == LocalFilterMode::DEG && !global) {
        return deg_negative_local_filter(deg_sample_indices, edges);
    }

    bool has_relations;

    if (edges.dim() == 3) {
        edges = edges.flatten(0, 1);
    } else if (edges.dim() != 2) {
        throw TensorSizeMismatchException(edges, "Edge list must have three (if chunked) or two dimensions");
    }

    if (edges.size(-1) == 3) {
        has_relations = true;
    } else if (edges.size(-1) == 2) {
        has_relations = false;
    } else {
        throw TensorSizeMismatchException(edges, "Edge list tensor must have 3 or 2 columns.");
    }

    int64_t num_chunks = corruption_nodes.size(0);
    int64_t num_edges = edges.size(0);
    int64_t chunk_size = ceil((double)num_edges / num_chunks);

    torch::Tensor all_sorted_edges;
    torch::Tensor all_sorted_nodes;
    torch::Tensor nodes;
    int tup_id;
    int corrupt_id;

    if (inverse) {
        if (has_relations) {
            tup_id = 2;
        } else {
            tup_id = 1;
        }

        corrupt_id = 0;

        nodes = edges.select(1, tup_id).contiguous();

        if (global) {
            if (graph->all_dst_sorted_edges_.defined()) {
                all_sorted_edges = graph->all_dst_sorted_edges_;
            } else {
                all_sorted_edges = graph->dst_sorted_edges_;
            }

        } else {
            all_sorted_edges = edges.index_select(0, nodes.argsort());
        }

        all_sorted_nodes = all_sorted_edges.select(1, tup_id).contiguous();

    } else {
        tup_id = 0;

        if (has_relations) {
            corrupt_id = 2;
        } else {
            corrupt_id = 1;
        }

        nodes = edges.select(1, tup_id).contiguous();

        if (global) {
            if (graph->all_src_sorted_edges_.defined()) {
                all_sorted_edges = graph->all_src_sorted_edges_;
            } else {
                all_sorted_edges = graph->src_sorted_edges_;
            }
        } else {
            all_sorted_edges = edges.index_select(0, nodes.argsort());
        }

        all_sorted_nodes = all_sorted_edges.select(1, tup_id).contiguous();
    }

    std::vector<std::vector<int64_t>> filters(num_edges);

    torch::Tensor starts = torch::searchsorted(all_sorted_nodes, nodes);
    torch::Tensor ends = torch::searchsorted(all_sorted_nodes, nodes + 1);

    auto edges_accessor = edges.accessor<int64_t, 2>();
    auto starts_accessor = starts.accessor<int64_t, 1>();
    auto ends_accessor = ends.accessor<int64_t, 1>();
    auto sorted_edges_accessor = all_sorted_edges.accessor<int64_t, 2>();
    auto negs_accessor = corruption_nodes.accessor<int64_t, 2>();

    if (global) {
#pragma omp parallel for
        for (int64_t edge_id = 0; edge_id < nodes.size(0); edge_id++) {
            int64_t curr_start = starts_accessor[edge_id];
            int64_t curr_end = ends_accessor[edge_id];

            for (int64_t curr = curr_start; curr < curr_end; curr++) {
                if ((has_relations && sorted_edges_accessor[curr][1] == edges_accessor[edge_id][1]) || !has_relations) {
                    filters[edge_id].emplace_back(sorted_edges_accessor[curr][corrupt_id]);
                }
            }
        }
    } else {
#pragma omp parallel for
        for (int64_t edge_id = 0; edge_id < nodes.size(0); edge_id++) {
            int64_t curr_start = starts_accessor[edge_id];
            int64_t curr_end = ends_accessor[edge_id];

            int chunk_id = edge_id / chunk_size;

            for (int64_t neg_id = 0; neg_id < corruption_nodes.size(1); neg_id++) {
                int64_t neg_node = negs_accessor[chunk_id][neg_id];

                for (int64_t curr = curr_start; curr < curr_end; curr++) {
                    if (sorted_edges_accessor[curr][corrupt_id] == neg_node) {
                        if ((has_relations && sorted_edges_accessor[curr][1] == edges_accessor[edge_id][1]) || !has_relations) {
                            filters[edge_id].emplace_back(neg_id);
                            break;
                        }
                    }
                }
            }
        }
    }

    int64_t num_filt = 0;

    for (int64_t edge_id = 0; edge_id < nodes.size(0); edge_id++) {
        num_filt += filters[edge_id].size();
    }

    torch::Tensor filter = torch::empty({num_filt, 2}, torch::kInt64);

    auto filter_accessor = filter.accessor<int64_t, 2>();

    int64_t offset = 0;
    for (int64_t edge_id = 0; edge_id < nodes.size(0); edge_id++) {
        for (int64_t j = 0; j < filters[edge_id].size(); j++) {
            filter_accessor[offset][0] = edge_id;
            filter_accessor[offset][1] = filters[edge_id][j];
            offset++;
        }
    }
    return filter;
}

torch::Tensor compute_filter_corruption_gpu(shared_ptr<GegeGraph> graph, torch::Tensor edges, torch::Tensor corruption_nodes, bool inverse, bool global,
                                            LocalFilterMode local_filter_mode, torch::Tensor deg_sample_indices) {
    if (local_filter_mode == LocalFilterMode::DEG && !global) {
        return deg_negative_local_filter(deg_sample_indices, edges);
    }

    bool has_relations;

    if (edges.dim() == 3) {
        edges = edges.flatten(0, 1);
    } else if (edges.dim() != 2) {
        throw TensorSizeMismatchException(edges, "Edge list must have three (if chunked) or two dimensions");
    }

    if (edges.size(-1) == 3) {
        has_relations = true;
    } else if (edges.size(-1) == 2) {
        has_relations = false;
    } else {
        throw TensorSizeMismatchException(edges, "Edge list tensor must have 3 or 2 columns.");
    }

    int64_t num_chunks = corruption_nodes.size(0);
    int64_t num_edges = edges.size(0);
    int64_t chunk_size = ceil((double)num_edges / num_chunks);

    int64_t negs_per_pos = corruption_nodes.size(1);

    torch::Tensor filter;
    torch::Tensor all_sorted_edges;
    torch::Tensor all_sorted_nodes;
    torch::Tensor nodes;
    int tup_id;
    int corrupt_id;
    int64_t prepare_elapsed = 0;
    int64_t searchsorted_elapsed = 0;
    int64_t offsets_elapsed = 0;
    int64_t repeat_interleave_elapsed = 0;
    int64_t neighbor_gather_elapsed = 0;
    int64_t relation_filter_elapsed = 0;
    int64_t finalize_elapsed = 0;

    auto prepare_start = std::chrono::high_resolution_clock::now();
    if (inverse) {
        if (has_relations) {
            tup_id = 2;
        } else {
            tup_id = 1;
        }

        corrupt_id = 0;

        nodes = edges.select(1, tup_id).contiguous();

        if (global) {
            all_sorted_edges = graph->all_dst_sorted_edges_;
        } else {
            all_sorted_edges = edges.index_select(0, nodes.argsort());
        }

        all_sorted_nodes = all_sorted_edges.select(1, tup_id).contiguous();
    } else {
        tup_id = 0;

        if (has_relations) {
            corrupt_id = 2;
        } else {
            corrupt_id = 1;
        }

        nodes = edges.select(1, tup_id).contiguous();

        if (global) {
            all_sorted_edges = graph->all_src_sorted_edges_;
        } else {
            all_sorted_edges = edges.index_select(0, nodes.argsort());
        }

        all_sorted_nodes = all_sorted_edges.select(1, tup_id).contiguous();
    }
    prepare_elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - prepare_start).count();

    auto searchsorted_start = std::chrono::high_resolution_clock::now();
    torch::Tensor starts = torch::searchsorted(all_sorted_nodes, nodes);
    torch::Tensor ends = torch::searchsorted(all_sorted_nodes, nodes + 1);
    searchsorted_elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - searchsorted_start).count();

    auto offsets_start = std::chrono::high_resolution_clock::now();
    torch::Tensor num_neighbors = ends - starts;
    torch::Tensor summed_num_neighbors = num_neighbors.cumsum(0);
    Indices local_offsets = summed_num_neighbors - num_neighbors;
    offsets_elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - offsets_start).count();

    if (global) {
        auto repeat_interleave_start = std::chrono::high_resolution_clock::now();
        torch::Tensor repeated_starts = starts.repeat_interleave(num_neighbors);
        torch::Tensor repeated_offsets = local_offsets.repeat_interleave(num_neighbors);
        torch::Tensor arange = torch::arange(repeated_offsets.size(0), edges.options());
        torch::Tensor sorted_list_idx = repeated_starts + arange - repeated_offsets;
        torch::Tensor edge_ids = torch::arange(edges.size(0), edges.options()).repeat_interleave(num_neighbors);
        repeat_interleave_elapsed =
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - repeat_interleave_start).count();

        auto neighbor_gather_start = std::chrono::high_resolution_clock::now();
        torch::Tensor batch_neighbors = all_sorted_edges.index_select(0, sorted_list_idx);
        neighbor_gather_elapsed =
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - neighbor_gather_start).count();

        if (has_relations) {
            auto relation_filter_start = std::chrono::high_resolution_clock::now();
            torch::Tensor filter_tmp_ids =
                torch::cat({edge_ids.view({-1, 1}), batch_neighbors.select(1, 1).view({-1, 1}), batch_neighbors.select(1, corrupt_id).view({-1, 1})}, 1);
            torch::Tensor rel_ids = edges.select(1, 1).repeat_interleave(num_neighbors);
            torch::Tensor mask = filter_tmp_ids.select(1, 1) == rel_ids;
            filter_tmp_ids = filter_tmp_ids.index_select(0, torch::arange(filter_tmp_ids.size(0), filter_tmp_ids.options()).masked_select(mask));
            relation_filter_elapsed =
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - relation_filter_start).count();
            auto finalize_start = std::chrono::high_resolution_clock::now();
            filter = torch::cat({filter_tmp_ids.select(1, 0).view({-1, 1}), filter_tmp_ids.select(1, 2).view({-1, 1})}, 1);
            finalize_elapsed =
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - finalize_start).count();
        } else {
            auto finalize_start = std::chrono::high_resolution_clock::now();
            filter = torch::cat({edge_ids.view({-1, 1}), batch_neighbors.select(1, corrupt_id).view({-1, 1})}, 1);
            finalize_elapsed =
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - finalize_start).count();
        }
    } else {
        // TODO implement local filtering on the GPU, filter needs to be an int64, shape [*, 2], unit tests for this would be good
        // like above when edges are int32 the filter may end up as int32
        //        torch::TensorOptions ind_opts = torch::TensorOptions().dtype(torch::kInt64).device(edges.device());
        //        filter = torch::empty({0, 2}, ind_opts);
        throw GegeRuntimeException("Local filtering against all edges in the batch not yet supported on GPU.");
    }
    if (active_negative_filter_breakdown != nullptr) {
        active_negative_filter_breakdown->gpu_prepare_ns += prepare_elapsed;
        active_negative_filter_breakdown->gpu_searchsorted_ns += searchsorted_elapsed;
        active_negative_filter_breakdown->gpu_offsets_ns += offsets_elapsed;
        active_negative_filter_breakdown->gpu_repeat_interleave_ns += repeat_interleave_elapsed;
        active_negative_filter_breakdown->gpu_neighbor_gather_ns += neighbor_gather_elapsed;
        active_negative_filter_breakdown->gpu_relation_filter_ns += relation_filter_elapsed;
        active_negative_filter_breakdown->gpu_finalize_ns += finalize_elapsed;
    }
    return filter;
}

torch::Tensor apply_score_filter(torch::Tensor scores, torch::Tensor filter) {
    if (filter.defined() && filter.size(0) > 0) {
        scores.index_put_({filter.select(1, 0), filter.select(1, 1)}, -1e9);
    }
    return scores;
}

CorruptNodeNegativeSampler::CorruptNodeNegativeSampler(int num_chunks, int num_negatives, float degree_fraction, bool filtered,
                                                       LocalFilterMode local_filter_mode) {
    num_chunks_ = num_chunks;
    num_negatives_ = num_negatives;
    degree_fraction_ = degree_fraction;
    filtered_ = filtered;
    local_filter_mode_ = local_filter_mode;

    if (filtered_) {
        num_chunks_ = 1;
        num_negatives_ = -1;
        degree_fraction_ = 0.0;
    }
}

std::tuple<torch::Tensor, torch::Tensor> CorruptNodeNegativeSampler::getNegatives(shared_ptr<GegeGraph> graph, torch::Tensor edges, bool inverse,
                                                                                  int32_t device_idx) {
    int64_t num_nodes = graph->num_nodes_in_memory_;

    int num_batch = (int)(num_negatives_ * degree_fraction_);
    int num_uni = num_negatives_ - num_batch;
    torch::TensorOptions ind_opts = torch::TensorOptions().dtype(torch::kInt64).device(edges.device());
    auto plan = build_chunk_negative_plan(edges, num_nodes, num_chunks_, num_uni, num_batch, torch::Tensor());
    auto output =
        materialize_negative_output(edges, num_nodes, num_chunks_, num_negatives_, degree_fraction_, local_filter_mode_, plan.uniform_ids,
                                    plan.sample_edge_ids, inverse, ind_opts);
    torch::Tensor score_filter = build_negative_filter(graph, edges, output, inverse, filtered_, local_filter_mode_);
    return std::forward_as_tuple(output.ids, score_filter);
}

NegativeSampler::NodeCorruptResult CorruptNodeNegativeSampler::getNodeCorruptNegatives(shared_ptr<GegeGraph> graph, torch::Tensor edges,
                                                                                       bool need_src_negatives, int32_t device_idx) {
    (void)device_idx;
    int64_t num_nodes = graph->num_nodes_in_memory_;
    int num_batch = (int)(num_negatives_ * degree_fraction_);
    int num_uni = num_negatives_ - num_batch;
    torch::TensorOptions ind_opts = torch::TensorOptions().dtype(torch::kInt64).device(edges.device());
    auto plan = build_chunk_negative_plan(edges, num_nodes, num_chunks_, num_uni, num_batch, torch::Tensor());

    torch::Tensor src_negatives;
    torch::Tensor src_filter;
    if (need_src_negatives) {
        auto src_output =
            materialize_negative_output(edges, num_nodes, num_chunks_, num_negatives_, degree_fraction_, local_filter_mode_, plan.uniform_ids,
                                        plan.sample_edge_ids, true, ind_opts);
        src_negatives = src_output.ids;
        src_filter = build_negative_filter(graph, edges, src_output, true, filtered_, local_filter_mode_);
    }

    auto dst_output =
        materialize_negative_output(edges, num_nodes, num_chunks_, num_negatives_, degree_fraction_, local_filter_mode_, plan.uniform_ids,
                                    plan.sample_edge_ids, false, ind_opts);
    torch::Tensor dst_filter = build_negative_filter(graph, edges, dst_output, false, filtered_, local_filter_mode_);
    return std::forward_as_tuple(src_negatives, src_filter, dst_output.ids, dst_filter);
}

NegativeSamplingBase::NegativeSamplingBase(int num_chunks, int num_negatives, float degree_fraction, bool filtered,
                                           int superbatch_negative_plan_batches, LocalFilterMode local_filter_mode, bool tournament_selection,
                                           bool tiled_tournament_scores, int tiled_tournament_groups_per_tile) {
    num_chunks_ = num_chunks;
    num_negatives_ = num_negatives;
    degree_fraction_ = degree_fraction;
    superbatch_negative_plan_batches_ = superbatch_negative_plan_batches;
    filtered_ = filtered;
    local_filter_mode_ = local_filter_mode;
    tournament_selection_ = tournament_selection;
    tiled_tournament_scores_ = tiled_tournament_scores;
    tiled_tournament_groups_per_tile_ = tiled_tournament_groups_per_tile;
    state_negative_pool_refresh_batches_ = std::max(parse_negative_env_int("GEGE_STATE_NEGATIVE_POOL_REFRESH_BATCHES", 0), 0);

    if (filtered_) {
        num_chunks_ = 1;
        num_negatives_ = -1;
        degree_fraction_ = 0.0;
        state_negative_pool_refresh_batches_ = 0;
    }
}

void NegativeSamplingBase::resetPlanCache() {
    std::lock_guard<std::mutex> lock(plan_mutex_);
    planned_uniform_negatives_[0].clear();
    planned_uniform_negatives_[1].clear();
    state_negative_pool_plan_cache_[0].clear();
    state_negative_pool_plan_cache_[1].clear();
}

void NegativeSamplingBase::initializePerfStats(std::size_t num_devices) {
    initialize_negative_perf_vector(device_get_negatives_total_ns_, num_devices);
    initialize_negative_perf_vector(device_get_negatives_call_count_, num_devices);
    initialize_negative_perf_vector(device_plan_lock_wait_ns_, num_devices);
    initialize_negative_perf_vector(device_plan_lock_wait_count_, num_devices);
    initialize_negative_perf_vector(device_state_pool_hit_count_, num_devices);
    initialize_negative_perf_vector(device_planned_uniform_fetch_count_, num_devices);
    initialize_negative_perf_vector(device_cuda_call_count_, num_devices);
    initialize_negative_perf_vector(device_cpu_call_count_, num_devices);
    initialize_negative_perf_vector(device_uniform_randint_ns_, num_devices);
    initialize_negative_perf_vector(device_sample_edge_randint_ns_, num_devices);
    initialize_negative_perf_vector(device_materialize_ns_, num_devices);
    initialize_negative_perf_vector(device_filter_ns_, num_devices);
    initialize_negative_perf_vector(device_filter_deg_chunk_ids_ns_, num_devices);
    initialize_negative_perf_vector(device_filter_deg_mask_ns_, num_devices);
    initialize_negative_perf_vector(device_filter_deg_nonzero_ns_, num_devices);
    initialize_negative_perf_vector(device_filter_deg_gather_ns_, num_devices);
    initialize_negative_perf_vector(device_filter_deg_finalize_ns_, num_devices);
    initialize_negative_perf_vector(device_filter_gpu_prepare_ns_, num_devices);
    initialize_negative_perf_vector(device_filter_gpu_searchsorted_ns_, num_devices);
    initialize_negative_perf_vector(device_filter_gpu_offsets_ns_, num_devices);
    initialize_negative_perf_vector(device_filter_gpu_repeat_interleave_ns_, num_devices);
    initialize_negative_perf_vector(device_filter_gpu_neighbor_gather_ns_, num_devices);
    initialize_negative_perf_vector(device_filter_gpu_relation_filter_ns_, num_devices);
    initialize_negative_perf_vector(device_filter_gpu_finalize_ns_, num_devices);
    initialize_negative_perf_samples(device_get_negatives_samples_ns_, num_devices);
    initialize_negative_perf_samples(device_plan_lock_wait_samples_ns_, num_devices);
    state_negative_pool_plan_cache_[0].assign(num_devices, NegativePoolPlanCacheEntry());
    state_negative_pool_plan_cache_[1].assign(num_devices, NegativePoolPlanCacheEntry());
}

void NegativeSamplingBase::resetPerfStats() {
    get_negatives_total_ns_.store(0);
    get_negatives_call_count_.store(0);
    plan_lock_wait_ns_.store(0);
    plan_lock_wait_count_.store(0);
    state_pool_hit_count_.store(0);
    planned_uniform_fetch_count_.store(0);
    cuda_call_count_.store(0);
    cpu_call_count_.store(0);
    uniform_randint_ns_.store(0);
    sample_edge_randint_ns_.store(0);
    materialize_ns_.store(0);
    filter_ns_.store(0);
    filter_deg_chunk_ids_ns_.store(0);
    filter_deg_mask_ns_.store(0);
    filter_deg_nonzero_ns_.store(0);
    filter_deg_gather_ns_.store(0);
    filter_deg_finalize_ns_.store(0);
    filter_gpu_prepare_ns_.store(0);
    filter_gpu_searchsorted_ns_.store(0);
    filter_gpu_offsets_ns_.store(0);
    filter_gpu_repeat_interleave_ns_.store(0);
    filter_gpu_neighbor_gather_ns_.store(0);
    filter_gpu_relation_filter_ns_.store(0);
    filter_gpu_finalize_ns_.store(0);
    std::fill(device_get_negatives_total_ns_.begin(), device_get_negatives_total_ns_.end(), 0);
    std::fill(device_get_negatives_call_count_.begin(), device_get_negatives_call_count_.end(), 0);
    std::fill(device_plan_lock_wait_ns_.begin(), device_plan_lock_wait_ns_.end(), 0);
    std::fill(device_plan_lock_wait_count_.begin(), device_plan_lock_wait_count_.end(), 0);
    std::fill(device_state_pool_hit_count_.begin(), device_state_pool_hit_count_.end(), 0);
    std::fill(device_planned_uniform_fetch_count_.begin(), device_planned_uniform_fetch_count_.end(), 0);
    std::fill(device_cuda_call_count_.begin(), device_cuda_call_count_.end(), 0);
    std::fill(device_cpu_call_count_.begin(), device_cpu_call_count_.end(), 0);
    std::fill(device_uniform_randint_ns_.begin(), device_uniform_randint_ns_.end(), 0);
    std::fill(device_sample_edge_randint_ns_.begin(), device_sample_edge_randint_ns_.end(), 0);
    std::fill(device_materialize_ns_.begin(), device_materialize_ns_.end(), 0);
    std::fill(device_filter_ns_.begin(), device_filter_ns_.end(), 0);
    std::fill(device_filter_deg_chunk_ids_ns_.begin(), device_filter_deg_chunk_ids_ns_.end(), 0);
    std::fill(device_filter_deg_mask_ns_.begin(), device_filter_deg_mask_ns_.end(), 0);
    std::fill(device_filter_deg_nonzero_ns_.begin(), device_filter_deg_nonzero_ns_.end(), 0);
    std::fill(device_filter_deg_gather_ns_.begin(), device_filter_deg_gather_ns_.end(), 0);
    std::fill(device_filter_deg_finalize_ns_.begin(), device_filter_deg_finalize_ns_.end(), 0);
    std::fill(device_filter_gpu_prepare_ns_.begin(), device_filter_gpu_prepare_ns_.end(), 0);
    std::fill(device_filter_gpu_searchsorted_ns_.begin(), device_filter_gpu_searchsorted_ns_.end(), 0);
    std::fill(device_filter_gpu_offsets_ns_.begin(), device_filter_gpu_offsets_ns_.end(), 0);
    std::fill(device_filter_gpu_repeat_interleave_ns_.begin(), device_filter_gpu_repeat_interleave_ns_.end(), 0);
    std::fill(device_filter_gpu_neighbor_gather_ns_.begin(), device_filter_gpu_neighbor_gather_ns_.end(), 0);
    std::fill(device_filter_gpu_relation_filter_ns_.begin(), device_filter_gpu_relation_filter_ns_.end(), 0);
    std::fill(device_filter_gpu_finalize_ns_.begin(), device_filter_gpu_finalize_ns_.end(), 0);
    for (auto &samples : device_get_negatives_samples_ns_) {
        samples.clear();
    }
    for (auto &samples : device_plan_lock_wait_samples_ns_) {
        samples.clear();
    }
}

NegativeSamplerPerfStats NegativeSamplingBase::getPerfStats() const {
    NegativeSamplerPerfStats stats;
    stats.get_negatives_total_ns = get_negatives_total_ns_.load();
    stats.get_negatives_call_count = get_negatives_call_count_.load();
    stats.plan_lock_wait_ns = plan_lock_wait_ns_.load();
    stats.plan_lock_wait_count = plan_lock_wait_count_.load();
    stats.state_pool_hit_count = state_pool_hit_count_.load();
    stats.planned_uniform_fetch_count = planned_uniform_fetch_count_.load();
    stats.cuda_call_count = cuda_call_count_.load();
    stats.cpu_call_count = cpu_call_count_.load();
    stats.uniform_randint_ns = uniform_randint_ns_.load();
    stats.sample_edge_randint_ns = sample_edge_randint_ns_.load();
    stats.materialize_ns = materialize_ns_.load();
    stats.filter_ns = filter_ns_.load();
    stats.filter_deg_chunk_ids_ns = filter_deg_chunk_ids_ns_.load();
    stats.filter_deg_mask_ns = filter_deg_mask_ns_.load();
    stats.filter_deg_nonzero_ns = filter_deg_nonzero_ns_.load();
    stats.filter_deg_gather_ns = filter_deg_gather_ns_.load();
    stats.filter_deg_finalize_ns = filter_deg_finalize_ns_.load();
    stats.filter_gpu_prepare_ns = filter_gpu_prepare_ns_.load();
    stats.filter_gpu_searchsorted_ns = filter_gpu_searchsorted_ns_.load();
    stats.filter_gpu_offsets_ns = filter_gpu_offsets_ns_.load();
    stats.filter_gpu_repeat_interleave_ns = filter_gpu_repeat_interleave_ns_.load();
    stats.filter_gpu_neighbor_gather_ns = filter_gpu_neighbor_gather_ns_.load();
    stats.filter_gpu_relation_filter_ns = filter_gpu_relation_filter_ns_.load();
    stats.filter_gpu_finalize_ns = filter_gpu_finalize_ns_.load();
    stats.device_get_negatives_total_ns = device_get_negatives_total_ns_;
    stats.device_get_negatives_call_count = device_get_negatives_call_count_;
    stats.device_plan_lock_wait_ns = device_plan_lock_wait_ns_;
    stats.device_plan_lock_wait_count = device_plan_lock_wait_count_;
    stats.device_state_pool_hit_count = device_state_pool_hit_count_;
    stats.device_planned_uniform_fetch_count = device_planned_uniform_fetch_count_;
    stats.device_cuda_call_count = device_cuda_call_count_;
    stats.device_cpu_call_count = device_cpu_call_count_;
    stats.device_uniform_randint_ns = device_uniform_randint_ns_;
    stats.device_sample_edge_randint_ns = device_sample_edge_randint_ns_;
    stats.device_materialize_ns = device_materialize_ns_;
    stats.device_filter_ns = device_filter_ns_;
    stats.device_filter_deg_chunk_ids_ns = device_filter_deg_chunk_ids_ns_;
    stats.device_filter_deg_mask_ns = device_filter_deg_mask_ns_;
    stats.device_filter_deg_nonzero_ns = device_filter_deg_nonzero_ns_;
    stats.device_filter_deg_gather_ns = device_filter_deg_gather_ns_;
    stats.device_filter_deg_finalize_ns = device_filter_deg_finalize_ns_;
    stats.device_filter_gpu_prepare_ns = device_filter_gpu_prepare_ns_;
    stats.device_filter_gpu_searchsorted_ns = device_filter_gpu_searchsorted_ns_;
    stats.device_filter_gpu_offsets_ns = device_filter_gpu_offsets_ns_;
    stats.device_filter_gpu_repeat_interleave_ns = device_filter_gpu_repeat_interleave_ns_;
    stats.device_filter_gpu_neighbor_gather_ns = device_filter_gpu_neighbor_gather_ns_;
    stats.device_filter_gpu_relation_filter_ns = device_filter_gpu_relation_filter_ns_;
    stats.device_filter_gpu_finalize_ns = device_filter_gpu_finalize_ns_;
    stats.device_get_negatives_samples_ns = device_get_negatives_samples_ns_;
    stats.device_plan_lock_wait_samples_ns = device_plan_lock_wait_samples_ns_;
    return stats;
}

std::tuple<torch::Tensor, torch::Tensor> NegativeSamplingBase::getNegatives(shared_ptr<GegeGraph> graph, torch::Tensor edges, bool inverse,
                                                                            int32_t device_idx) {
    auto get_negatives_start = std::chrono::high_resolution_clock::now();
    bool used_state_pool = false;
    bool used_planned_uniform = false;
    bool call_on_cuda = edges.is_cuda();
    int64_t uniform_randint_elapsed = 0;
    int64_t sample_edge_randint_elapsed = 0;
    int64_t materialize_elapsed = 0;
    int64_t filter_elapsed = 0;
    NegativeFilterBreakdown filter_breakdown;
    int64_t num_nodes = graph->num_nodes_in_memory_;
    int num_batch = (int)(num_negatives_ * degree_fraction_);
    int num_uni = num_negatives_ - num_batch;
    torch::TensorOptions ind_opts = torch::TensorOptions().dtype(torch::kInt64).device(edges.device());

    torch::Tensor uniform_ids;
    torch::Tensor sample_edge_ids;
    const int cache_id = inverse ? 1 : 0;
    const bool state_pool_enabled = state_negative_pool_refresh_batches_ > 1 && num_negatives_ != -1 && device_idx >= 0;

    if (state_pool_enabled) {
        std::lock_guard<std::mutex> lock(plan_mutex_);
        if (static_cast<std::size_t>(device_idx) >= state_negative_pool_plan_cache_[cache_id].size()) {
            state_negative_pool_plan_cache_[cache_id].resize(device_idx + 1);
        }

        auto &cache_entry = state_negative_pool_plan_cache_[cache_id][device_idx];
        if (cache_entry.graph_key == graph.get() && cache_entry.num_nodes == num_nodes && cache_entry.batch_size == edges.size(0) &&
            cache_entry.remaining_uses > 0) {
            uniform_ids = cache_entry.uniform_ids;
            sample_edge_ids = cache_entry.sample_edge_ids;
            cache_entry.remaining_uses--;
            used_state_pool = true;
        }
    }

    torch::Tensor planned_uniform_ids;
    if (!uniform_ids.defined() && num_negatives_ != -1 && num_uni > 0 && superbatch_negative_plan_batches_ > 0) {
        auto lock_wait_start = std::chrono::high_resolution_clock::now();
        std::lock_guard<std::mutex> lock(plan_mutex_);
        int64_t lock_wait_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        std::chrono::high_resolution_clock::now() - lock_wait_start)
                                        .count();
        add_negative_perf_stat(plan_lock_wait_ns_, device_plan_lock_wait_ns_, device_idx, lock_wait_elapsed);
        add_negative_perf_count(plan_lock_wait_count_, device_plan_lock_wait_count_, device_idx);
        add_negative_perf_sample(device_plan_lock_wait_samples_ns_, device_idx, lock_wait_elapsed);
        auto &queue = planned_uniform_negatives_[inverse ? 1 : 0][planned_uniform_cache_key(edges.device(), num_nodes, num_chunks_, num_uni)];
        while ((int)queue.size() < superbatch_negative_plan_batches_) {
            auto uniform_start = std::chrono::high_resolution_clock::now();
            queue.emplace_back(torch::randint(num_nodes, {num_chunks_, num_uni}, ind_opts));
            uniform_randint_elapsed += std::chrono::duration_cast<std::chrono::nanoseconds>(
                                           std::chrono::high_resolution_clock::now() - uniform_start)
                                           .count();
        }
        planned_uniform_ids = queue.front();
        queue.pop_front();
        used_planned_uniform = true;
    }

    bool need_uniform_ids = num_uni > 0 && !uniform_ids.defined();
    bool need_sample_edge_ids = num_batch > 0 && !sample_edge_ids.defined();
    if (need_uniform_ids || need_sample_edge_ids) {
        auto plan = build_chunk_negative_plan(edges, num_nodes, num_chunks_, num_uni, num_batch, planned_uniform_ids);
        uniform_randint_elapsed += plan.uniform_randint_ns;
        sample_edge_randint_elapsed += plan.sample_edge_randint_ns;
        if (need_uniform_ids) {
            uniform_ids = plan.uniform_ids;
        }
        if (need_sample_edge_ids) {
            sample_edge_ids = plan.sample_edge_ids;
        }
        if (state_pool_enabled) {
            std::lock_guard<std::mutex> lock(plan_mutex_);
            if (static_cast<std::size_t>(device_idx) >= state_negative_pool_plan_cache_[cache_id].size()) {
                state_negative_pool_plan_cache_[cache_id].resize(device_idx + 1);
            }
            auto &cache_entry = state_negative_pool_plan_cache_[cache_id][device_idx];
            cache_entry.graph_key = graph.get();
            cache_entry.num_nodes = num_nodes;
            cache_entry.batch_size = edges.size(0);
            cache_entry.remaining_uses = state_negative_pool_refresh_batches_ - 1;
            cache_entry.uniform_ids = uniform_ids;
            cache_entry.sample_edge_ids = sample_edge_ids;
        }
    }

    auto materialize_start = std::chrono::high_resolution_clock::now();
    auto output = materialize_negative_output(edges, num_nodes, num_chunks_, num_negatives_, degree_fraction_, local_filter_mode_, uniform_ids,
                                              sample_edge_ids, inverse, ind_opts);
    materialize_elapsed += std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::high_resolution_clock::now() - materialize_start)
                               .count();
    auto filter_start = std::chrono::high_resolution_clock::now();
    torch::Tensor score_filter = build_negative_filter(graph, edges, output, inverse, filtered_, local_filter_mode_, &filter_breakdown);
    filter_elapsed += std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::high_resolution_clock::now() - filter_start)
                          .count();
    int64_t get_negatives_elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - get_negatives_start).count();
    record_negative_perf_call(get_negatives_total_ns_, get_negatives_call_count_, device_get_negatives_total_ns_, device_get_negatives_call_count_,
                              device_get_negatives_samples_ns_, device_idx, get_negatives_elapsed, 1);
    if (used_state_pool) {
        add_negative_perf_count(state_pool_hit_count_, device_state_pool_hit_count_, device_idx, 1);
    }
    if (used_planned_uniform) {
        add_negative_perf_count(planned_uniform_fetch_count_, device_planned_uniform_fetch_count_, device_idx, 1);
    }
    if (call_on_cuda) {
        add_negative_perf_count(cuda_call_count_, device_cuda_call_count_, device_idx, 1);
    } else {
        add_negative_perf_count(cpu_call_count_, device_cpu_call_count_, device_idx, 1);
    }
    add_negative_perf_stat(uniform_randint_ns_, device_uniform_randint_ns_, device_idx, uniform_randint_elapsed);
    add_negative_perf_stat(sample_edge_randint_ns_, device_sample_edge_randint_ns_, device_idx, sample_edge_randint_elapsed);
    add_negative_perf_stat(materialize_ns_, device_materialize_ns_, device_idx, materialize_elapsed);
    add_negative_perf_stat(filter_ns_, device_filter_ns_, device_idx, filter_elapsed);
    add_negative_perf_stat(filter_deg_chunk_ids_ns_, device_filter_deg_chunk_ids_ns_, device_idx, filter_breakdown.deg_chunk_ids_ns);
    add_negative_perf_stat(filter_deg_mask_ns_, device_filter_deg_mask_ns_, device_idx, filter_breakdown.deg_mask_ns);
    add_negative_perf_stat(filter_deg_nonzero_ns_, device_filter_deg_nonzero_ns_, device_idx, filter_breakdown.deg_nonzero_ns);
    add_negative_perf_stat(filter_deg_gather_ns_, device_filter_deg_gather_ns_, device_idx, filter_breakdown.deg_gather_ns);
    add_negative_perf_stat(filter_deg_finalize_ns_, device_filter_deg_finalize_ns_, device_idx, filter_breakdown.deg_finalize_ns);
    add_negative_perf_stat(filter_gpu_prepare_ns_, device_filter_gpu_prepare_ns_, device_idx, filter_breakdown.gpu_prepare_ns);
    add_negative_perf_stat(filter_gpu_searchsorted_ns_, device_filter_gpu_searchsorted_ns_, device_idx, filter_breakdown.gpu_searchsorted_ns);
    add_negative_perf_stat(filter_gpu_offsets_ns_, device_filter_gpu_offsets_ns_, device_idx, filter_breakdown.gpu_offsets_ns);
    add_negative_perf_stat(filter_gpu_repeat_interleave_ns_, device_filter_gpu_repeat_interleave_ns_, device_idx, filter_breakdown.gpu_repeat_interleave_ns);
    add_negative_perf_stat(filter_gpu_neighbor_gather_ns_, device_filter_gpu_neighbor_gather_ns_, device_idx, filter_breakdown.gpu_neighbor_gather_ns);
    add_negative_perf_stat(filter_gpu_relation_filter_ns_, device_filter_gpu_relation_filter_ns_, device_idx, filter_breakdown.gpu_relation_filter_ns);
    add_negative_perf_stat(filter_gpu_finalize_ns_, device_filter_gpu_finalize_ns_, device_idx, filter_breakdown.gpu_finalize_ns);
    return std::forward_as_tuple(output.ids, score_filter);
}

NegativeSampler::NodeCorruptResult NegativeSamplingBase::getNodeCorruptNegatives(shared_ptr<GegeGraph> graph, torch::Tensor edges,
                                                                                 bool need_src_negatives, int32_t device_idx) {
    auto get_negatives_start = std::chrono::high_resolution_clock::now();
    bool used_state_pool = false;
    bool used_planned_uniform = false;
    bool call_on_cuda = edges.is_cuda();
    int64_t uniform_randint_elapsed = 0;
    int64_t sample_edge_randint_elapsed = 0;
    int64_t materialize_elapsed = 0;
    int64_t filter_elapsed = 0;
    NegativeFilterBreakdown filter_breakdown;
    int64_t num_nodes = graph->num_nodes_in_memory_;
    int num_batch = (int)(num_negatives_ * degree_fraction_);
    int num_uni = num_negatives_ - num_batch;
    torch::TensorOptions ind_opts = torch::TensorOptions().dtype(torch::kInt64).device(edges.device());

    torch::Tensor uniform_ids;
    torch::Tensor sample_edge_ids;
    const bool state_pool_enabled = state_negative_pool_refresh_batches_ > 1 && num_negatives_ != -1 && device_idx >= 0;

    if (state_pool_enabled) {
        std::lock_guard<std::mutex> lock(plan_mutex_);
        if (static_cast<std::size_t>(device_idx) >= state_negative_pool_plan_cache_[0].size()) {
            state_negative_pool_plan_cache_[0].resize(device_idx + 1);
            state_negative_pool_plan_cache_[1].resize(device_idx + 1);
        }

        auto &dst_cache_entry = state_negative_pool_plan_cache_[0][device_idx];
        auto &src_cache_entry = state_negative_pool_plan_cache_[1][device_idx];
        bool reuse_dst = dst_cache_entry.graph_key == graph.get() && dst_cache_entry.num_nodes == num_nodes &&
                         dst_cache_entry.batch_size == edges.size(0) && dst_cache_entry.remaining_uses > 0;
        bool reuse_src = !need_src_negatives ||
                         (src_cache_entry.graph_key == graph.get() && src_cache_entry.num_nodes == num_nodes &&
                          src_cache_entry.batch_size == edges.size(0) && src_cache_entry.remaining_uses > 0);
        if (reuse_dst && reuse_src) {
            uniform_ids = dst_cache_entry.uniform_ids;
            sample_edge_ids = dst_cache_entry.sample_edge_ids;
            dst_cache_entry.remaining_uses--;
            if (need_src_negatives) {
                src_cache_entry.remaining_uses--;
            }
            used_state_pool = true;
        }
    }

    torch::Tensor planned_uniform_ids;
    if (!uniform_ids.defined() && num_negatives_ != -1 && num_uni > 0 && superbatch_negative_plan_batches_ > 0) {
        auto lock_wait_start = std::chrono::high_resolution_clock::now();
        std::lock_guard<std::mutex> lock(plan_mutex_);
        int64_t lock_wait_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        std::chrono::high_resolution_clock::now() - lock_wait_start)
                                        .count();
        add_negative_perf_stat(plan_lock_wait_ns_, device_plan_lock_wait_ns_, device_idx, lock_wait_elapsed);
        add_negative_perf_count(plan_lock_wait_count_, device_plan_lock_wait_count_, device_idx);
        add_negative_perf_sample(device_plan_lock_wait_samples_ns_, device_idx, lock_wait_elapsed);
        auto &queue = planned_uniform_negatives_[0][planned_uniform_cache_key(edges.device(), num_nodes, num_chunks_, num_uni)];
        while ((int)queue.size() < superbatch_negative_plan_batches_) {
            auto uniform_start = std::chrono::high_resolution_clock::now();
            queue.emplace_back(torch::randint(num_nodes, {num_chunks_, num_uni}, ind_opts));
            uniform_randint_elapsed += std::chrono::duration_cast<std::chrono::nanoseconds>(
                                           std::chrono::high_resolution_clock::now() - uniform_start)
                                           .count();
        }
        planned_uniform_ids = queue.front();
        queue.pop_front();
        used_planned_uniform = true;
    }

    bool need_uniform_ids = num_uni > 0 && !uniform_ids.defined();
    bool need_sample_edge_ids = num_batch > 0 && !sample_edge_ids.defined();
    if (need_uniform_ids || need_sample_edge_ids) {
        auto plan = build_chunk_negative_plan(edges, num_nodes, num_chunks_, num_uni, num_batch, planned_uniform_ids);
        uniform_randint_elapsed += plan.uniform_randint_ns;
        sample_edge_randint_elapsed += plan.sample_edge_randint_ns;
        if (need_uniform_ids) {
            uniform_ids = plan.uniform_ids;
        }
        if (need_sample_edge_ids) {
            sample_edge_ids = plan.sample_edge_ids;
        }
        if (state_pool_enabled) {
            std::lock_guard<std::mutex> lock(plan_mutex_);
            if (static_cast<std::size_t>(device_idx) >= state_negative_pool_plan_cache_[0].size()) {
                state_negative_pool_plan_cache_[0].resize(device_idx + 1);
                state_negative_pool_plan_cache_[1].resize(device_idx + 1);
            }
            int max_cache_id = need_src_negatives ? 2 : 1;
            for (int cache_id = 0; cache_id < max_cache_id; cache_id++) {
                auto &cache_entry = state_negative_pool_plan_cache_[cache_id][device_idx];
                cache_entry.graph_key = graph.get();
                cache_entry.num_nodes = num_nodes;
                cache_entry.batch_size = edges.size(0);
                cache_entry.remaining_uses = state_negative_pool_refresh_batches_ - 1;
                cache_entry.uniform_ids = uniform_ids;
                cache_entry.sample_edge_ids = sample_edge_ids;
            }
        }
    }

    torch::Tensor shared_deg_filter;
    bool can_share_deg_filter = !filtered_ && local_filter_mode_ == LocalFilterMode::DEG && sample_edge_ids.defined();
    if (can_share_deg_filter) {
        auto filter_start = std::chrono::high_resolution_clock::now();
        ScopedNegativeFilterBreakdownCapture capture(&filter_breakdown);
        shared_deg_filter = deg_negative_local_filter(sample_edge_ids, edges);
        filter_elapsed += std::chrono::duration_cast<std::chrono::nanoseconds>(
                              std::chrono::high_resolution_clock::now() - filter_start)
                              .count();
    }

    torch::Tensor src_negatives;
    torch::Tensor src_filter;
    if (need_src_negatives) {
        auto materialize_start = std::chrono::high_resolution_clock::now();
        auto src_output = materialize_negative_output(edges, num_nodes, num_chunks_, num_negatives_, degree_fraction_, local_filter_mode_, uniform_ids,
                                                      sample_edge_ids, true, ind_opts);
        materialize_elapsed += std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::high_resolution_clock::now() - materialize_start)
                                   .count();
        src_negatives = src_output.ids;
        if (shared_deg_filter.defined()) {
            src_filter = shared_deg_filter;
        } else {
            auto filter_start = std::chrono::high_resolution_clock::now();
            src_filter = build_negative_filter(graph, edges, src_output, true, filtered_, local_filter_mode_, &filter_breakdown);
            filter_elapsed += std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  std::chrono::high_resolution_clock::now() - filter_start)
                                  .count();
        }
    }

    auto materialize_start = std::chrono::high_resolution_clock::now();
    auto dst_output = materialize_negative_output(edges, num_nodes, num_chunks_, num_negatives_, degree_fraction_, local_filter_mode_, uniform_ids,
                                                  sample_edge_ids, false, ind_opts);
    materialize_elapsed += std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::high_resolution_clock::now() - materialize_start)
                               .count();
    torch::Tensor dst_filter;
    if (shared_deg_filter.defined()) {
        dst_filter = shared_deg_filter;
    } else {
        auto filter_start = std::chrono::high_resolution_clock::now();
        dst_filter = build_negative_filter(graph, edges, dst_output, false, filtered_, local_filter_mode_, &filter_breakdown);
        filter_elapsed += std::chrono::duration_cast<std::chrono::nanoseconds>(
                              std::chrono::high_resolution_clock::now() - filter_start)
                              .count();
    }

    int64_t get_negatives_elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - get_negatives_start).count();
    int64_t logical_call_count = need_src_negatives ? 2 : 1;
    record_negative_perf_call(get_negatives_total_ns_, get_negatives_call_count_, device_get_negatives_total_ns_, device_get_negatives_call_count_,
                              device_get_negatives_samples_ns_, device_idx, get_negatives_elapsed, logical_call_count);
    if (used_state_pool) {
        add_negative_perf_count(state_pool_hit_count_, device_state_pool_hit_count_, device_idx, logical_call_count);
    }
    if (used_planned_uniform) {
        add_negative_perf_count(planned_uniform_fetch_count_, device_planned_uniform_fetch_count_, device_idx, logical_call_count);
    }
    if (call_on_cuda) {
        add_negative_perf_count(cuda_call_count_, device_cuda_call_count_, device_idx, logical_call_count);
    } else {
        add_negative_perf_count(cpu_call_count_, device_cpu_call_count_, device_idx, logical_call_count);
    }
    add_negative_perf_stat(uniform_randint_ns_, device_uniform_randint_ns_, device_idx, uniform_randint_elapsed);
    add_negative_perf_stat(sample_edge_randint_ns_, device_sample_edge_randint_ns_, device_idx, sample_edge_randint_elapsed);
    add_negative_perf_stat(materialize_ns_, device_materialize_ns_, device_idx, materialize_elapsed);
    add_negative_perf_stat(filter_ns_, device_filter_ns_, device_idx, filter_elapsed);
    add_negative_perf_stat(filter_deg_chunk_ids_ns_, device_filter_deg_chunk_ids_ns_, device_idx, filter_breakdown.deg_chunk_ids_ns);
    add_negative_perf_stat(filter_deg_mask_ns_, device_filter_deg_mask_ns_, device_idx, filter_breakdown.deg_mask_ns);
    add_negative_perf_stat(filter_deg_nonzero_ns_, device_filter_deg_nonzero_ns_, device_idx, filter_breakdown.deg_nonzero_ns);
    add_negative_perf_stat(filter_deg_gather_ns_, device_filter_deg_gather_ns_, device_idx, filter_breakdown.deg_gather_ns);
    add_negative_perf_stat(filter_deg_finalize_ns_, device_filter_deg_finalize_ns_, device_idx, filter_breakdown.deg_finalize_ns);
    add_negative_perf_stat(filter_gpu_prepare_ns_, device_filter_gpu_prepare_ns_, device_idx, filter_breakdown.gpu_prepare_ns);
    add_negative_perf_stat(filter_gpu_searchsorted_ns_, device_filter_gpu_searchsorted_ns_, device_idx, filter_breakdown.gpu_searchsorted_ns);
    add_negative_perf_stat(filter_gpu_offsets_ns_, device_filter_gpu_offsets_ns_, device_idx, filter_breakdown.gpu_offsets_ns);
    add_negative_perf_stat(filter_gpu_repeat_interleave_ns_, device_filter_gpu_repeat_interleave_ns_, device_idx, filter_breakdown.gpu_repeat_interleave_ns);
    add_negative_perf_stat(filter_gpu_neighbor_gather_ns_, device_filter_gpu_neighbor_gather_ns_, device_idx, filter_breakdown.gpu_neighbor_gather_ns);
    add_negative_perf_stat(filter_gpu_relation_filter_ns_, device_filter_gpu_relation_filter_ns_, device_idx, filter_breakdown.gpu_relation_filter_ns);
    add_negative_perf_stat(filter_gpu_finalize_ns_, device_filter_gpu_finalize_ns_, device_idx, filter_breakdown.gpu_finalize_ns);
    return std::forward_as_tuple(src_negatives, src_filter, dst_output.ids, dst_filter);
}
