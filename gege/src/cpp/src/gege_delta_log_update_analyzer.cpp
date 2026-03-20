#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "configuration/options.h"
#include "data/ordering.h"

namespace {

using CountMap = std::unordered_map<int64_t, int64_t>;

struct Args {
    std::string dataset_dir;
    EdgeBucketOrdering ordering;
    int num_partitions;
    int buffer_capacity;
    int fine_to_coarse_ratio;
    int num_cache_partitions;
    bool randomly_assign_edge_buckets;
    int active_devices;
    bool regroup = false;
    bool access_aware = false;
    bool access_aware_generate = false;
    int64_t seed = 12345;
    int64_t edge_sample_stride = 4096;
    int64_t batch_size = 10000;
    int64_t num_chunks = 10;
    int64_t negatives_per_positive = 1000;
    bool include_local_negatives = true;
    int64_t max_edges = -1;
    std::vector<int64_t> top_ks = {256, 1024, 4096};
    std::vector<int64_t> residency_windows = {1, 2, 4, 8, 16};
};

struct ResidencyStats {
    int64_t partition_id = -1;
    int64_t lane = -1;
    int64_t start_state_idx = -1;
    int64_t end_state_idx = -1;
    int64_t start_lane_pos = -1;
    int64_t end_lane_pos = -1;
    int64_t state_count = 0;
    int64_t batch_count = 0;
    int64_t total_batch_touches = 0;
    CountMap entity_batch_touch_counts;
};

struct SpanStats {
    int64_t unique_rows = 0;
    int64_t run_count = 0;
    int64_t max_run_length = 0;
    double mean_run_length = 0.0;
};

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " <dataset_dir> <ordering> <num_partitions> <buffer_capacity> <fine_to_coarse_ratio> <num_cache_partitions>"
                 " <randomly_assign_edge_buckets:0|1> <active_devices>"
                 " [--regroup] [--access-aware] [--access-aware-generate]"
                 " [--seed <int64>] [--edge-sample-stride <int64>]"
                 " [--batch-size <int64>] [--num-chunks <int64>] [--negatives-per-positive <int64>]"
                 " [--include-local-negatives 0|1] [--max-edges <int64>] [--top-ks <csv>]"
                 " [--residency-windows <csv>]\n";
}

std::vector<int64_t> parse_csv_int64(const std::string& text) {
    std::vector<int64_t> values;
    std::size_t start = 0;
    while (start < text.size()) {
        auto comma = text.find(',', start);
        auto token = text.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (!token.empty()) {
            values.emplace_back(std::stoll(token));
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return values;
}

std::vector<int64_t> tensor_to_vector(torch::Tensor tensor) {
    tensor = tensor.to(torch::kCPU).to(torch::kInt64).contiguous();
    auto* data = tensor.data_ptr<int64_t>();
    return std::vector<int64_t>(data, data + tensor.numel());
}

std::vector<int64_t> read_offsets(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open offsets file: " + path);
    }
    std::vector<int64_t> offsets;
    int64_t value = 0;
    while (in >> value) {
        offsets.emplace_back(value);
    }
    return offsets;
}

int64_t read_num_nodes(const std::string& dataset_yaml_path) {
    std::ifstream in(dataset_yaml_path);
    if (!in) {
        throw std::runtime_error("Failed to open dataset yaml: " + dataset_yaml_path);
    }
    std::string line;
    while (std::getline(in, line)) {
        auto pos = line.find("num_nodes:");
        if (pos != std::string::npos) {
            std::string value = line.substr(pos + std::string("num_nodes:").size());
            value.erase(0, value.find_first_not_of(" \t"));
            return std::stoll(value);
        }
    }
    throw std::runtime_error("num_nodes not found in dataset yaml");
}

int infer_edge_columns(const std::string& path, int64_t num_edges) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("Failed to open edges file: " + path);
    }
    int64_t bytes = static_cast<int64_t>(in.tellg());
    if (num_edges <= 0) {
        throw std::runtime_error("num_edges must be positive");
    }
    int64_t bytes_per_edge = bytes / num_edges;
    if (bytes % num_edges != 0) {
        throw std::runtime_error("Edge file size does not divide evenly by number of edges");
    }
    if (bytes_per_edge == 8 || bytes_per_edge == 16) return 2;
    if (bytes_per_edge == 12 || bytes_per_edge == 24) return 3;
    throw std::runtime_error("Unsupported bytes per edge: " + std::to_string(bytes_per_edge));
}

int infer_edge_dtype_bytes(const std::string& path, int64_t num_edges, int edge_cols) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("Failed to open edges file: " + path);
    }
    int64_t bytes = static_cast<int64_t>(in.tellg());
    int64_t bytes_per_value = bytes / (num_edges * edge_cols);
    if (bytes_per_value != 4 && bytes_per_value != 8) {
        throw std::runtime_error("Unsupported edge value width: " + std::to_string(bytes_per_value));
    }
    return static_cast<int>(bytes_per_value);
}

Args parse_args(int argc, char** argv) {
    if (argc < 9) {
        print_usage(argv[0]);
        throw std::runtime_error("Not enough arguments");
    }

    Args args;
    args.dataset_dir = argv[1];
    args.ordering = getEdgeBucketOrderingEnum(argv[2]);
    args.num_partitions = std::stoi(argv[3]);
    args.buffer_capacity = std::stoi(argv[4]);
    args.fine_to_coarse_ratio = std::stoi(argv[5]);
    args.num_cache_partitions = std::stoi(argv[6]);
    args.randomly_assign_edge_buckets = std::stoi(argv[7]) != 0;
    args.active_devices = std::stoi(argv[8]);

    for (int i = 9; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--regroup") {
            args.regroup = true;
        } else if (arg == "--access-aware") {
            args.access_aware = true;
        } else if (arg == "--access-aware-generate") {
            args.access_aware_generate = true;
        } else if (arg == "--seed" && i + 1 < argc) {
            args.seed = std::stoll(argv[++i]);
        } else if (arg == "--edge-sample-stride" && i + 1 < argc) {
            args.edge_sample_stride = std::max<int64_t>(1, std::stoll(argv[++i]));
        } else if (arg == "--batch-size" && i + 1 < argc) {
            args.batch_size = std::max<int64_t>(1, std::stoll(argv[++i]));
        } else if (arg == "--num-chunks" && i + 1 < argc) {
            args.num_chunks = std::max<int64_t>(1, std::stoll(argv[++i]));
        } else if (arg == "--negatives-per-positive" && i + 1 < argc) {
            args.negatives_per_positive = std::max<int64_t>(1, std::stoll(argv[++i]));
        } else if (arg == "--include-local-negatives" && i + 1 < argc) {
            args.include_local_negatives = std::stoi(argv[++i]) != 0;
        } else if (arg == "--max-edges" && i + 1 < argc) {
            args.max_edges = std::stoll(argv[++i]);
        } else if (arg == "--top-ks" && i + 1 < argc) {
            args.top_ks = parse_csv_int64(argv[++i]);
        } else if (arg == "--residency-windows" && i + 1 < argc) {
            args.residency_windows = parse_csv_int64(argv[++i]);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    args.top_ks.erase(std::remove_if(args.top_ks.begin(), args.top_ks.end(), [](int64_t value) { return value <= 0; }), args.top_ks.end());
    if (args.top_ks.empty()) {
        args.top_ks = {256, 1024, 4096};
    }
    std::sort(args.top_ks.begin(), args.top_ks.end());
    args.top_ks.erase(std::unique(args.top_ks.begin(), args.top_ks.end()), args.top_ks.end());

    args.residency_windows.erase(
        std::remove_if(args.residency_windows.begin(), args.residency_windows.end(), [](int64_t value) { return value <= 0; }),
        args.residency_windows.end());
    if (args.residency_windows.empty()) {
        args.residency_windows = {1, 2, 4, 8, 16};
    }
    std::sort(args.residency_windows.begin(), args.residency_windows.end());
    args.residency_windows.erase(std::unique(args.residency_windows.begin(), args.residency_windows.end()),
                                 args.residency_windows.end());

    return args;
}

std::vector<int64_t> build_bucket_to_state(const std::vector<torch::Tensor>& edge_buckets_per_buffer, int num_partitions) {
    std::vector<int64_t> bucket_to_state(num_partitions * num_partitions, -1);
    for (std::size_t state_idx = 0; state_idx < edge_buckets_per_buffer.size(); state_idx++) {
        auto buckets = edge_buckets_per_buffer[state_idx].to(torch::kCPU).to(torch::kInt64).contiguous();
        auto acc = buckets.accessor<int64_t, 2>();
        for (int64_t i = 0; i < buckets.size(0); i++) {
            int64_t bucket_idx = acc[i][0] * num_partitions + acc[i][1];
            if (bucket_idx < 0 || bucket_idx >= static_cast<int64_t>(bucket_to_state.size())) {
                throw std::runtime_error("Bucket index out of range");
            }
            if (bucket_to_state[bucket_idx] != -1) {
                throw std::runtime_error("Edge bucket assigned to multiple states");
            }
            bucket_to_state[bucket_idx] = static_cast<int64_t>(state_idx);
        }
    }
    return bucket_to_state;
}

std::vector<std::vector<int64_t>> build_lane_states(std::size_t num_states, int active_devices) {
    std::vector<std::vector<int64_t>> lanes(active_devices);
    for (int lane = 0; lane < active_devices; lane++) {
        for (std::size_t idx = static_cast<std::size_t>(lane); idx < num_states; idx += static_cast<std::size_t>(active_devices)) {
            lanes[lane].push_back(static_cast<int64_t>(idx));
        }
    }
    return lanes;
}

std::vector<std::vector<int64_t>> build_partition_segments(const std::vector<int64_t>& lane_states,
                                                           const std::vector<std::vector<int64_t>>& state_partitions,
                                                           int num_partitions) {
    std::vector<std::vector<int64_t>> segment_ids(num_partitions);
    for (int partition_id = 0; partition_id < num_partitions; partition_id++) {
        std::vector<int64_t> current_segment;
        bool active = false;
        for (auto state_idx : lane_states) {
            const auto& parts = state_partitions[state_idx];
            bool present = std::find(parts.begin(), parts.end(), partition_id) != parts.end();
            if (present) {
                current_segment.emplace_back(state_idx);
                active = true;
            } else if (active) {
                segment_ids[partition_id].insert(segment_ids[partition_id].end(), current_segment.begin(), current_segment.end());
                current_segment.clear();
                active = false;
            }
        }
        if (active) {
            segment_ids[partition_id].insert(segment_ids[partition_id].end(), current_segment.begin(), current_segment.end());
        }
    }
    return segment_ids;
}

int64_t node_partition(int64_t node_id, int64_t partition_size, int num_partitions) {
    int64_t part = node_id / partition_size;
    if (part < 0) {
        return 0;
    }
    if (part >= num_partitions) {
        return num_partitions - 1;
    }
    return part;
}

std::vector<std::pair<int64_t, int64_t>> sort_counts_desc(const CountMap& counts) {
    std::vector<std::pair<int64_t, int64_t>> ordered(counts.begin(), counts.end());
    std::sort(ordered.begin(), ordered.end(), [](const auto& left, const auto& right) {
        if (left.second != right.second) {
            return left.second > right.second;
        }
        return left.first < right.first;
    });
    return ordered;
}

std::vector<int64_t> sorted_local_rows(const ResidencyStats& segment, int64_t partition_size) {
    int64_t partition_start = segment.partition_id * partition_size;
    std::vector<int64_t> rows;
    rows.reserve(segment.entity_batch_touch_counts.size());
    for (const auto& [entity_id, _] : segment.entity_batch_touch_counts) {
        rows.emplace_back(entity_id - partition_start);
    }
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    return rows;
}

SpanStats analyze_spans(const std::vector<int64_t>& rows) {
    SpanStats stats;
    stats.unique_rows = static_cast<int64_t>(rows.size());
    if (rows.empty()) {
        return stats;
    }

    int64_t run_start = rows.front();
    int64_t prev = rows.front();
    int64_t total_run_len = 0;
    int64_t run_count = 0;
    int64_t max_run_length = 0;
    for (std::size_t i = 1; i < rows.size(); i++) {
        if (rows[i] == prev + 1) {
            prev = rows[i];
            continue;
        }
        int64_t run_len = prev - run_start + 1;
        total_run_len += run_len;
        run_count += 1;
        max_run_length = std::max(max_run_length, run_len);
        run_start = rows[i];
        prev = rows[i];
    }
    int64_t final_run_len = prev - run_start + 1;
    total_run_len += final_run_len;
    run_count += 1;
    max_run_length = std::max(max_run_length, final_run_len);

    stats.run_count = run_count;
    stats.max_run_length = max_run_length;
    stats.mean_run_length = run_count > 0 ? static_cast<double>(total_run_len) / static_cast<double>(run_count) : 0.0;
    return stats;
}

double touched_chunk_fill(const std::vector<int64_t>& rows, int64_t chunk_size) {
    if (rows.empty() || chunk_size <= 0) {
        return 0.0;
    }
    std::unordered_set<int64_t> chunks;
    chunks.reserve(rows.size());
    for (auto row : rows) {
        chunks.insert(row / chunk_size);
    }
    if (chunks.empty()) {
        return 0.0;
    }
    return static_cast<double>(rows.size()) / static_cast<double>(chunks.size() * chunk_size);
}

double touched_chunk_fraction(const std::vector<int64_t>& rows, int64_t chunk_size, int64_t partition_rows) {
    if (rows.empty() || chunk_size <= 0 || partition_rows <= 0) {
        return 0.0;
    }
    std::unordered_set<int64_t> chunks;
    chunks.reserve(rows.size());
    for (auto row : rows) {
        chunks.insert(row / chunk_size);
    }
    int64_t total_chunks = (partition_rows + chunk_size - 1) / chunk_size;
    if (total_chunks <= 0) {
        return 0.0;
    }
    return static_cast<double>(chunks.size()) / static_cast<double>(total_chunks);
}

double top_k_coverage(const CountMap& counts, int64_t k) {
    if (counts.empty() || k <= 0) {
        return 0.0;
    }
    auto ordered = sort_counts_desc(counts);
    int64_t total = 0;
    int64_t covered = 0;
    for (const auto& [_, count] : ordered) {
        total += count;
    }
    for (int64_t i = 0; i < k && i < static_cast<int64_t>(ordered.size()); i++) {
        covered += ordered[i].second;
    }
    return total > 0 ? static_cast<double>(covered) / static_cast<double>(total) : 0.0;
}

void finalize_batch_updates(std::vector<std::unordered_set<int64_t>>& touched_per_partition,
                            std::vector<ResidencyStats>& active_segments,
                            const std::unordered_map<int64_t, int64_t>& state_to_segment,
                            std::unordered_map<int64_t, int64_t>& state_partition_batch_touches) {
    for (std::size_t partition_id = 0; partition_id < touched_per_partition.size(); partition_id++) {
        auto& touched = touched_per_partition[partition_id];
        if (touched.empty()) {
            continue;
        }
        state_partition_batch_touches[static_cast<int64_t>(partition_id)] += static_cast<int64_t>(touched.size());
        auto seg_it = state_to_segment.find(static_cast<int64_t>(partition_id));
        if (seg_it == state_to_segment.end()) {
            touched.clear();
            continue;
        }
        auto& segment = active_segments[seg_it->second];
        segment.batch_count += 1;
        segment.total_batch_touches += static_cast<int64_t>(touched.size());
        for (auto entity_id : touched) {
            segment.entity_batch_touch_counts[entity_id] += 1;
        }
        touched.clear();
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        std::mt19937_64 rng(static_cast<uint64_t>(args.seed));
        torch::manual_seed(args.seed);

        const std::string edge_path = args.dataset_dir + "/edges/train_edges.bin";
        const std::string offsets_path = args.dataset_dir + "/edges/train_partition_offsets.txt";
        const std::string dataset_yaml_path = args.dataset_dir + "/dataset.yaml";
        const int64_t num_nodes = read_num_nodes(dataset_yaml_path);
        const int64_t partition_size = static_cast<int64_t>(std::ceil(static_cast<double>(num_nodes) / static_cast<double>(args.num_partitions)));
        auto bucket_sizes = read_offsets(offsets_path);
        if (bucket_sizes.size() != static_cast<std::size_t>(args.num_partitions * args.num_partitions)) {
            throw std::runtime_error("Unexpected number of edge buckets in offsets file");
        }
        const int64_t num_edges = std::accumulate(bucket_sizes.begin(), bucket_sizes.end(), static_cast<int64_t>(0));
        const int edge_cols = infer_edge_columns(edge_path, num_edges);
        const int dtype_bytes = infer_edge_dtype_bytes(edge_path, num_edges, edge_cols);

        auto ordering_result = args.access_aware_generate
            ? getAccessAwareCustomEdgeBucketOrdering(args.num_partitions, args.buffer_capacity, args.active_devices)
            : getEdgeBucketOrdering(args.ordering,
                                    args.num_partitions,
                                    args.buffer_capacity,
                                    args.fine_to_coarse_ratio,
                                    args.num_cache_partitions,
                                    args.randomly_assign_edge_buckets);
        auto buffer_states = std::get<0>(ordering_result);
        auto edge_buckets_per_buffer = std::get<1>(ordering_result);
        if (!args.access_aware_generate && (args.access_aware || args.regroup)) {
            auto permutation = args.access_aware
                ? getAccessAwareDisjointBufferStatePermutation(buffer_states, edge_buckets_per_buffer, args.active_devices)
                : getDisjointBufferStatePermutation(buffer_states, args.active_devices);
            std::vector<torch::Tensor> reordered_states;
            std::vector<torch::Tensor> reordered_buckets;
            reordered_states.reserve(buffer_states.size());
            reordered_buckets.reserve(edge_buckets_per_buffer.size());
            for (auto idx : permutation) {
                reordered_states.emplace_back(buffer_states[idx]);
                reordered_buckets.emplace_back(edge_buckets_per_buffer[idx]);
            }
            buffer_states = std::move(reordered_states);
            edge_buckets_per_buffer = std::move(reordered_buckets);
        }

        auto bucket_to_state = build_bucket_to_state(edge_buckets_per_buffer, args.num_partitions);
        std::vector<std::vector<int64_t>> state_partitions;
        state_partitions.reserve(buffer_states.size());
        for (const auto& state : buffer_states) {
            state_partitions.emplace_back(tensor_to_vector(state));
        }

        auto lanes = build_lane_states(buffer_states.size(), args.active_devices);
        std::vector<ResidencyStats> segments;
        std::vector<std::unordered_map<int64_t, int64_t>> state_to_segment(buffer_states.size());
        for (int lane = 0; lane < static_cast<int>(lanes.size()); lane++) {
            const auto& lane_states = lanes[lane];
            for (int partition_id = 0; partition_id < args.num_partitions; partition_id++) {
                bool active = false;
                ResidencyStats current;
                for (int64_t lane_pos = 0; lane_pos < static_cast<int64_t>(lane_states.size()); lane_pos++) {
                    auto state_idx = lane_states[lane_pos];
                    const auto& parts = state_partitions[state_idx];
                    bool present = std::find(parts.begin(), parts.end(), partition_id) != parts.end();
                    if (present) {
                        if (!active) {
                            current = ResidencyStats{};
                            current.partition_id = partition_id;
                            current.lane = lane;
                            current.start_state_idx = state_idx;
                            current.start_lane_pos = lane_pos;
                            active = true;
                        }
                        current.end_state_idx = state_idx;
                        current.end_lane_pos = lane_pos;
                        current.state_count += 1;
                    } else if (active) {
                        int64_t segment_idx = static_cast<int64_t>(segments.size());
                        for (int64_t idx = current.start_state_idx; idx <= current.end_state_idx; idx += args.active_devices) {
                            state_to_segment[idx][partition_id] = segment_idx;
                        }
                        segments.emplace_back(std::move(current));
                        active = false;
                    }
                }
                if (active) {
                    int64_t segment_idx = static_cast<int64_t>(segments.size());
                    for (int64_t idx = current.start_state_idx; idx <= current.end_state_idx; idx += args.active_devices) {
                        state_to_segment[idx][partition_id] = segment_idx;
                    }
                    segments.emplace_back(std::move(current));
                }
            }
        }

        std::ifstream edge_in(edge_path, std::ios::binary);
        if (!edge_in) {
            throw std::runtime_error("Failed to open edge file");
        }
        constexpr int64_t rows_per_chunk = 1 << 16;
        std::vector<char> raw(rows_per_chunk * edge_cols * dtype_bytes);

        auto decode_value = [&](const char* ptr) -> int64_t {
            if (dtype_bytes == 4) {
                int32_t value = 0;
                std::memcpy(&value, ptr, sizeof(int32_t));
                return static_cast<int64_t>(value);
            }
            int64_t value = 0;
            std::memcpy(&value, ptr, sizeof(int64_t));
            return value;
        };

        std::vector<std::unordered_set<int64_t>> batch_touched_per_partition(args.num_partitions);
        std::vector<std::unordered_map<int64_t, int64_t>> state_partition_batch_touches(buffer_states.size());
        int64_t raw_edges_in_state_batch = 0;
        int64_t current_state_idx = -1;
        std::vector<int64_t> current_state_partitions;
        int64_t processed_raw_edges_total = 0;

        auto sample_local_negative = [&](const std::vector<int64_t>& parts) -> int64_t {
            std::uniform_int_distribution<int> part_dist(0, static_cast<int>(parts.size()) - 1);
            int64_t partition_id = parts[part_dist(rng)];
            int64_t start = partition_id * partition_size;
            int64_t end = std::min<int64_t>(num_nodes, start + partition_size);
            if (end <= start) {
                return start;
            }
            std::uniform_int_distribution<int64_t> node_dist(start, end - 1);
            return node_dist(rng);
        };

        auto flush_batch = [&](double batch_fraction) {
            if (current_state_idx < 0) {
                return;
            }
            if (args.include_local_negatives && !current_state_partitions.empty()) {
                int64_t neg_draws = static_cast<int64_t>(std::llround(
                    static_cast<double>(2 * args.num_chunks * args.negatives_per_positive) * batch_fraction));
                neg_draws = std::max<int64_t>(neg_draws, 0);
                for (int64_t draw = 0; draw < neg_draws; draw++) {
                    int64_t neg_id = sample_local_negative(current_state_partitions);
                    int64_t part = node_partition(neg_id, partition_size, args.num_partitions);
                    batch_touched_per_partition[part].insert(neg_id);
                }
            }
            finalize_batch_updates(batch_touched_per_partition,
                                   segments,
                                   state_to_segment[current_state_idx],
                                   state_partition_batch_touches[current_state_idx]);
        };

        for (std::size_t bucket_idx = 0; bucket_idx < bucket_sizes.size(); bucket_idx++) {
            if (args.max_edges > 0 && processed_raw_edges_total >= args.max_edges) {
                break;
            }
            int64_t bucket_rows = bucket_sizes[bucket_idx];
            int64_t state_idx = bucket_to_state[bucket_idx];
            int64_t processed = 0;
            while (processed < bucket_rows) {
                if (args.max_edges > 0 && processed_raw_edges_total >= args.max_edges) {
                    break;
                }
                int64_t take = std::min<int64_t>(rows_per_chunk, bucket_rows - processed);
                if (args.max_edges > 0) {
                    take = std::min<int64_t>(take, args.max_edges - processed_raw_edges_total);
                }
                int64_t bytes = take * edge_cols * dtype_bytes;
                edge_in.read(raw.data(), bytes);
                if (!edge_in) {
                    throw std::runtime_error("Failed while reading edge file");
                }
                if (state_idx < 0) {
                    processed += take;
                    continue;
                }

                if (current_state_idx != state_idx) {
                    if (current_state_idx >= 0) {
                        if (raw_edges_in_state_batch > 0) {
                            flush_batch(static_cast<double>(raw_edges_in_state_batch) / static_cast<double>(args.batch_size));
                            raw_edges_in_state_batch = 0;
                        }
                    }
                    current_state_idx = state_idx;
                    current_state_partitions = state_partitions[state_idx];
                }

                int64_t chunk_offset = 0;
                while (chunk_offset < take) {
                    int64_t batch_room = args.batch_size - raw_edges_in_state_batch;
                    int64_t span = std::min<int64_t>(batch_room, take - chunk_offset);
                    int64_t global_span_start = processed + chunk_offset;
                    int64_t first_sample = 0;
                    int64_t mod = global_span_start % args.edge_sample_stride;
                    if (mod != 0) {
                        first_sample = args.edge_sample_stride - mod;
                    }
                    for (int64_t local = first_sample; local < span; local += args.edge_sample_stride) {
                        const char* row = raw.data() + (chunk_offset + local) * edge_cols * dtype_bytes;
                        int64_t src = decode_value(row);
                        int64_t dst = decode_value(row + dtype_bytes * (edge_cols - 1));
                        batch_touched_per_partition[node_partition(src, partition_size, args.num_partitions)].insert(src);
                        batch_touched_per_partition[node_partition(dst, partition_size, args.num_partitions)].insert(dst);
                    }
                    raw_edges_in_state_batch += span;
                    chunk_offset += span;
                    processed_raw_edges_total += span;
                    if (raw_edges_in_state_batch == args.batch_size) {
                        flush_batch(1.0);
                        raw_edges_in_state_batch = 0;
                    }
                }
                processed += take;
            }
        }

        if (current_state_idx >= 0 && raw_edges_in_state_batch > 0) {
            flush_batch(static_cast<double>(raw_edges_in_state_batch) / static_cast<double>(args.batch_size));
        }

        int64_t segment_count = 0;
        int64_t total_batches = 0;
        int64_t total_touch_writes = 0;
        int64_t total_unique_writes = 0;
        double mean_segment_duplicate_ratio = 0.0;
        double weighted_duplicate_ratio = 0.0;
        double mean_repeated_touch_fraction = 0.0;
        double mean_segment_states = 0.0;
        double mean_segment_batches = 0.0;
        double mean_run_count = 0.0;
        double mean_mean_run_length = 0.0;
        double mean_max_run_length = 0.0;
        int64_t segments_with_future_revisit = 0;
        std::vector<int64_t> revisit_gaps;
        std::unordered_map<int64_t, double> mean_topk_coverage;
        std::unordered_map<int64_t, double> revisit_within_fraction;
        std::unordered_map<int64_t, double> mean_future_touch_capture;
        std::unordered_map<int64_t, double> weighted_future_touch_capture_numer;
        std::unordered_map<int64_t, int64_t> weighted_future_touch_capture_denom;
        std::unordered_map<int64_t, double> mean_window_current_touch_ratio;
        const std::vector<int64_t> chunk_sizes = {64, 256, 1024, 4096};
        std::unordered_map<int64_t, double> mean_chunk_fill;
        std::unordered_map<int64_t, double> weighted_chunk_fill_numer;
        std::unordered_map<int64_t, double> mean_chunk_fraction;
        for (auto k : args.top_ks) {
            mean_topk_coverage[k] = 0.0;
        }
        for (auto window : args.residency_windows) {
            revisit_within_fraction[window] = 0.0;
            mean_future_touch_capture[window] = 0.0;
            weighted_future_touch_capture_numer[window] = 0.0;
            weighted_future_touch_capture_denom[window] = 0;
            mean_window_current_touch_ratio[window] = 0.0;
        }
        for (auto chunk : chunk_sizes) {
            mean_chunk_fill[chunk] = 0.0;
            weighted_chunk_fill_numer[chunk] = 0.0;
            mean_chunk_fraction[chunk] = 0.0;
        }

        for (const auto& segment : segments) {
            if (segment.batch_count == 0 || segment.total_batch_touches == 0) {
                continue;
            }
            segment_count += 1;
            total_batches += segment.batch_count;
            total_touch_writes += segment.total_batch_touches;
            total_unique_writes += static_cast<int64_t>(segment.entity_batch_touch_counts.size());
            mean_segment_states += static_cast<double>(segment.state_count);
            mean_segment_batches += static_cast<double>(segment.batch_count);
            double duplicate_ratio = 1.0 - static_cast<double>(segment.entity_batch_touch_counts.size()) / static_cast<double>(segment.total_batch_touches);
            mean_segment_duplicate_ratio += duplicate_ratio;
            int64_t repeated_touches = 0;
            for (const auto& [_, count] : segment.entity_batch_touch_counts) {
                if (count > 1) {
                    repeated_touches += count;
                }
            }
            mean_repeated_touch_fraction += static_cast<double>(repeated_touches) / static_cast<double>(segment.total_batch_touches);
            for (auto k : args.top_ks) {
                mean_topk_coverage[k] += top_k_coverage(segment.entity_batch_touch_counts, k);
            }

            auto rows = sorted_local_rows(segment, partition_size);
            auto span_stats = analyze_spans(rows);
            mean_run_count += static_cast<double>(span_stats.run_count);
            mean_mean_run_length += span_stats.mean_run_length;
            mean_max_run_length += static_cast<double>(span_stats.max_run_length);

            const auto& lane_states = lanes[segment.lane];
            int64_t total_future_partition_touches = 0;
            int64_t next_revisit_gap = -1;
            for (int64_t pos = segment.end_lane_pos + 1; pos < static_cast<int64_t>(lane_states.size()); pos++) {
                auto future_state_idx = lane_states[pos];
                const auto& parts = state_partitions[future_state_idx];
                bool present = std::find(parts.begin(), parts.end(), segment.partition_id) != parts.end();
                if (!present) {
                    continue;
                }
                if (next_revisit_gap < 0) {
                    next_revisit_gap = pos - segment.end_lane_pos;
                }
                auto touch_it = state_partition_batch_touches[future_state_idx].find(segment.partition_id);
                if (touch_it != state_partition_batch_touches[future_state_idx].end()) {
                    total_future_partition_touches += touch_it->second;
                }
            }
            if (next_revisit_gap > 0) {
                segments_with_future_revisit += 1;
                revisit_gaps.emplace_back(next_revisit_gap);
            }
            for (auto window : args.residency_windows) {
                if (next_revisit_gap > 0 && next_revisit_gap <= window) {
                    revisit_within_fraction[window] += 1.0;
                }
                int64_t window_partition_touches = 0;
                int64_t upper = std::min<int64_t>(segment.end_lane_pos + window, static_cast<int64_t>(lane_states.size()) - 1);
                for (int64_t pos = segment.end_lane_pos + 1; pos <= upper; pos++) {
                    auto future_state_idx = lane_states[pos];
                    const auto& parts = state_partitions[future_state_idx];
                    bool present = std::find(parts.begin(), parts.end(), segment.partition_id) != parts.end();
                    if (!present) {
                        continue;
                    }
                    auto touch_it = state_partition_batch_touches[future_state_idx].find(segment.partition_id);
                    if (touch_it != state_partition_batch_touches[future_state_idx].end()) {
                        window_partition_touches += touch_it->second;
                    }
                }
                mean_window_current_touch_ratio[window] += segment.total_batch_touches > 0
                    ? static_cast<double>(window_partition_touches) / static_cast<double>(segment.total_batch_touches)
                    : 0.0;
                if (total_future_partition_touches > 0) {
                    mean_future_touch_capture[window] += static_cast<double>(window_partition_touches) /
                                                         static_cast<double>(total_future_partition_touches);
                    weighted_future_touch_capture_numer[window] += static_cast<double>(window_partition_touches);
                    weighted_future_touch_capture_denom[window] += total_future_partition_touches;
                }
            }

            int64_t partition_rows = std::min<int64_t>(partition_size, num_nodes - segment.partition_id * partition_size);
            for (auto chunk : chunk_sizes) {
                double fill = touched_chunk_fill(rows, chunk);
                mean_chunk_fill[chunk] += fill;
                weighted_chunk_fill_numer[chunk] += fill * static_cast<double>(rows.size());
                mean_chunk_fraction[chunk] += touched_chunk_fraction(rows, chunk, partition_rows);
            }
        }

        if (segment_count > 0) {
            mean_segment_duplicate_ratio /= static_cast<double>(segment_count);
            mean_repeated_touch_fraction /= static_cast<double>(segment_count);
            mean_segment_states /= static_cast<double>(segment_count);
            mean_segment_batches /= static_cast<double>(segment_count);
            mean_run_count /= static_cast<double>(segment_count);
            mean_mean_run_length /= static_cast<double>(segment_count);
            mean_max_run_length /= static_cast<double>(segment_count);
            for (auto& [_, value] : mean_topk_coverage) {
                value /= static_cast<double>(segment_count);
            }
            for (auto& [_, value] : revisit_within_fraction) {
                value /= static_cast<double>(segment_count);
            }
            for (auto& [_, value] : mean_future_touch_capture) {
                value /= static_cast<double>(segment_count);
            }
            for (auto& [_, value] : mean_window_current_touch_ratio) {
                value /= static_cast<double>(segment_count);
            }
            for (auto& [_, value] : mean_chunk_fill) {
                value /= static_cast<double>(segment_count);
            }
            for (auto& [_, value] : mean_chunk_fraction) {
                value /= static_cast<double>(segment_count);
            }
        }
        if (total_touch_writes > 0) {
            weighted_duplicate_ratio = 1.0 - static_cast<double>(total_unique_writes) / static_cast<double>(total_touch_writes);
        }

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "dataset_dir=" << args.dataset_dir << "\n";
        std::cout << "ordering_states=" << buffer_states.size() << "\n";
        std::cout << "active_devices=" << args.active_devices << "\n";
        std::cout << "regroup=" << (args.regroup ? 1 : 0) << "\n";
        std::cout << "access_aware=" << (args.access_aware ? 1 : 0) << "\n";
        std::cout << "access_aware_generate=" << (args.access_aware_generate ? 1 : 0) << "\n";
        std::cout << "num_nodes=" << num_nodes << "\n";
        std::cout << "num_edges=" << num_edges << "\n";
        std::cout << "batch_size=" << args.batch_size << "\n";
        std::cout << "num_chunks=" << args.num_chunks << "\n";
        std::cout << "negatives_per_positive=" << args.negatives_per_positive << "\n";
        std::cout << "include_local_negatives=" << (args.include_local_negatives ? 1 : 0) << "\n";
        std::cout << "max_edges=" << args.max_edges << "\n";
        std::cout << "processed_raw_edges=" << processed_raw_edges_total << "\n";
        std::cout << "edge_sample_stride=" << args.edge_sample_stride << "\n";
        std::cout << "segment_count=" << segment_count << "\n";
        std::cout << "total_batches_observed=" << total_batches << "\n";
        std::cout << "total_batch_touch_writes=" << total_touch_writes << "\n";
        std::cout << "total_unique_writes_before_eviction=" << total_unique_writes << "\n";
        std::cout << "weighted_duplicate_ratio=" << weighted_duplicate_ratio << "\n";
        std::cout << "mean_segment_duplicate_ratio=" << mean_segment_duplicate_ratio << "\n";
        std::cout << "mean_repeated_touch_fraction=" << mean_repeated_touch_fraction << "\n";
        std::cout << "mean_segment_states=" << mean_segment_states << "\n";
        std::cout << "mean_segment_batches=" << mean_segment_batches << "\n";
        std::cout << "mean_run_count=" << mean_run_count << "\n";
        std::cout << "mean_mean_run_length=" << mean_mean_run_length << "\n";
        std::cout << "mean_max_run_length=" << mean_max_run_length << "\n";
        std::cout << "segments_with_future_revisit=" << segments_with_future_revisit << "\n";
        if (!revisit_gaps.empty()) {
            auto sorted_gaps = revisit_gaps;
            std::sort(sorted_gaps.begin(), sorted_gaps.end());
            double mean_gap = std::accumulate(sorted_gaps.begin(), sorted_gaps.end(), 0.0) / static_cast<double>(sorted_gaps.size());
            double median_gap = sorted_gaps[sorted_gaps.size() / 2];
            std::cout << "mean_next_revisit_gap=" << mean_gap << "\n";
            std::cout << "median_next_revisit_gap=" << median_gap << "\n";
        } else {
            std::cout << "mean_next_revisit_gap=0.000000\n";
            std::cout << "median_next_revisit_gap=0.000000\n";
        }
        for (auto [k, value] : mean_topk_coverage) {
            std::cout << "mean_segment_top" << k << "_coverage=" << value << "\n";
        }
        for (auto window : args.residency_windows) {
            std::cout << "revisit_within_" << window << "_fraction=" << revisit_within_fraction[window] << "\n";
            std::cout << "mean_future_touch_capture_" << window << "=" << mean_future_touch_capture[window] << "\n";
            std::cout << "weighted_future_touch_capture_" << window << "="
                      << (weighted_future_touch_capture_denom[window] > 0
                              ? weighted_future_touch_capture_numer[window] / static_cast<double>(weighted_future_touch_capture_denom[window])
                              : 0.0)
                      << "\n";
            std::cout << "mean_window_current_touch_ratio_" << window << "=" << mean_window_current_touch_ratio[window] << "\n";
        }
        for (auto chunk : chunk_sizes) {
            std::cout << "mean_chunk_fill_" << chunk << "=" << mean_chunk_fill[chunk] << "\n";
            std::cout << "weighted_chunk_fill_" << chunk << "="
                      << (total_unique_writes > 0 ? weighted_chunk_fill_numer[chunk] / static_cast<double>(total_unique_writes) : 0.0)
                      << "\n";
            std::cout << "mean_chunk_fraction_" << chunk << "=" << mean_chunk_fraction[chunk] << "\n";
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
