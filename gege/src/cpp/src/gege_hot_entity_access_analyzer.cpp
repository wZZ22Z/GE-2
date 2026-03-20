#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
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
    int64_t edge_sample_stride = 1;
    std::vector<int64_t> cache_sizes = {256, 1024, 4096, 16384};
    std::vector<int64_t> history_lengths = {1, 4};
    std::vector<double> top_fracs = {0.001, 0.01, 0.05, 0.10};
};

struct StateAccessStats {
    CountMap counts;
    int64_t total_accesses = 0;
};

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " <dataset_dir> <ordering> <num_partitions> <buffer_capacity> <fine_to_coarse_ratio> <num_cache_partitions>"
                 " <randomly_assign_edge_buckets:0|1> <active_devices>"
                 " [--regroup] [--access-aware] [--access-aware-generate]"
                 " [--seed <int64>] [--edge-sample-stride <int64>]"
                 " [--cache-sizes <csv>] [--history-lengths <csv>]\n";
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
        } else if (arg == "--cache-sizes" && i + 1 < argc) {
            args.cache_sizes = parse_csv_int64(argv[++i]);
        } else if (arg == "--history-lengths" && i + 1 < argc) {
            args.history_lengths = parse_csv_int64(argv[++i]);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    args.cache_sizes.erase(std::remove_if(args.cache_sizes.begin(), args.cache_sizes.end(),
                                          [](int64_t value) { return value <= 0; }),
                           args.cache_sizes.end());
    args.history_lengths.erase(std::remove_if(args.history_lengths.begin(), args.history_lengths.end(),
                                              [](int64_t value) { return value <= 0; }),
                               args.history_lengths.end());
    if (args.cache_sizes.empty()) {
        args.cache_sizes = {256, 1024, 4096, 16384};
    }
    if (args.history_lengths.empty()) {
        args.history_lengths = {1, 4};
    }
    std::sort(args.cache_sizes.begin(), args.cache_sizes.end());
    args.cache_sizes.erase(std::unique(args.cache_sizes.begin(), args.cache_sizes.end()), args.cache_sizes.end());
    std::sort(args.history_lengths.begin(), args.history_lengths.end());
    args.history_lengths.erase(std::unique(args.history_lengths.begin(), args.history_lengths.end()), args.history_lengths.end());

    return args;
}

CountMap merge_counts(const std::vector<StateAccessStats>& states, const std::vector<int64_t>& indices) {
    CountMap merged;
    for (auto idx : indices) {
        const auto& state = states.at(idx);
        for (const auto& [entity, count] : state.counts) {
            merged[entity] += count;
        }
    }
    return merged;
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

double coverage_from_top_k(const CountMap& source, const CountMap& target, int64_t k) {
    if (target.empty() || k <= 0) {
        return 0.0;
    }
    auto ordered = sort_counts_desc(source);
    if (ordered.empty()) {
        return 0.0;
    }
    std::unordered_set<int64_t> top_entities;
    top_entities.reserve(static_cast<std::size_t>(std::min<int64_t>(k, ordered.size())));
    for (int64_t i = 0; i < k && i < static_cast<int64_t>(ordered.size()); i++) {
        top_entities.insert(ordered[i].first);
    }
    int64_t covered = 0;
    int64_t total = 0;
    for (const auto& [entity, count] : target) {
        total += count;
        if (top_entities.find(entity) != top_entities.end()) {
            covered += count;
        }
    }
    return total > 0 ? static_cast<double>(covered) / static_cast<double>(total) : 0.0;
}

double coverage_from_any_seen(const CountMap& source, const CountMap& target) {
    if (target.empty() || source.empty()) {
        return 0.0;
    }
    int64_t covered = 0;
    int64_t total = 0;
    for (const auto& [entity, count] : target) {
        total += count;
        if (source.find(entity) != source.end()) {
            covered += count;
        }
    }
    return total > 0 ? static_cast<double>(covered) / static_cast<double>(total) : 0.0;
}

double fractional_hot_coverage(const CountMap& counts, double frac) {
    if (counts.empty()) {
        return 0.0;
    }
    auto ordered = sort_counts_desc(counts);
    int64_t total = 0;
    for (const auto& [_, count] : ordered) {
        total += count;
    }
    int64_t keep = std::max<int64_t>(1, static_cast<int64_t>(std::ceil(frac * static_cast<double>(ordered.size()))));
    keep = std::min<int64_t>(keep, static_cast<int64_t>(ordered.size()));
    int64_t covered = 0;
    for (int64_t i = 0; i < keep; i++) {
        covered += ordered[i].second;
    }
    return total > 0 ? static_cast<double>(covered) / static_cast<double>(total) : 0.0;
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

std::string join_ints(const std::vector<int64_t>& values) {
    std::string result;
    for (std::size_t i = 0; i < values.size(); i++) {
        if (i > 0) {
            result += ",";
        }
        result += std::to_string(values[i]);
    }
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        std::srand(static_cast<unsigned int>(args.seed));
        torch::manual_seed(args.seed);

        const std::string edge_path = args.dataset_dir + "/edges/train_edges.bin";
        const std::string offsets_path = args.dataset_dir + "/edges/train_partition_offsets.txt";
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
        std::vector<StateAccessStats> states(buffer_states.size());

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

        for (std::size_t bucket_idx = 0; bucket_idx < bucket_sizes.size(); bucket_idx++) {
            const int64_t state_idx = bucket_to_state[bucket_idx];
            const int64_t bucket_rows = bucket_sizes[bucket_idx];
            if (state_idx < 0) {
                edge_in.seekg(bucket_rows * edge_cols * dtype_bytes, std::ios::cur);
                continue;
            }

            int64_t processed = 0;
            auto& state = states[state_idx];
            while (processed < bucket_rows) {
                const int64_t take = std::min<int64_t>(rows_per_chunk, bucket_rows - processed);
                const int64_t bytes = take * edge_cols * dtype_bytes;
                edge_in.read(raw.data(), bytes);
                if (!edge_in) {
                    throw std::runtime_error("Failed while reading edge file");
                }
                for (int64_t local = 0; local < take; local += args.edge_sample_stride) {
                    const char* row = raw.data() + local * edge_cols * dtype_bytes;
                    const int64_t src = decode_value(row);
                    const int64_t dst = decode_value(row + dtype_bytes * (edge_cols - 1));
                    state.counts[src] += 1;
                    state.counts[dst] += 1;
                    state.total_accesses += 2;
                }
                processed += take;
            }
        }

        CountMap global_counts;
        int64_t total_sampled_accesses = 0;
        double mean_state_unique = 0.0;
        double mean_state_accesses = 0.0;
        std::unordered_map<std::string, double> mean_state_frac_coverage;
        for (double frac : args.top_fracs) {
            std::ostringstream key;
            key << std::fixed << std::setprecision(3) << frac;
            mean_state_frac_coverage[key.str()] = 0.0;
        }

        for (const auto& state : states) {
            total_sampled_accesses += state.total_accesses;
            mean_state_unique += static_cast<double>(state.counts.size());
            mean_state_accesses += static_cast<double>(state.total_accesses);
            for (const auto& [entity, count] : state.counts) {
                global_counts[entity] += count;
            }
            for (double frac : args.top_fracs) {
                std::ostringstream key;
                key << std::fixed << std::setprecision(3) << frac;
                mean_state_frac_coverage[key.str()] += fractional_hot_coverage(state.counts, frac);
            }
        }
        if (!states.empty()) {
            mean_state_unique /= static_cast<double>(states.size());
            mean_state_accesses /= static_cast<double>(states.size());
            for (auto& [_, value] : mean_state_frac_coverage) {
                value /= static_cast<double>(states.size());
            }
        }

        std::unordered_map<std::string, double> global_frac_coverage;
        for (double frac : args.top_fracs) {
            std::ostringstream key;
            key << std::fixed << std::setprecision(3) << frac;
            global_frac_coverage[key.str()] = fractional_hot_coverage(global_counts, frac);
        }

        auto lanes = build_lane_states(states.size(), args.active_devices);
        int64_t lane_pairs = 0;
        std::unordered_map<std::string, double> cache_hit_sums;
        for (auto history : args.history_lengths) {
            cache_hit_sums["history" + std::to_string(history) + "_any"] = 0.0;
            for (auto cache_size : args.cache_sizes) {
                cache_hit_sums["history" + std::to_string(history) + "_top" + std::to_string(cache_size)] = 0.0;
            }
        }

        for (const auto& lane : lanes) {
            for (std::size_t pos = 1; pos < lane.size(); pos++) {
                lane_pairs++;
                for (auto history : args.history_lengths) {
                    const auto begin_pos = pos > static_cast<std::size_t>(history) ? pos - static_cast<std::size_t>(history) : 0;
                    std::vector<int64_t> history_indices(lane.begin() + begin_pos, lane.begin() + pos);
                    auto history_counts = merge_counts(states, history_indices);
                    const auto& current = states[lane[pos]].counts;
                    cache_hit_sums["history" + std::to_string(history) + "_any"] += coverage_from_any_seen(history_counts, current);
                    for (auto cache_size : args.cache_sizes) {
                        cache_hit_sums["history" + std::to_string(history) + "_top" + std::to_string(cache_size)] +=
                            coverage_from_top_k(history_counts, current, cache_size);
                    }
                }
            }
        }

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "dataset_dir=" << args.dataset_dir << "\n";
        std::cout << "sample_type=positive_endpoints\n";
        std::cout << "ordering_states=" << states.size() << "\n";
        std::cout << "active_devices=" << args.active_devices << "\n";
        std::cout << "regroup=" << (args.regroup ? 1 : 0) << "\n";
        std::cout << "access_aware=" << (args.access_aware ? 1 : 0) << "\n";
        std::cout << "access_aware_generate=" << (args.access_aware_generate ? 1 : 0) << "\n";
        std::cout << "buffer_capacity=" << args.buffer_capacity << "\n";
        std::cout << "num_partitions=" << args.num_partitions << "\n";
        std::cout << "num_edges=" << num_edges << "\n";
        std::cout << "edge_cols=" << edge_cols << "\n";
        std::cout << "edge_dtype_bytes=" << dtype_bytes << "\n";
        std::cout << "edge_sample_stride=" << args.edge_sample_stride << "\n";
        std::cout << "cache_sizes=" << join_ints(args.cache_sizes) << "\n";
        std::cout << "history_lengths=" << join_ints(args.history_lengths) << "\n";
        std::cout << "total_sampled_accesses=" << total_sampled_accesses << "\n";
        std::cout << "global_unique_entities=" << global_counts.size() << "\n";
        std::cout << "mean_state_unique_entities=" << mean_state_unique << "\n";
        std::cout << "mean_state_sampled_accesses=" << mean_state_accesses << "\n";
        for (const auto& [label, value] : global_frac_coverage) {
            std::cout << "global_top_frac_" << label << "_coverage=" << value << "\n";
        }
        for (const auto& [label, value] : mean_state_frac_coverage) {
            std::cout << "mean_state_top_frac_" << label << "_coverage=" << value << "\n";
        }
        std::cout << "lane_pair_count=" << lane_pairs << "\n";
        for (auto& [label, sum] : cache_hit_sums) {
            std::cout << "mean_" << label << "_coverage=" << (lane_pairs > 0 ? sum / static_cast<double>(lane_pairs) : 0.0) << "\n";
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
