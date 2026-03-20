#include <algorithm>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "configuration/options.h"
#include "data/ordering.h"

namespace {

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
    bool verbose = false;
    int64_t seed = 12345;
    int64_t sketch_k = 256;
    int64_t edge_sample_stride = 1;
};

void print_usage(const char *prog) {
    std::cerr << "Usage: " << prog
              << " <dataset_dir> <ordering> <num_partitions> <buffer_capacity> <fine_to_coarse_ratio> <num_cache_partitions>"
                 " <randomly_assign_edge_buckets:0|1> <active_devices>"
                 " [--regroup] [--access-aware] [--access-aware-generate] [--verbose] [--seed <int64>] [--sketch-k <int64>] [--edge-sample-stride <int64>]\n";
}

uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

std::vector<int64_t> tensor_to_vector(torch::Tensor tensor) {
    tensor = tensor.to(torch::kCPU).to(torch::kInt64).contiguous();
    auto *data = tensor.data_ptr<int64_t>();
    return std::vector<int64_t>(data, data + tensor.numel());
}

struct KMVSketch {
    explicit KMVSketch(int64_t k = 256) : k_(std::max<int64_t>(k, 1)) {}

    void insert(int64_t value) {
        uint64_t h = splitmix64(static_cast<uint64_t>(value));
        auto it = std::lower_bound(values_.begin(), values_.end(), h);
        if (it != values_.end() && *it == h) {
            return;
        }
        if (static_cast<int64_t>(values_.size()) < k_) {
            values_.insert(it, h);
            return;
        }
        if (h >= values_.back()) {
            return;
        }
        values_.insert(it, h);
        values_.pop_back();
    }

    void merge(const KMVSketch &other) {
        std::vector<uint64_t> merged;
        merged.reserve(std::min<int64_t>(k_, values_.size() + other.values_.size()));
        std::size_t i = 0;
        std::size_t j = 0;
        while (merged.size() < static_cast<std::size_t>(k_) && (i < values_.size() || j < other.values_.size())) {
            uint64_t next = 0;
            if (j >= other.values_.size() || (i < values_.size() && values_[i] < other.values_[j])) {
                next = values_[i++];
            } else if (i >= values_.size() || other.values_[j] < values_[i]) {
                next = other.values_[j++];
            } else {
                next = values_[i];
                i++;
                j++;
            }
            if (merged.empty() || merged.back() != next) {
                merged.emplace_back(next);
            }
        }
        values_.swap(merged);
    }

    double estimate_cardinality() const {
        if (values_.empty()) {
            return 0.0;
        }
        if (static_cast<int64_t>(values_.size()) < k_) {
            return static_cast<double>(values_.size());
        }
        constexpr long double denom = static_cast<long double>(std::numeric_limits<uint64_t>::max());
        long double kth = static_cast<long double>(values_.back()) / denom;
        if (kth <= 0.0L) {
            return static_cast<double>(k_);
        }
        return static_cast<double>((static_cast<long double>(k_ - 1) / kth));
    }

    double jaccard(const KMVSketch &other) const {
        if (values_.empty() && other.values_.empty()) {
            return 1.0;
        }
        std::size_t i = 0;
        std::size_t j = 0;
        int64_t overlap = 0;
        while (i < values_.size() && j < other.values_.size()) {
            if (values_[i] == other.values_[j]) {
                overlap++;
                i++;
                j++;
            } else if (values_[i] < other.values_[j]) {
                i++;
            } else {
                j++;
            }
        }
        int64_t denom = std::min<int64_t>(values_.size(), other.values_.size());
        return denom > 0 ? static_cast<double>(overlap) / static_cast<double>(denom) : 0.0;
    }

    int64_t size() const { return static_cast<int64_t>(values_.size()); }

   private:
    int64_t k_;
    std::vector<uint64_t> values_;
};

std::vector<int64_t> read_offsets(const std::string &path) {
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

int infer_edge_columns(const std::string &path, int64_t num_edges) {
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
    if (bytes_per_edge == 8) return 2;
    if (bytes_per_edge == 12) return 3;
    if (bytes_per_edge == 16) return 2;
    if (bytes_per_edge == 24) return 3;
    throw std::runtime_error("Unsupported bytes per edge: " + std::to_string(bytes_per_edge));
}

int infer_edge_dtype_bytes(const std::string &path, int64_t num_edges, int edge_cols) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    int64_t bytes = static_cast<int64_t>(in.tellg());
    int64_t bytes_per_value = bytes / (num_edges * edge_cols);
    if (bytes_per_value != 4 && bytes_per_value != 8) {
        throw std::runtime_error("Unsupported edge value width: " + std::to_string(bytes_per_value));
    }
    return static_cast<int>(bytes_per_value);
}

Args parse_args(int argc, char **argv) {
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
        } else if (arg == "--verbose") {
            args.verbose = true;
        } else if (arg == "--seed" && (i + 1) < argc) {
            args.seed = std::stoll(argv[++i]);
        } else if (arg == "--sketch-k" && (i + 1) < argc) {
            args.sketch_k = std::stoll(argv[++i]);
        } else if (arg == "--edge-sample-stride" && (i + 1) < argc) {
            args.edge_sample_stride = std::max<int64_t>(1, std::stoll(argv[++i]));
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    return args;
}

struct StateMetrics {
    KMVSketch positive_nodes;
    std::vector<int64_t> partitions;
    int64_t assigned_buckets = 0;

    explicit StateMetrics(int64_t sketch_k = 256) : positive_nodes(sketch_k) {}
};

int64_t partition_overlap(std::vector<int64_t> left, std::vector<int64_t> right) {
    std::sort(left.begin(), left.end());
    std::sort(right.begin(), right.end());
    int64_t overlap = 0;
    std::size_t i = 0;
    std::size_t j = 0;
    while (i < left.size() && j < right.size()) {
        if (left[i] == right[j]) {
            overlap++;
            i++;
            j++;
        } else if (left[i] < right[j]) {
            i++;
        } else {
            j++;
        }
    }
    return overlap;
}

}  // namespace

int main(int argc, char **argv) {
    try {
        Args args = parse_args(argc, argv);
        std::srand(static_cast<unsigned int>(args.seed));
        torch::manual_seed(args.seed);

        std::string edge_path = args.dataset_dir + "/edges/train_edges.bin";
        std::string offsets_path = args.dataset_dir + "/edges/train_partition_offsets.txt";
        auto bucket_sizes = read_offsets(offsets_path);
        if (bucket_sizes.size() != static_cast<std::size_t>(args.num_partitions * args.num_partitions)) {
            throw std::runtime_error("Unexpected number of edge buckets in offsets file");
        }
        int64_t num_edges = std::accumulate(bucket_sizes.begin(), bucket_sizes.end(), static_cast<int64_t>(0));
        int edge_cols = infer_edge_columns(edge_path, num_edges);
        int dtype_bytes = infer_edge_dtype_bytes(edge_path, num_edges, edge_cols);

        auto ordering_result = args.access_aware_generate
            ? getAccessAwareCustomEdgeBucketOrdering(args.num_partitions, args.buffer_capacity, args.active_devices)
            : getEdgeBucketOrdering(args.ordering, args.num_partitions, args.buffer_capacity, args.fine_to_coarse_ratio,
                                    args.num_cache_partitions, args.randomly_assign_edge_buckets);
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

        if (buffer_states.empty() || edge_buckets_per_buffer.size() != buffer_states.size()) {
            throw std::runtime_error("Invalid buffer ordering");
        }

        std::vector<int64_t> bucket_to_state(args.num_partitions * args.num_partitions, -1);
        std::vector<StateMetrics> states;
        states.reserve(buffer_states.size());
        for (std::size_t state_idx = 0; state_idx < buffer_states.size(); state_idx++) {
            states.emplace_back(args.sketch_k);
            states.back().partitions = tensor_to_vector(buffer_states[state_idx]);
            auto buckets = edge_buckets_per_buffer[state_idx].to(torch::kCPU).to(torch::kInt64).contiguous();
            auto acc = buckets.accessor<int64_t, 2>();
            for (int64_t i = 0; i < buckets.size(0); i++) {
                int64_t src_part = acc[i][0];
                int64_t dst_part = acc[i][1];
                int64_t bucket_idx = src_part * args.num_partitions + dst_part;
                if (bucket_idx < 0 || bucket_idx >= static_cast<int64_t>(bucket_to_state.size())) {
                    throw std::runtime_error("Bucket index out of range");
                }
                if (bucket_to_state[bucket_idx] != -1) {
                    throw std::runtime_error("Edge bucket assigned to multiple states");
                }
                bucket_to_state[bucket_idx] = static_cast<int64_t>(state_idx);
                states.back().assigned_buckets++;
            }
        }

        std::ifstream edge_in(edge_path, std::ios::binary);
        if (!edge_in) {
            throw std::runtime_error("Failed to open edge file");
        }
        constexpr int64_t rows_per_chunk = 1 << 16;
        std::vector<char> raw;
        raw.resize(rows_per_chunk * edge_cols * dtype_bytes);

        auto decode_value = [&](const char *ptr) -> int64_t {
            if (dtype_bytes == 4) {
                int32_t value = 0;
                std::memcpy(&value, ptr, sizeof(int32_t));
                return static_cast<int64_t>(value);
            }
            int64_t value = 0;
            std::memcpy(&value, ptr, sizeof(int64_t));
            return value;
        };

        int64_t edge_offset = 0;
        for (std::size_t bucket_idx = 0; bucket_idx < bucket_sizes.size(); bucket_idx++) {
            int64_t state_idx = bucket_to_state[bucket_idx];
            int64_t bucket_rows = bucket_sizes[bucket_idx];
            int64_t bucket_bytes = bucket_rows * edge_cols * dtype_bytes;
            if (state_idx < 0) {
                edge_in.seekg(bucket_bytes, std::ios::cur);
                edge_offset += bucket_rows;
                continue;
            }

            int64_t processed = 0;
            while (processed < bucket_rows) {
                int64_t take = std::min<int64_t>(rows_per_chunk, bucket_rows - processed);
                int64_t bytes = take * edge_cols * dtype_bytes;
                edge_in.read(raw.data(), bytes);
                if (!edge_in) {
                    throw std::runtime_error("Failed while reading edge file");
                }
                for (int64_t local = 0; local < take; local += args.edge_sample_stride) {
                    const char *row = raw.data() + local * edge_cols * dtype_bytes;
                    int64_t src = decode_value(row);
                    int64_t dst = decode_value(row + dtype_bytes * (edge_cols - 1));
                    states[state_idx].positive_nodes.insert(src);
                    states[state_idx].positive_nodes.insert(dst);
                }
                processed += take;
            }
            edge_offset += bucket_rows;
        }

        int64_t total_supersteps = (static_cast<int64_t>(states.size()) + args.active_devices - 1) / args.active_devices;
        double mean_state_cardinality = 0.0;
        int64_t total_states_used = 0;
        for (auto &state : states) {
            if (state.assigned_buckets > 0) {
                mean_state_cardinality += state.positive_nodes.estimate_cardinality();
                total_states_used++;
            }
        }
        mean_state_cardinality = total_states_used > 0 ? mean_state_cardinality / static_cast<double>(total_states_used) : 0.0;

        double total_lane_partition_overlap = 0.0;
        double total_lane_partition_fraction = 0.0;
        double total_lane_node_jaccard = 0.0;
        double max_lane_node_jaccard = 0.0;
        int64_t lane_pairs = 0;

        for (int lane = 0; lane < args.active_devices; lane++) {
            std::vector<int64_t> lane_states;
            for (int64_t superstep = 0; superstep < total_supersteps; superstep++) {
                int64_t idx = superstep * args.active_devices + lane;
                if (idx < static_cast<int64_t>(states.size())) {
                    lane_states.emplace_back(idx);
                }
            }
            for (std::size_t i = 1; i < lane_states.size(); i++) {
                auto &prev = states[lane_states[i - 1]];
                auto &curr = states[lane_states[i]];
                int64_t p_overlap = partition_overlap(prev.partitions, curr.partitions);
                total_lane_partition_overlap += static_cast<double>(p_overlap);
                total_lane_partition_fraction += static_cast<double>(p_overlap) / static_cast<double>(args.buffer_capacity);
                double node_jaccard = prev.positive_nodes.jaccard(curr.positive_nodes);
                total_lane_node_jaccard += node_jaccard;
                max_lane_node_jaccard = std::max(max_lane_node_jaccard, node_jaccard);
                lane_pairs++;
                if (args.verbose) {
                    std::cout << "lane=" << lane << " step_pair=" << (i - 1) << "->" << i
                              << " partition_overlap=" << p_overlap
                              << " partition_fraction=" << static_cast<double>(p_overlap) / static_cast<double>(args.buffer_capacity)
                              << " node_jaccard_est=" << node_jaccard << "\n";
                }
            }
        }

        double mean_lane_partition_overlap = lane_pairs > 0 ? total_lane_partition_overlap / static_cast<double>(lane_pairs) : 0.0;
        double mean_lane_partition_fraction = lane_pairs > 0 ? total_lane_partition_fraction / static_cast<double>(lane_pairs) : 0.0;
        double mean_lane_node_jaccard = lane_pairs > 0 ? total_lane_node_jaccard / static_cast<double>(lane_pairs) : 0.0;

        std::cout << "dataset_dir=" << args.dataset_dir << "\n";
        std::cout << "ordering_states=" << states.size() << "\n";
        std::cout << "active_devices=" << args.active_devices << "\n";
        std::cout << "regroup=" << (args.regroup ? 1 : 0) << "\n";
        std::cout << "access_aware=" << (args.access_aware ? 1 : 0) << "\n";
        std::cout << "access_aware_generate=" << (args.access_aware_generate ? 1 : 0) << "\n";
        std::cout << "buffer_capacity=" << args.buffer_capacity << "\n";
        std::cout << "num_partitions=" << args.num_partitions << "\n";
        std::cout << "edge_cols=" << edge_cols << "\n";
        std::cout << "edge_dtype_bytes=" << dtype_bytes << "\n";
        std::cout << "num_edges=" << num_edges << "\n";
        std::cout << "sketch_k=" << args.sketch_k << "\n";
        std::cout << "edge_sample_stride=" << args.edge_sample_stride << "\n";
        std::cout << "mean_state_positive_unique_est=" << std::fixed << std::setprecision(3) << mean_state_cardinality << "\n";
        std::cout << "lane_pair_count=" << lane_pairs << "\n";
        std::cout << "mean_lane_partition_overlap=" << mean_lane_partition_overlap << "\n";
        std::cout << "mean_lane_partition_overlap_fraction=" << mean_lane_partition_fraction << "\n";
        std::cout << "mean_lane_positive_node_jaccard_est=" << mean_lane_node_jaccard << "\n";
        std::cout << "max_lane_positive_node_jaccard_est=" << max_lane_node_jaccard << "\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
