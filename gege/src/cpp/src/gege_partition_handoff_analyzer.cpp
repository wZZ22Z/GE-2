#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "configuration/options.h"
#include "data/ordering.h"

namespace {

struct Args {
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
};

void print_usage(const char *prog) {
    std::cerr << "Usage: " << prog
              << " <ordering> <num_partitions> <buffer_capacity> <fine_to_coarse_ratio> <num_cache_partitions>"
                 " <randomly_assign_edge_buckets:0|1> <active_devices>"
                 " [--regroup] [--access-aware] [--access-aware-generate]\n";
}

Args parse_args(int argc, char **argv) {
    if (argc < 8) {
        print_usage(argv[0]);
        throw std::runtime_error("Not enough arguments");
    }

    Args args;
    args.ordering = getEdgeBucketOrderingEnum(argv[1]);
    args.num_partitions = std::stoi(argv[2]);
    args.buffer_capacity = std::stoi(argv[3]);
    args.fine_to_coarse_ratio = std::stoi(argv[4]);
    args.num_cache_partitions = std::stoi(argv[5]);
    args.randomly_assign_edge_buckets = std::stoi(argv[6]) != 0;
    args.active_devices = std::stoi(argv[7]);

    for (int i = 8; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--regroup") {
            args.regroup = true;
        } else if (arg == "--access-aware") {
            args.access_aware = true;
        } else if (arg == "--access-aware-generate") {
            args.access_aware_generate = true;
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    return args;
}

std::vector<int64_t> tensor_to_vector(torch::Tensor tensor) {
    tensor = tensor.to(torch::kCPU).to(torch::kInt64).contiguous();
    auto *data = tensor.data_ptr<int64_t>();
    return std::vector<int64_t>(data, data + tensor.numel());
}

std::vector<std::vector<int64_t>> to_supersteps(const std::vector<torch::Tensor> &buffer_states, int active_devices) {
    std::vector<std::vector<int64_t>> flat_states;
    flat_states.reserve(buffer_states.size());
    for (auto &state : buffer_states) {
        flat_states.emplace_back(tensor_to_vector(state));
    }

    return flat_states;
}

bool contains_partition(const std::vector<int64_t> &state, int64_t partition_id) {
    return std::find(state.begin(), state.end(), partition_id) != state.end();
}

std::vector<std::vector<std::vector<int64_t>>> group_into_supersteps(const std::vector<std::vector<int64_t>> &states, int active_devices) {
    std::vector<std::vector<std::vector<int64_t>>> supersteps;
    if (active_devices <= 0) {
        return supersteps;
    }
    for (std::size_t i = 0; i < states.size(); i += active_devices) {
        std::vector<std::vector<int64_t>> group;
        for (std::size_t j = i; j < states.size() && j < i + static_cast<std::size_t>(active_devices); j++) {
            group.emplace_back(states[j]);
        }
        supersteps.emplace_back(std::move(group));
    }
    return supersteps;
}

struct HorizonStats {
    int64_t peer_hits = 0;
    int64_t any_hits = 0;
};

}  // namespace

int main(int argc, char **argv) {
    try {
        Args args = parse_args(argc, argv);

        std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> tup;
        if (args.access_aware_generate && args.ordering == EdgeBucketOrdering::CUSTOM) {
            tup = getAccessAwareCustomEdgeBucketOrdering(args.num_partitions, args.buffer_capacity, args.active_devices);
        } else {
            tup = getEdgeBucketOrdering(args.ordering, args.num_partitions, args.buffer_capacity, args.fine_to_coarse_ratio,
                                        args.num_cache_partitions, args.randomly_assign_edge_buckets);
        }

        auto buffer_states = std::get<0>(tup);
        auto edge_buckets_per_buffer = std::get<1>(tup);

        if (!args.access_aware_generate && args.active_devices > 1) {
            std::vector<int64_t> permutation;
            if (args.access_aware) {
                permutation = getAccessAwareDisjointBufferStatePermutation(buffer_states, edge_buckets_per_buffer, args.active_devices);
            } else if (args.regroup) {
                permutation = getDisjointBufferStatePermutation(buffer_states, args.active_devices);
            }

            if (!permutation.empty()) {
                std::vector<torch::Tensor> reordered_states;
                reordered_states.reserve(buffer_states.size());
                for (auto idx : permutation) {
                    reordered_states.emplace_back(buffer_states[idx]);
                }
                buffer_states = std::move(reordered_states);
            }
        }

        auto flat_states = to_supersteps(buffer_states, args.active_devices);
        auto supersteps = group_into_supersteps(flat_states, args.active_devices);

        const std::vector<int> horizons = {1, 2, 4, 8};
        std::vector<HorizonStats> horizon_stats(horizons.size());
        int64_t total_current_states = 0;
        int64_t total_retained = 0;
        int64_t total_evicted = 0;

        for (std::size_t step = 0; step + 1 < supersteps.size(); step++) {
            const auto &current_group = supersteps[step];
            const auto &next_group = supersteps[step + 1];
            for (std::size_t lane = 0; lane < current_group.size(); lane++) {
                const auto &current_state = current_group[lane];
                std::unordered_set<int64_t> next_same_lane;
                if (lane < next_group.size()) {
                    next_same_lane.insert(next_group[lane].begin(), next_group[lane].end());
                }

                total_current_states++;
                for (auto partition_id : current_state) {
                    if (next_same_lane.find(partition_id) != next_same_lane.end()) {
                        total_retained++;
                        continue;
                    }

                    total_evicted++;
                    for (std::size_t h = 0; h < horizons.size(); h++) {
                        bool peer_hit = false;
                        bool any_hit = false;
                        for (int delta = 1; delta <= horizons[h] && step + delta < supersteps.size(); delta++) {
                            const auto &future_group = supersteps[step + delta];
                            for (std::size_t future_lane = 0; future_lane < future_group.size(); future_lane++) {
                                if (!contains_partition(future_group[future_lane], partition_id)) {
                                    continue;
                                }
                                any_hit = true;
                                if (future_lane != lane) {
                                    peer_hit = true;
                                }
                            }
                            if (peer_hit) {
                                break;
                            }
                        }
                        horizon_stats[h].peer_hits += peer_hit ? 1 : 0;
                        horizon_stats[h].any_hits += any_hit ? 1 : 0;
                    }
                }
            }
        }

        std::cout << "ordering_states=" << buffer_states.size() << "\n";
        std::cout << "supersteps=" << supersteps.size() << "\n";
        std::cout << "active_devices=" << args.active_devices << "\n";
        std::cout << "total_lane_states=" << total_current_states << "\n";
        std::cout << "retained_partitions_next=" << total_retained << "\n";
        std::cout << "evicted_partitions_next=" << total_evicted << "\n";
        std::cout << "retained_fraction_next="
                  << (total_retained + total_evicted > 0 ? static_cast<double>(total_retained) / static_cast<double>(total_retained + total_evicted) : 0.0)
                  << "\n";

        for (std::size_t h = 0; h < horizons.size(); h++) {
            double peer_fraction =
                total_evicted > 0 ? static_cast<double>(horizon_stats[h].peer_hits) / static_cast<double>(total_evicted) : 0.0;
            double any_fraction =
                total_evicted > 0 ? static_cast<double>(horizon_stats[h].any_hits) / static_cast<double>(total_evicted) : 0.0;
            std::cout << "peer_relay_h" << horizons[h] << "_fraction=" << peer_fraction << "\n";
            std::cout << "future_reuse_h" << horizons[h] << "_fraction=" << any_fraction << "\n";
        }

        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
