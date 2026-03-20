#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "configuration/options.h"
#include "data/ordering.h"

namespace {

void print_usage(const char *prog) {
    std::cerr << "Usage: " << prog
              << " <ordering> <num_partitions> <buffer_capacity> <fine_to_coarse_ratio> <num_cache_partitions>"
                 " <randomly_assign_edge_buckets:0|1> <active_devices> [--verbose] [--seed <int64>] [--regroup]\n";
}

std::vector<int64_t> tensor_to_vector(torch::Tensor tensor) {
    tensor = tensor.to(torch::kCPU).to(torch::kInt64).contiguous();
    auto *data = tensor.data_ptr<int64_t>();
    return std::vector<int64_t>(data, data + tensor.numel());
}

int64_t pair_overlap(std::vector<int64_t> left, std::vector<int64_t> right) {
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
    if (argc < 8) {
        print_usage(argv[0]);
        return 1;
    }

    bool verbose = false;
    bool regroup = false;
    int64_t seed = 12345;
    for (int i = 8; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--verbose") {
            verbose = true;
        } else if (arg == "--seed" && (i + 1) < argc) {
            seed = std::stoll(argv[++i]);
        } else if (arg == "--regroup") {
            regroup = true;
        }
    }

    std::srand(static_cast<unsigned int>(seed));
    torch::manual_seed(seed);

    EdgeBucketOrdering ordering = getEdgeBucketOrderingEnum(argv[1]);
    int num_partitions = std::stoi(argv[2]);
    int buffer_capacity = std::stoi(argv[3]);
    int fine_to_coarse_ratio = std::stoi(argv[4]);
    int num_cache_partitions = std::stoi(argv[5]);
    bool randomly_assign_edge_buckets = std::stoi(argv[6]) != 0;
    int active_devices = std::stoi(argv[7]);

    auto ordering_result = getEdgeBucketOrdering(ordering, num_partitions, buffer_capacity, fine_to_coarse_ratio,
                                                 num_cache_partitions, randomly_assign_edge_buckets);
    auto buffer_states = std::get<0>(ordering_result);
    if (regroup) {
        auto permutation = getDisjointBufferStatePermutation(buffer_states, active_devices);
        std::vector<torch::Tensor> reordered;
        reordered.reserve(buffer_states.size());
        for (auto idx : permutation) {
            reordered.emplace_back(buffer_states[idx]);
        }
        buffer_states = std::move(reordered);
    }

    if (buffer_states.empty()) {
        std::cerr << "No buffer states generated\n";
        return 2;
    }

    int64_t conflicting_supersteps = 0;
    int64_t total_pairwise_overlap = 0;
    int64_t max_pairwise_overlap = 0;
    int64_t total_pairs = 0;
    int64_t total_supersteps = (static_cast<int64_t>(buffer_states.size()) + active_devices - 1) / active_devices;

    for (int64_t superstep = 0; superstep < total_supersteps; superstep++) {
        std::vector<std::vector<int64_t>> active_states;
        for (int device_idx = 0; device_idx < active_devices; device_idx++) {
            int64_t state_idx = superstep * active_devices + device_idx;
            if (state_idx >= static_cast<int64_t>(buffer_states.size())) {
                break;
            }
            active_states.emplace_back(tensor_to_vector(buffer_states[state_idx]));
        }

        int64_t superstep_overlap = 0;
        for (std::size_t i = 0; i < active_states.size(); i++) {
            for (std::size_t j = i + 1; j < active_states.size(); j++) {
                int64_t overlap = pair_overlap(active_states[i], active_states[j]);
                total_pairwise_overlap += overlap;
                max_pairwise_overlap = std::max(max_pairwise_overlap, overlap);
                superstep_overlap += overlap;
                total_pairs++;
            }
        }

        if (superstep_overlap > 0) {
            conflicting_supersteps++;
        }

        if (verbose) {
            std::cout << "superstep " << superstep << ":";
            for (std::size_t i = 0; i < active_states.size(); i++) {
                std::cout << " gpu" << i << "=[";
                for (std::size_t j = 0; j < active_states[i].size(); j++) {
                    if (j > 0) {
                        std::cout << ",";
                    }
                    std::cout << active_states[i][j];
                }
                std::cout << "]";
            }
            std::cout << " pairwise_overlap=" << superstep_overlap << "\n";
        }
    }

    double mean_pairwise_overlap = total_pairs > 0 ? static_cast<double>(total_pairwise_overlap) / static_cast<double>(total_pairs) : 0.0;

    std::cout << "ordering_states=" << buffer_states.size() << "\n";
    std::cout << "active_devices=" << active_devices << "\n";
    std::cout << "buffer_capacity=" << buffer_capacity << "\n";
    std::cout << "total_supersteps=" << total_supersteps << "\n";
    std::cout << "conflicting_supersteps=" << conflicting_supersteps << "\n";
    std::cout << "max_pairwise_overlap=" << max_pairwise_overlap << "\n";
    std::cout << "mean_pairwise_overlap=" << mean_pairwise_overlap << "\n";
    std::cout << "disjoint_active_partitions=" << (conflicting_supersteps == 0 ? "true" : "false") << "\n";

    return 0;
}
