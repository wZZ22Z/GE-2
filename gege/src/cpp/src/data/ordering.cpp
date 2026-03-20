#include "common/datatypes.h"
#include "data/ordering.h"
#include "reporting/logger.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>

#ifdef GEGE_OMP
#include "omp.h"
#endif

namespace {

std::tuple<torch::Tensor, torch::Tensor> unique_with_counts_sorted(torch::Tensor values) {
    auto sort_tup = torch::sort(values.to(torch::kInt64), 0, false);
    torch::Tensor sorted_values = std::get<0>(sort_tup);
    auto unique_tup = torch::unique_consecutive(sorted_values, false, false, true);
    return std::forward_as_tuple(std::get<0>(unique_tup), std::get<2>(unique_tup));
}

struct StateAccessSummary {
    std::vector<int64_t> partitions;
    std::unordered_map<int64_t, int64_t> incident_bucket_counts;
};

std::vector<int64_t> tensor_to_partitions(torch::Tensor tensor) {
    tensor = tensor.to(torch::kCPU).to(torch::kInt64).contiguous();
    auto *data = tensor.data_ptr<int64_t>();
    return std::vector<int64_t>(data, data + tensor.numel());
}

bool optimized_custom_schedule_enabled() {
    const char *raw = std::getenv("GEGE_OPTIMIZED_CUSTOM_SCHEDULE");
    if (raw == nullptr) {
        return true;
    }

    std::string value(raw);
    return !(value == "0" || value == "false" || value == "False" || value == "FALSE");
}

int64_t optimized_custom_schedule_restarts() {
    const char *raw = std::getenv("GEGE_CUSTOM_OPTIMIZER_RESTARTS");
    if (raw == nullptr) {
        return 8;
    }

    try {
        return std::max<int64_t>(std::stoll(std::string(raw)), 1);
    } catch (...) {
        return 8;
    }
}

int64_t optimized_custom_schedule_seed() {
    const char *raw = std::getenv("GEGE_CUSTOM_OPTIMIZER_SEED");
    if (raw == nullptr) {
        return 12345;
    }

    try {
        return std::stoll(std::string(raw));
    } catch (...) {
        return 12345;
    }
}

struct CustomStateMetrics {
    std::vector<int> partitions;
    int64_t weight = 0;
    int64_t batches = 0;
    int64_t bucket_count = 0;
};

struct CustomScheduleScore {
    int64_t worst_round_spread = 0;
    int64_t worst_batch_spread = 0;
    int64_t worst_state_weight = 0;
    int64_t total_round_spread = 0;
    int64_t continuity_hotness = 0;
    int64_t continuity_new_partitions = 0;
    int64_t total_abs_deviation = 0;

    auto as_tuple() const {
        return std::make_tuple(worst_round_spread, worst_batch_spread, worst_state_weight, total_round_spread, continuity_hotness,
                               continuity_new_partitions, total_abs_deviation);
    }
};

struct CustomEvaluatedSchedule {
    std::vector<int> slot_to_partition;
    std::vector<CustomStateMetrics> states;
    std::vector<std::vector<int>> rounds;
    std::vector<std::vector<int>> lane_rounds;
    CustomScheduleScore score;
};

int64_t compute_state_resident_weight(const std::vector<int> &partitions,
                                      const std::vector<int64_t> &edge_bucket_sizes,
                                      int num_partitions) {
    int64_t resident_weight = 0;
    for (auto src_part : partitions) {
        for (auto dst_part : partitions) {
            resident_weight += edge_bucket_sizes[src_part * num_partitions + dst_part];
        }
    }
    return resident_weight;
}

int select_startup_round(const std::vector<std::vector<int>> &lane_rounds,
                         const std::vector<int64_t> &resident_state_weights,
                         const std::vector<CustomStateMetrics> &state_metrics) {
    int best_round = 0;
    auto best_key = std::make_tuple(std::numeric_limits<int64_t>::max(),
                                    std::numeric_limits<int64_t>::max(),
                                    std::numeric_limits<int64_t>::max(),
                                    std::numeric_limits<int64_t>::max(),
                                    std::numeric_limits<int64_t>::max());

    for (int round_idx = 0; round_idx < static_cast<int>(lane_rounds.size()); round_idx++) {
        int64_t max_resident_weight = 0;
        int64_t total_resident_weight = 0;
        int64_t max_assigned_weight = 0;
        int64_t total_assigned_weight = 0;

        for (auto state_idx : lane_rounds[round_idx]) {
            max_resident_weight = std::max<int64_t>(max_resident_weight, resident_state_weights[state_idx]);
            total_resident_weight += resident_state_weights[state_idx];
            max_assigned_weight = std::max<int64_t>(max_assigned_weight, state_metrics[state_idx].weight);
            total_assigned_weight += state_metrics[state_idx].weight;
        }

        auto key = std::make_tuple(max_resident_weight, total_resident_weight, max_assigned_weight, total_assigned_weight, round_idx);
        if (key < best_key) {
            best_key = key;
            best_round = round_idx;
        }
    }

    return best_round;
}

bool custom_score_better(const CustomScheduleScore &lhs, const CustomScheduleScore &rhs) {
    return lhs.as_tuple() < rhs.as_tuple();
}

int int_pow_local(int a, int x) {
    int ans = 1;
    int temp = a;
    while (x) {
        if (x & 1) {
            ans *= temp;
        }
        temp *= temp;
        x >>= 1;
    }
    return ans;
}

std::vector<std::vector<int>> build_custom_template_states(int num_partitions, int buffer_capacity) {
    assert(buffer_capacity == 4);
    int32_t sub_chunk_per_perm = num_partitions / buffer_capacity;
    int32_t log2l = 0;

    while (int_pow_local(2, log2l) < num_partitions) {
        log2l += 1;
    }

    assert(int_pow_local(2, log2l) == num_partitions);

    std::vector<std::vector<std::vector<int>>> offset_supergroup = {
        {{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}},
        {{0, 1, 2, 3}, {1, 0, 3, 2}, {2, 3, 0, 1}, {3, 2, 1, 0}},
        {{0, 2, 3, 1}, {1, 3, 2, 0}, {2, 0, 1, 3}, {3, 1, 0, 2}},
        {{0, 3, 1, 2}, {1, 2, 0, 3}, {2, 1, 3, 0}, {3, 0, 2, 1}},
    };
    std::vector<std::vector<std::vector<int>>> p = {{{0, 1, 2, 3}}};

    for (int log4l_pre = 1; log4l_pre < log2l / 2; log4l_pre++) {
        auto p_pre = p;
        p = std::vector<std::vector<std::vector<int>>>();
        for (auto &s : p_pre) {
            std::vector<std::vector<int>> s_cur;
            for (int offset = 0; offset < int_pow_local(4, log4l_pre + 1); offset += int_pow_local(4, log4l_pre)) {
                for (auto &g : s) {
                    std::vector<int> g_cur;
                    for (auto &x : g) {
                        g_cur.emplace_back(x + offset);
                    }
                    s_cur.emplace_back(g_cur);
                }
            }
            p.emplace_back(s_cur);
        }
        int32_t len = p_pre.size();
        for (int i = len - int_pow_local(4, log4l_pre - 1); i < len; i++) {
            auto s = p_pre[i];
            for (auto &offset_s : offset_supergroup) {
                std::vector<std::vector<int>> s_cur;
                for (auto &g : s) {
                    for (auto &offset_g : offset_s) {
                        std::vector<int> g_cur;
                        for (int j = 0; j < 4; j++) {
                            g_cur.emplace_back(g[j] * 4 + offset_g[j]);
                        }
                        s_cur.emplace_back(g_cur);
                    }
                }
                p.emplace_back(s_cur);
            }
        }
    }

    std::vector<std::vector<std::vector<int>>> pairing_chunks = {
        {{0, 2}, {1, 3}},
        {{0, 3}, {1, 2}}
    };

    if (log2l % 2 == 1) {
        int32_t len_chunk = sub_chunk_per_perm;
        auto p_pre = p;
        p = std::vector<std::vector<std::vector<int>>>();

        for (auto &s : p_pre) {
            std::vector<std::vector<int>> s_cur;
            for (int i = 0; i < int_pow_local(2, log2l); i += int_pow_local(2, log2l - 1)) {
                for (auto &g : s) {
                    std::vector<int> g_cur;
                    for (auto &x : g) {
                        g_cur.emplace_back(x + i);
                    }
                    s_cur.emplace_back(g_cur);
                }
            }
            p.emplace_back(s_cur);
        }

        int32_t len = p_pre.size();
        for (int i = len - int_pow_local(2, log2l - 3); i < len; i++) {
            std::vector<std::vector<int>> s = p_pre[i];
            for (auto &pairing_s : pairing_chunks) {
                std::vector<std::vector<int>> s_cur;
                for (auto &chunk_index : pairing_s) {
                    for (auto &g : s) {
                        std::vector<int> g_cur;
                        for (auto &x : g) {
                            g_cur.emplace_back(chunk_index[x / len_chunk] * len_chunk + x % len_chunk);
                        }
                        s_cur.emplace_back(g_cur);
                    }
                }
                p.emplace_back(s_cur);
            }
        }
    }

    std::vector<std::vector<int>> buffer_states;
    for (auto &supergroup : p) {
        for (auto &state : supergroup) {
            buffer_states.emplace_back(state);
        }
    }
    return buffer_states;
}

std::vector<std::vector<int>> build_slot_pair_owners(const std::vector<std::vector<int>> &template_states, int num_slots) {
    std::vector<std::vector<int>> owners(num_slots, std::vector<int>(num_slots, -1));
    for (int state_idx = 0; state_idx < template_states.size(); state_idx++) {
        for (auto src_slot : template_states[state_idx]) {
            for (auto dst_slot : template_states[state_idx]) {
                if (owners[src_slot][dst_slot] == -1) {
                    owners[src_slot][dst_slot] = state_idx;
                }
            }
        }
    }

    for (int src_slot = 0; src_slot < num_slots; src_slot++) {
        for (int dst_slot = 0; dst_slot < num_slots; dst_slot++) {
            if (owners[src_slot][dst_slot] == -1) {
                throw std::runtime_error("No owner state found for slot pair");
            }
        }
    }

    return owners;
}

std::vector<int64_t> build_partition_hotness(const std::vector<int64_t> &edge_bucket_sizes, int num_partitions) {
    std::vector<int64_t> hotness(num_partitions, 0);
    for (int partition = 0; partition < num_partitions; partition++) {
        int64_t outgoing = 0;
        int64_t incoming = 0;
        for (int other = 0; other < num_partitions; other++) {
            outgoing += edge_bucket_sizes[partition * num_partitions + other];
            incoming += edge_bucket_sizes[other * num_partitions + partition];
        }
        hotness[partition] = outgoing + incoming - edge_bucket_sizes[partition * num_partitions + partition];
    }
    return hotness;
}

std::vector<int> lite_initial_assignment(const std::vector<std::vector<int>> &template_states,
                                         const std::vector<int64_t> &edge_bucket_sizes,
                                         const std::vector<int64_t> &hotness,
                                         int num_partitions,
                                         int active_devices) {
    auto slot_pair_owners = build_slot_pair_owners(template_states, num_partitions);
    std::vector<int64_t> state_weights(template_states.size(), 0);
    std::vector<int> slot_to_partition(num_partitions, -1);
    std::vector<int> assigned_slots;
    const double total_weight = std::accumulate(edge_bucket_sizes.begin(), edge_bucket_sizes.end(), 0.0);
    const double target_state_weight = total_weight / static_cast<double>(template_states.size());

    std::vector<int> sorted_partitions(num_partitions);
    std::iota(sorted_partitions.begin(), sorted_partitions.end(), 0);
    std::sort(sorted_partitions.begin(), sorted_partitions.end(), [&](int lhs, int rhs) {
        if (hotness[lhs] != hotness[rhs]) {
            return hotness[lhs] > hotness[rhs];
        }
        return lhs < rhs;
    });

    for (auto partition : sorted_partitions) {
        int best_slot = -1;
        std::tuple<int64_t, int64_t, double, double, int> best_key{std::numeric_limits<int64_t>::max(),
                                                                    std::numeric_limits<int64_t>::max(),
                                                                    std::numeric_limits<double>::max(),
                                                                    std::numeric_limits<double>::max(),
                                                                    std::numeric_limits<int>::max()};

        for (int slot = 0; slot < num_partitions; slot++) {
            if (slot_to_partition[slot] != -1) {
                continue;
            }

            auto candidate_weights = state_weights;
            int diagonal_owner = slot_pair_owners[slot][slot];
            candidate_weights[diagonal_owner] += edge_bucket_sizes[partition * num_partitions + partition];

            for (auto other_slot : assigned_slots) {
                int other_partition = slot_to_partition[other_slot];
                int forward_owner = slot_pair_owners[slot][other_slot];
                int reverse_owner = slot_pair_owners[other_slot][slot];
                candidate_weights[forward_owner] += edge_bucket_sizes[partition * num_partitions + other_partition];
                candidate_weights[reverse_owner] += edge_bucket_sizes[other_partition * num_partitions + partition];
            }

            int64_t max_round_max = 0;
            int64_t max_round_spread = 0;
            double total_over_target = 0.0;
            double total_abs_deviation = 0.0;
            for (int round_start = 0; round_start < candidate_weights.size(); round_start += active_devices) {
                auto begin = candidate_weights.begin() + round_start;
                auto end = begin + std::min<int>(active_devices, candidate_weights.size() - round_start);
                auto [round_min_it, round_max_it] = std::minmax_element(begin, end);
                max_round_max = std::max<int64_t>(max_round_max, *round_max_it);
                max_round_spread = std::max<int64_t>(max_round_spread, *round_max_it - *round_min_it);
            }
            for (auto weight : candidate_weights) {
                total_over_target += std::max<double>(weight - target_state_weight, 0.0);
                total_abs_deviation += std::abs(weight - target_state_weight);
            }

            auto candidate_key = std::make_tuple(max_round_max, max_round_spread, total_over_target, total_abs_deviation, slot);
            if (candidate_key < best_key) {
                best_key = candidate_key;
                best_slot = slot;
            }
        }

        if (best_slot == -1) {
            throw std::runtime_error("Failed to construct optimized CUSTOM initial assignment");
        }

        slot_to_partition[best_slot] = partition;
        int diagonal_owner = slot_pair_owners[best_slot][best_slot];
        state_weights[diagonal_owner] += edge_bucket_sizes[partition * num_partitions + partition];
        for (auto other_slot : assigned_slots) {
            int other_partition = slot_to_partition[other_slot];
            int forward_owner = slot_pair_owners[best_slot][other_slot];
            int reverse_owner = slot_pair_owners[other_slot][best_slot];
            state_weights[forward_owner] += edge_bucket_sizes[partition * num_partitions + other_partition];
            state_weights[reverse_owner] += edge_bucket_sizes[other_partition * num_partitions + partition];
        }
        assigned_slots.emplace_back(best_slot);
    }

    return slot_to_partition;
}

std::pair<int64_t, int64_t> custom_state_transition_cost(const CustomStateMetrics &previous_state,
                                                         const CustomStateMetrics &next_state,
                                                         const std::vector<int64_t> &hotness) {
    int64_t transition_hotness = 0;
    int64_t transition_new_partitions = 0;

    for (auto partition : next_state.partitions) {
        if (std::find(previous_state.partitions.begin(), previous_state.partitions.end(), partition) == previous_state.partitions.end()) {
            transition_hotness += hotness[partition];
            transition_new_partitions++;
        }
    }
    return std::make_pair(transition_hotness, transition_new_partitions);
}

std::tuple<std::vector<std::vector<int>>, int64_t, int64_t> optimize_custom_lane_assignment(
    const std::vector<std::vector<int>> &rounds,
    const std::vector<CustomStateMetrics> &states,
    const std::vector<int64_t> &hotness) {
    if (rounds.empty()) {
        return std::make_tuple(std::vector<std::vector<int>>(), 0, 0);
    }

    std::vector<std::vector<std::vector<int>>> all_permutations;
    all_permutations.reserve(rounds.size());
    for (auto &round : rounds) {
        std::vector<int> permutation(round.size());
        std::iota(permutation.begin(), permutation.end(), 0);
        std::vector<std::vector<int>> round_permutations;
        do {
            round_permutations.emplace_back(permutation);
        } while (std::next_permutation(permutation.begin(), permutation.end()));
        all_permutations.emplace_back(std::move(round_permutations));
    }

    std::vector<std::vector<std::pair<int64_t, int64_t>>> dp(rounds.size());
    std::vector<std::vector<int>> backpointers(rounds.size());
    dp[0].assign(all_permutations[0].size(), std::make_pair(0, 0));
    backpointers[0].assign(all_permutations[0].size(), -1);

    for (int round_idx = 1; round_idx < rounds.size(); round_idx++) {
        dp[round_idx].assign(all_permutations[round_idx].size(),
                             std::make_pair(std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::max()));
        backpointers[round_idx].assign(all_permutations[round_idx].size(), -1);

        for (int permutation_idx = 0; permutation_idx < all_permutations[round_idx].size(); permutation_idx++) {
            const auto &permutation = all_permutations[round_idx][permutation_idx];

            for (int previous_permutation_idx = 0; previous_permutation_idx < all_permutations[round_idx - 1].size(); previous_permutation_idx++) {
                const auto &previous_permutation = all_permutations[round_idx - 1][previous_permutation_idx];
                int64_t transition_hotness = 0;
                int64_t transition_new_partitions = 0;

                for (int lane_idx = 0; lane_idx < permutation.size(); lane_idx++) {
                    const auto &previous_state = states[rounds[round_idx - 1][previous_permutation[lane_idx]]];
                    const auto &next_state = states[rounds[round_idx][permutation[lane_idx]]];
                    auto [lane_hotness, lane_new_partitions] = custom_state_transition_cost(previous_state, next_state, hotness);
                    transition_hotness += lane_hotness;
                    transition_new_partitions += lane_new_partitions;
                }

                auto candidate_cost =
                    std::make_pair(dp[round_idx - 1][previous_permutation_idx].first + transition_hotness,
                                   dp[round_idx - 1][previous_permutation_idx].second + transition_new_partitions);
                if (candidate_cost < dp[round_idx][permutation_idx]) {
                    dp[round_idx][permutation_idx] = candidate_cost;
                    backpointers[round_idx][permutation_idx] = previous_permutation_idx;
                }
            }
        }
    }

    int best_final_idx = 0;
    for (int permutation_idx = 1; permutation_idx < dp.back().size(); permutation_idx++) {
        if (dp.back()[permutation_idx] < dp.back()[best_final_idx]) {
            best_final_idx = permutation_idx;
        }
    }

    std::vector<int> chosen(rounds.size(), -1);
    chosen.back() = best_final_idx;
    for (int round_idx = rounds.size() - 1; round_idx > 0; round_idx--) {
        chosen[round_idx - 1] = backpointers[round_idx][chosen[round_idx]];
    }

    std::vector<std::vector<int>> lane_rounds;
    lane_rounds.reserve(rounds.size());
    for (int round_idx = 0; round_idx < rounds.size(); round_idx++) {
        std::vector<int> lane_round;
        lane_round.reserve(rounds[round_idx].size());
        for (auto local_idx : all_permutations[round_idx][chosen[round_idx]]) {
            lane_round.emplace_back(rounds[round_idx][local_idx]);
        }
        lane_rounds.emplace_back(std::move(lane_round));
    }

    return std::make_tuple(lane_rounds, dp.back()[best_final_idx].first, dp.back()[best_final_idx].second);
}

CustomEvaluatedSchedule summarize_custom_schedule(const std::vector<std::vector<int>> &template_states,
                                                 const std::vector<int> &slot_to_partition,
                                                 const std::vector<int64_t> &edge_bucket_sizes,
                                                 const std::vector<int64_t> &hotness,
                                                 int num_partitions,
                                                 int active_devices,
                                                 int batch_size) {
    std::vector<std::vector<int>> mapped_states = template_states;
    for (auto &state : mapped_states) {
        for (auto &slot : state) {
            slot = slot_to_partition[slot];
        }
    }

    auto edge_buckets = greedyAssignEdgeBucketsToBuffers(mapped_states, num_partitions);

    std::vector<CustomStateMetrics> state_metrics;
    state_metrics.reserve(mapped_states.size());
    for (int state_idx = 0; state_idx < mapped_states.size(); state_idx++) {
        int64_t weight = 0;
        for (auto &[src, dst] : edge_buckets[state_idx]) {
            weight += edge_bucket_sizes[src * num_partitions + dst];
        }
        int64_t batches = (weight + batch_size - 1) / batch_size;
        state_metrics.push_back({mapped_states[state_idx], weight, batches, static_cast<int64_t>(edge_buckets[state_idx].size())});
    }

    std::vector<std::vector<int>> rounds;
    for (int i = 0; i < state_metrics.size(); i += active_devices) {
        std::vector<int> round;
        for (int j = i; j < std::min<int>(i + active_devices, state_metrics.size()); j++) {
            round.emplace_back(j);
        }
        rounds.emplace_back(std::move(round));
    }

    auto [lane_rounds, continuity_hotness, continuity_new_partitions] = optimize_custom_lane_assignment(rounds, state_metrics, hotness);

    int64_t worst_round_spread = 0;
    int64_t worst_batch_spread = 0;
    int64_t worst_state_weight = 0;
    int64_t total_round_spread = 0;
    double total_abs_deviation = 0.0;
    for (auto &round : rounds) {
        int64_t round_min_weight = std::numeric_limits<int64_t>::max();
        int64_t round_max_weight = 0;
        int64_t round_min_batches = std::numeric_limits<int64_t>::max();
        int64_t round_max_batches = 0;
        double round_mean_weight = 0.0;
        for (auto state_idx : round) {
            round_min_weight = std::min<int64_t>(round_min_weight, state_metrics[state_idx].weight);
            round_max_weight = std::max<int64_t>(round_max_weight, state_metrics[state_idx].weight);
            round_min_batches = std::min<int64_t>(round_min_batches, state_metrics[state_idx].batches);
            round_max_batches = std::max<int64_t>(round_max_batches, state_metrics[state_idx].batches);
            worst_state_weight = std::max<int64_t>(worst_state_weight, state_metrics[state_idx].weight);
            round_mean_weight += static_cast<double>(state_metrics[state_idx].weight);
        }
        round_mean_weight /= static_cast<double>(round.size());
        for (auto state_idx : round) {
            total_abs_deviation += std::abs(static_cast<double>(state_metrics[state_idx].weight) - round_mean_weight);
        }
        int64_t round_spread = round_max_weight - round_min_weight;
        int64_t batch_spread = round_max_batches - round_min_batches;
        worst_round_spread = std::max<int64_t>(worst_round_spread, round_spread);
        worst_batch_spread = std::max<int64_t>(worst_batch_spread, batch_spread);
        total_round_spread += round_spread;
    }

    CustomScheduleScore score = {
        worst_round_spread,
        worst_batch_spread,
        worst_state_weight,
        total_round_spread,
        continuity_hotness,
        continuity_new_partitions,
        static_cast<int64_t>(total_abs_deviation),
    };

    return {slot_to_partition, state_metrics, rounds, lane_rounds, score};
}

CustomEvaluatedSchedule steepest_descent_custom_schedule(const std::vector<std::vector<int>> &template_states,
                                                         const std::vector<int> &initial_assignment,
                                                         const std::vector<int64_t> &edge_bucket_sizes,
                                                         const std::vector<int64_t> &hotness,
                                                         int num_partitions,
                                                         int active_devices,
                                                         int batch_size) {
    std::vector<int> assignment = initial_assignment;
    auto current = summarize_custom_schedule(template_states, assignment, edge_bucket_sizes, hotness, num_partitions, active_devices, batch_size);

    bool improved = true;
    while (improved) {
        improved = false;
        auto best_candidate = current;
        std::pair<int, int> best_swap{-1, -1};

        for (int i = 0; i < assignment.size(); i++) {
            for (int j = i + 1; j < assignment.size(); j++) {
                auto candidate_assignment = assignment;
                std::swap(candidate_assignment[i], candidate_assignment[j]);
                auto candidate =
                    summarize_custom_schedule(template_states, candidate_assignment, edge_bucket_sizes, hotness, num_partitions, active_devices, batch_size);
                if (custom_score_better(candidate.score, best_candidate.score)) {
                    best_candidate = std::move(candidate);
                    best_swap = std::make_pair(i, j);
                }
            }
        }

        if (best_swap.first != -1) {
            std::swap(assignment[best_swap.first], assignment[best_swap.second]);
            current = std::move(best_candidate);
            improved = true;
        }
    }

    return current;
}

CustomEvaluatedSchedule optimize_custom_schedule(const std::vector<std::vector<int>> &template_states,
                                                 const std::vector<int64_t> &edge_bucket_sizes,
                                                 int num_partitions,
                                                 int active_devices,
                                                 int batch_size) {
    auto hotness = build_partition_hotness(edge_bucket_sizes, num_partitions);
    std::mt19937 rng(static_cast<uint32_t>(optimized_custom_schedule_seed()));
    int64_t restarts = optimized_custom_schedule_restarts();

    std::vector<std::vector<int>> initial_assignments;
    initial_assignments.emplace_back(lite_initial_assignment(template_states, edge_bucket_sizes, hotness, num_partitions, active_devices));

    std::vector<int> identity(num_partitions);
    std::iota(identity.begin(), identity.end(), 0);
    initial_assignments.emplace_back(identity);

    for (int64_t restart = 1; restart < restarts; restart++) {
        auto candidate = identity;
        std::shuffle(candidate.begin(), candidate.end(), rng);
        initial_assignments.emplace_back(std::move(candidate));
    }

    bool has_best = false;
    CustomEvaluatedSchedule best;
    for (auto &initial_assignment : initial_assignments) {
        auto candidate = steepest_descent_custom_schedule(template_states, initial_assignment, edge_bucket_sizes, hotness, num_partitions,
                                                          active_devices, batch_size);
        if (!has_best || custom_score_better(candidate.score, best.score)) {
            best = std::move(candidate);
            has_best = true;
        }
    }

    if (!has_best) {
        throw std::runtime_error("No optimized CUSTOM schedule candidates were generated");
    }

    return best;
}

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> build_optimized_custom_edge_bucket_ordering(int num_partitions,
                                                                                                      int buffer_capacity,
                                                                                                      int active_devices,
                                                                                                      int batch_size,
                                                                                                      const std::vector<int64_t> &edge_bucket_sizes) {
    auto template_states = build_custom_template_states(num_partitions, buffer_capacity);
    auto optimized = optimize_custom_schedule(template_states, edge_bucket_sizes, num_partitions, active_devices, batch_size);

    std::vector<std::vector<int>> mapped_states = template_states;
    for (auto &state : mapped_states) {
        for (auto &slot : state) {
            slot = optimized.slot_to_partition[slot];
        }
    }

    auto edge_buckets_per_buffer = greedyAssignEdgeBucketsToBuffers(mapped_states, num_partitions);
    std::vector<int64_t> resident_state_weights;
    resident_state_weights.reserve(mapped_states.size());
    for (const auto &state : mapped_states) {
        resident_state_weights.emplace_back(compute_state_resident_weight(state, edge_bucket_sizes, num_partitions));
    }
    std::vector<std::vector<int>> ordered_states;
    std::vector<std::vector<std::pair<int, int>>> ordered_buckets;
    ordered_states.reserve(mapped_states.size());
    ordered_buckets.reserve(edge_buckets_per_buffer.size());

    auto ordered_lane_rounds = optimized.lane_rounds;
    int startup_round = select_startup_round(ordered_lane_rounds, resident_state_weights, optimized.states);
    if (startup_round > 0) {
        std::rotate(ordered_lane_rounds.begin(), ordered_lane_rounds.begin() + startup_round, ordered_lane_rounds.end());
    }

    for (const auto &lane_round : ordered_lane_rounds) {
        for (auto state_idx : lane_round) {
            ordered_states.emplace_back(mapped_states[state_idx]);
            ordered_buckets.emplace_back(edge_buckets_per_buffer[state_idx]);
        }
    }

    std::ostringstream slot_mapping;
    for (int idx = 0; idx < optimized.slot_to_partition.size(); idx++) {
        if (idx > 0) {
            slot_mapping << ",";
        }
        slot_mapping << optimized.slot_to_partition[idx];
    }

    SPDLOG_INFO(
        "Using optimized CUSTOM ordering: worst_round_spread={:.3f}M worst_batch_spread={} total_round_spread={:.3f}M continuity_hotness={:.3f}M continuity_new_partitions={}",
        optimized.score.worst_round_spread / 1000000.0,
        optimized.score.worst_batch_spread,
        optimized.score.total_round_spread / 1000000.0,
        optimized.score.continuity_hotness / 1000000.0,
        optimized.score.continuity_new_partitions);
    SPDLOG_INFO("Optimized CUSTOM slot_to_partition=[{}]", slot_mapping.str());
    if (!ordered_lane_rounds.empty()) {
        int64_t startup_max_resident_weight = 0;
        int64_t startup_total_resident_weight = 0;
        for (auto state_idx : ordered_lane_rounds.front()) {
            startup_max_resident_weight = std::max<int64_t>(startup_max_resident_weight, resident_state_weights[state_idx]);
            startup_total_resident_weight += resident_state_weights[state_idx];
        }
        SPDLOG_INFO("Optimized CUSTOM startup round={} startup_max_resident_edges={:.3f}M startup_total_resident_edges={:.3f}M",
                    startup_round,
                    startup_max_resident_weight / 1000000.0,
                    startup_total_resident_weight / 1000000.0);
    }

    return convertEdgeBucketOrderToTensors(ordered_states, ordered_buckets);
}

bool states_disjoint(const std::vector<int64_t> &lhs, const std::vector<int64_t> &rhs) {
    for (auto left_part : lhs) {
        for (auto right_part : rhs) {
            if (left_part == right_part) {
                return false;
            }
        }
    }
    return true;
}

bool search_disjoint_groups(const std::vector<std::vector<bool>> &compatible, const std::vector<int64_t> &remaining, int active_devices,
                            std::vector<std::vector<int64_t>> &groups);

bool search_group_members(const std::vector<std::vector<bool>> &compatible, const std::vector<int64_t> &remaining, const int active_devices,
                          const int target_group_size, const std::vector<int64_t> &candidates, std::vector<int64_t> &current_group,
                          std::vector<std::vector<int64_t>> &groups) {
    if (current_group.size() == static_cast<std::size_t>(target_group_size)) {
        std::vector<int64_t> next_remaining;
        next_remaining.reserve(remaining.size() - current_group.size());
        for (auto state_idx : remaining) {
            if (std::find(current_group.begin(), current_group.end(), state_idx) == current_group.end()) {
                next_remaining.emplace_back(state_idx);
            }
        }
        groups.emplace_back(current_group);
        if (search_disjoint_groups(compatible, next_remaining, active_devices, groups)) {
            return true;
        }
        groups.pop_back();
        return false;
    }

    if (current_group.size() + candidates.size() < static_cast<std::size_t>(target_group_size)) {
        return false;
    }

    for (std::size_t i = 0; i < candidates.size(); i++) {
        int64_t candidate = candidates[i];
        std::vector<int64_t> next_candidates;
        next_candidates.reserve(candidates.size() - i - 1);
        for (std::size_t j = i + 1; j < candidates.size(); j++) {
            if (compatible[candidate][candidates[j]]) {
                next_candidates.emplace_back(candidates[j]);
            }
        }
        current_group.emplace_back(candidate);
        if (search_group_members(compatible, remaining, active_devices, target_group_size, next_candidates, current_group, groups)) {
            return true;
        }
        current_group.pop_back();
    }

    return false;
}

bool search_disjoint_groups(const std::vector<std::vector<bool>> &compatible, const std::vector<int64_t> &remaining, int active_devices,
                            std::vector<std::vector<int64_t>> &groups) {
    if (remaining.empty()) {
        return true;
    }

    int target_group_size = std::min<int>(active_devices, remaining.size());
    if (target_group_size <= 1) {
        groups.emplace_back(remaining);
        return true;
    }

    auto anchor_it = std::min_element(remaining.begin(), remaining.end(), [&](int64_t lhs, int64_t rhs) {
        int lhs_degree = 0;
        int rhs_degree = 0;
        for (auto state_idx : remaining) {
            lhs_degree += compatible[lhs][state_idx] ? 1 : 0;
            rhs_degree += compatible[rhs][state_idx] ? 1 : 0;
        }
        return lhs_degree < rhs_degree;
    });
    int64_t anchor = *anchor_it;

    std::vector<int64_t> candidates;
    for (auto state_idx : remaining) {
        if (state_idx != anchor && compatible[anchor][state_idx]) {
            candidates.emplace_back(state_idx);
        }
    }

    std::vector<int64_t> current_group = {anchor};
    return search_group_members(compatible, remaining, active_devices, target_group_size, candidates, current_group, groups);
}

std::vector<std::vector<int64_t>> build_disjoint_groups(const vector<torch::Tensor> &buffer_states, int active_devices) {
    std::vector<std::vector<int64_t>> groups;
    if (active_devices <= 1 || buffer_states.size() <= 1) {
        groups.reserve(buffer_states.size());
        for (std::size_t i = 0; i < buffer_states.size(); i++) {
            groups.push_back({static_cast<int64_t>(i)});
        }
        return groups;
    }

    std::vector<std::vector<int64_t>> state_partitions;
    state_partitions.reserve(buffer_states.size());
    for (auto &state : buffer_states) {
        state_partitions.emplace_back(tensor_to_partitions(state));
    }

    std::vector<std::vector<bool>> compatible(buffer_states.size(), std::vector<bool>(buffer_states.size(), false));
    for (std::size_t i = 0; i < buffer_states.size(); i++) {
        compatible[i][i] = true;
        for (std::size_t j = i + 1; j < buffer_states.size(); j++) {
            bool disjoint = states_disjoint(state_partitions[i], state_partitions[j]);
            compatible[i][j] = disjoint;
            compatible[j][i] = disjoint;
        }
    }

    std::vector<int64_t> remaining(buffer_states.size());
    std::iota(remaining.begin(), remaining.end(), 0);
    if (!search_disjoint_groups(compatible, remaining, active_devices, groups)) {
        groups.clear();
        groups.reserve(buffer_states.size());
        for (std::size_t i = 0; i < buffer_states.size(); i++) {
            groups.push_back({static_cast<int64_t>(i)});
        }
    }

    return groups;
}

std::vector<StateAccessSummary> build_state_access_summaries(const vector<torch::Tensor> &buffer_states,
                                                             const vector<torch::Tensor> &edge_buckets_per_buffer) {
    std::vector<StateAccessSummary> summaries(buffer_states.size());
    for (std::size_t i = 0; i < buffer_states.size(); i++) {
        auto partitions = tensor_to_partitions(buffer_states[i]);
        std::sort(partitions.begin(), partitions.end());
        summaries[i].partitions = std::move(partitions);

        if (i >= edge_buckets_per_buffer.size()) {
            continue;
        }

        if (!edge_buckets_per_buffer[i].defined()) {
            continue;
        }

        auto edge_buckets = edge_buckets_per_buffer[i].to(torch::kCPU).to(torch::kInt64).contiguous();
        if (!edge_buckets.defined() || edge_buckets.numel() == 0) {
            continue;
        }
        auto accessor = edge_buckets.accessor<int64_t, 2>();
        for (int64_t row = 0; row < edge_buckets.size(0); row++) {
            summaries[i].incident_bucket_counts[accessor[row][0]]++;
            summaries[i].incident_bucket_counts[accessor[row][1]]++;
        }
    }
    return summaries;
}

int64_t state_access_overlap_score(const StateAccessSummary &lhs, const StateAccessSummary &rhs) {
    std::size_t i = 0;
    std::size_t j = 0;
    int64_t score = 0;
    while (i < lhs.partitions.size() && j < rhs.partitions.size()) {
        if (lhs.partitions[i] == rhs.partitions[j]) {
            int64_t partition_id = lhs.partitions[i];
            auto lhs_it = lhs.incident_bucket_counts.find(partition_id);
            auto rhs_it = rhs.incident_bucket_counts.find(partition_id);
            int64_t lhs_incident = lhs_it == lhs.incident_bucket_counts.end() ? 0 : lhs_it->second;
            int64_t rhs_incident = rhs_it == rhs.incident_bucket_counts.end() ? 0 : rhs_it->second;
            score += 1000 + std::min(lhs_incident, rhs_incident);
            i++;
            j++;
        } else if (lhs.partitions[i] < rhs.partitions[j]) {
            i++;
        } else {
            j++;
        }
    }
    return score;
}

struct GroupAlignmentResult {
    int64_t score = std::numeric_limits<int64_t>::min();
    std::vector<int64_t> ordered_group;
};

void search_best_group_alignment(const std::vector<int64_t> &prev_group,
                                 const std::vector<int64_t> &candidate_group,
                                 const std::vector<StateAccessSummary> &summaries,
                                 std::vector<bool> &used,
                                 std::vector<int64_t> &current,
                                 std::size_t slot,
                                 int64_t current_score,
                                 GroupAlignmentResult &best) {
    std::size_t target = std::min(prev_group.size(), candidate_group.size());
    if (slot == target) {
        std::vector<int64_t> ordered = current;
        for (auto candidate_state : candidate_group) {
            if (std::find(ordered.begin(), ordered.end(), candidate_state) == ordered.end()) {
                ordered.emplace_back(candidate_state);
            }
        }
        if (current_score > best.score || (current_score == best.score && ordered < best.ordered_group)) {
            best.score = current_score;
            best.ordered_group = std::move(ordered);
        }
        return;
    }

    for (std::size_t i = 0; i < candidate_group.size(); i++) {
        if (used[i]) {
            continue;
        }
        used[i] = true;
        int64_t candidate_state = candidate_group[i];
        current.emplace_back(candidate_state);
        int64_t step_score = state_access_overlap_score(summaries[prev_group[slot]], summaries[candidate_state]);
        search_best_group_alignment(prev_group, candidate_group, summaries, used, current, slot + 1, current_score + step_score, best);
        current.pop_back();
        used[i] = false;
    }
}

GroupAlignmentResult get_best_group_alignment(const std::vector<int64_t> &prev_group,
                                              const std::vector<int64_t> &candidate_group,
                                              const std::vector<StateAccessSummary> &summaries) {
    GroupAlignmentResult result;
    if (candidate_group.empty()) {
        result.score = 0;
        return result;
    }
    if (prev_group.empty()) {
        result.score = 0;
        result.ordered_group = candidate_group;
        return result;
    }

    std::vector<bool> used(candidate_group.size(), false);
    std::vector<int64_t> current;
    current.reserve(candidate_group.size());
    search_best_group_alignment(prev_group, candidate_group, summaries, used, current, 0, 0, result);
    if (result.score == std::numeric_limits<int64_t>::min()) {
        result.score = 0;
        result.ordered_group = candidate_group;
    }
    return result;
}

struct GroupSearchResult {
    int64_t score = std::numeric_limits<int64_t>::min();
    std::vector<int64_t> ordered_group;
    std::vector<int64_t> chosen_states;
};

void search_best_disjoint_group(const std::vector<std::vector<bool>> &compatible,
                                const std::vector<int64_t> &remaining,
                                const std::vector<int64_t> &prev_group,
                                const std::vector<StateAccessSummary> &summaries,
                                int target_group_size,
                                std::size_t start_idx,
                                std::vector<int64_t> &current_group,
                                GroupSearchResult &best) {
    if (current_group.size() == static_cast<std::size_t>(target_group_size)) {
        GroupSearchResult candidate;
        candidate.chosen_states = current_group;
        if (prev_group.empty()) {
            int64_t score = 0;
            for (auto remaining_state : remaining) {
                if (std::find(current_group.begin(), current_group.end(), remaining_state) != current_group.end()) {
                    continue;
                }
                int64_t best_overlap = 0;
                for (auto group_state : current_group) {
                    best_overlap = std::max(best_overlap, state_access_overlap_score(summaries[group_state], summaries[remaining_state]));
                }
                score += best_overlap;
            }
            candidate.score = score;
            candidate.ordered_group = current_group;
        } else {
            auto alignment = get_best_group_alignment(prev_group, current_group, summaries);
            candidate.score = alignment.score;
            candidate.ordered_group = std::move(alignment.ordered_group);
        }

        if (candidate.score > best.score || (candidate.score == best.score && candidate.ordered_group < best.ordered_group)) {
            best = std::move(candidate);
        }
        return;
    }

    for (std::size_t i = start_idx; i < remaining.size(); i++) {
        int64_t candidate_state = remaining[i];
        bool valid = true;
        for (auto chosen_state : current_group) {
            if (!compatible[candidate_state][chosen_state]) {
                valid = false;
                break;
            }
        }
        if (!valid) {
            continue;
        }
        current_group.emplace_back(candidate_state);
        search_best_disjoint_group(compatible, remaining, prev_group, summaries, target_group_size, i + 1, current_group, best);
        current_group.pop_back();
    }
}

struct AccessAwareGeneratedState {
    std::vector<int> partitions;
    std::vector<std::pair<int, int>> newly_covered_buckets;
};

struct AccessAwareGroupSearch {
    int64_t score = std::numeric_limits<int64_t>::min();
    std::vector<AccessAwareGeneratedState> states;
};

int64_t partition_overlap_count(const std::vector<int> &lhs, const std::vector<int> &rhs) {
    int64_t overlap = 0;
    for (auto left_part : lhs) {
        for (auto right_part : rhs) {
            if (left_part == right_part) {
                overlap++;
            }
        }
    }
    return overlap;
}

std::vector<int> sorted_partitions_from_mask(uint64_t mask, int num_partitions) {
    std::vector<int> partitions;
    for (int part = 0; part < num_partitions; part++) {
        if (((mask >> part) & 1ULL) != 0ULL) {
            partitions.emplace_back(part);
        }
    }
    return partitions;
}

AccessAwareGeneratedState build_generated_state(uint64_t mask, const std::vector<bool> &uncovered, int num_partitions) {
    AccessAwareGeneratedState state;
    state.partitions = sorted_partitions_from_mask(mask, num_partitions);
    for (auto src_part : state.partitions) {
        for (auto dst_part : state.partitions) {
            int bucket_idx = src_part * num_partitions + dst_part;
            if (uncovered[bucket_idx]) {
                state.newly_covered_buckets.emplace_back(src_part, dst_part);
            }
        }
    }
    return state;
}

int64_t state_new_bucket_gain(uint64_t mask, const std::vector<bool> &uncovered, int num_partitions) {
    int64_t gain = 0;
    for (int src_part = 0; src_part < num_partitions; src_part++) {
        if (((mask >> src_part) & 1ULL) == 0ULL) {
            continue;
        }
        for (int dst_part = 0; dst_part < num_partitions; dst_part++) {
            if (((mask >> dst_part) & 1ULL) == 0ULL) {
                continue;
            }
            gain += uncovered[src_part * num_partitions + dst_part] ? 1 : 0;
        }
    }
    return gain;
}

int64_t group_new_bucket_gain(const std::vector<uint64_t> &group_masks, const std::vector<bool> &uncovered, int num_partitions) {
    int64_t gain = 0;
    for (auto mask : group_masks) {
        gain += state_new_bucket_gain(mask, uncovered, num_partitions);
    }
    return gain;
}

void search_access_aware_groups(uint64_t available_mask, int num_partitions, int buffer_capacity,
                                const std::vector<bool> &uncovered, const std::vector<std::vector<int>> &prev_group,
                                std::vector<uint64_t> &current_group, AccessAwareGroupSearch &best) {
    int remaining_partitions = static_cast<int>(__builtin_popcountll(available_mask));
    if (remaining_partitions == 0) {
        AccessAwareGroupSearch candidate;
        std::vector<std::vector<int>> current_partitions;
        current_partitions.reserve(current_group.size());
        for (auto mask : current_group) {
            current_partitions.emplace_back(sorted_partitions_from_mask(mask, num_partitions));
        }

        int64_t coverage_gain = group_new_bucket_gain(current_group, uncovered, num_partitions);
        int64_t overlap_gain = 0;
        if (!prev_group.empty() && prev_group.size() == current_partitions.size()) {
            for (std::size_t lane = 0; lane < current_partitions.size(); lane++) {
                overlap_gain += partition_overlap_count(prev_group[lane], current_partitions[lane]);
            }
        }

        // Coverage dominates; overlap breaks ties toward better locality.
        candidate.score = coverage_gain * 100 + overlap_gain * 7;
        for (auto mask : current_group) {
            candidate.states.emplace_back(build_generated_state(mask, uncovered, num_partitions));
        }
        if (candidate.score > best.score) {
            best = std::move(candidate);
        }
        return;
    }

    if (remaining_partitions < buffer_capacity) {
        return;
    }

    int anchor = 0;
    while (((available_mask >> anchor) & 1ULL) == 0ULL) {
        anchor++;
    }

    std::vector<int> rest;
    rest.reserve(remaining_partitions - 1);
    for (int part = anchor + 1; part < num_partitions; part++) {
        if (((available_mask >> part) & 1ULL) != 0ULL) {
            rest.emplace_back(part);
        }
    }

    for (std::size_t i = 0; i < rest.size(); i++) {
        for (std::size_t j = i + 1; j < rest.size(); j++) {
            for (std::size_t k = j + 1; k < rest.size(); k++) {
                uint64_t state_mask = (1ULL << anchor) | (1ULL << rest[i]) | (1ULL << rest[j]) | (1ULL << rest[k]);
                current_group.emplace_back(state_mask);
                search_access_aware_groups(available_mask & ~state_mask, num_partitions, buffer_capacity, uncovered, prev_group, current_group, best);
                current_group.pop_back();
            }
        }
    }
}

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> generate_access_aware_states(int num_partitions, int buffer_capacity, int active_devices) {
    if (buffer_capacity != 4 || active_devices <= 1 || active_devices * buffer_capacity != num_partitions || num_partitions > 63) {
        return getCustomEdgeBucketOrdering(num_partitions, buffer_capacity, false);
    }

    const int total_buckets = num_partitions * num_partitions;
    std::vector<bool> uncovered(total_buckets, true);
    int uncovered_count = total_buckets;

    std::vector<std::vector<int>> prev_group;
    std::vector<std::vector<int>> buffer_states;
    std::vector<std::vector<std::pair<int, int>>> edge_buckets_per_buffer;

    int superstep = 0;
    const int max_supersteps = total_buckets;
    while (uncovered_count > 0 && superstep < max_supersteps) {
        AccessAwareGroupSearch best_group;
        std::vector<uint64_t> current_group;
        current_group.reserve(active_devices);
        const uint64_t all_partitions_mask = (1ULL << num_partitions) - 1ULL;
        search_access_aware_groups(all_partitions_mask, num_partitions, buffer_capacity, uncovered, prev_group, current_group, best_group);

        if (best_group.states.empty()) {
            break;
        }

        int64_t covered_this_step = 0;
        std::vector<std::vector<int>> current_partitions;
        current_partitions.reserve(best_group.states.size());
        for (auto &state : best_group.states) {
            current_partitions.emplace_back(state.partitions.begin(), state.partitions.end());
            buffer_states.emplace_back(state.partitions.begin(), state.partitions.end());
            edge_buckets_per_buffer.emplace_back();
            auto &assigned = edge_buckets_per_buffer.back();
            assigned.reserve(state.newly_covered_buckets.size());
            for (auto &bucket : state.newly_covered_buckets) {
                int bucket_idx = bucket.first * num_partitions + bucket.second;
                if (!uncovered[bucket_idx]) {
                    continue;
                }
                uncovered[bucket_idx] = false;
                uncovered_count--;
                covered_this_step++;
                assigned.emplace_back(bucket);
            }
        }

        if (covered_this_step == 0) {
            break;
        }

        prev_group = std::move(current_partitions);
        superstep++;
    }

    if (uncovered_count > 0 || buffer_states.empty()) {
        return getCustomEdgeBucketOrdering(num_partitions, buffer_capacity, false);
    }

    return convertEdgeBucketOrderToTensors(buffer_states, edge_buckets_per_buffer);
}

}  // namespace

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getEdgeBucketOrdering(EdgeBucketOrdering edge_bucket_ordering, int num_partitions, int buffer_capacity,
                                                                               int fine_to_coarse_ratio, int num_cache_partitions,
                                                                               bool randomly_assign_edge_buckets) {
    switch (edge_bucket_ordering) {
        case EdgeBucketOrdering::OLD_BETA:
            SPDLOG_INFO("Generating Old Beta Ordering");
            return getTwoLevelBetaOrdering(num_partitions, buffer_capacity, 1, 0, false);
        case EdgeBucketOrdering::NEW_BETA:
            SPDLOG_INFO("Generating New Beta Ordering");
            return getTwoLevelBetaOrdering(num_partitions, buffer_capacity, 1, 0, true);
        case EdgeBucketOrdering::ALL_BETA:
            return getCustomEdgeBucketOrdering();
        case EdgeBucketOrdering::COMET:
            SPDLOG_INFO("Generating COMET Ordering");
            return getTwoLevelBetaOrdering(num_partitions, buffer_capacity, fine_to_coarse_ratio, num_cache_partitions, randomly_assign_edge_buckets);
        case EdgeBucketOrdering::CUSTOM:
            SPDLOG_INFO("Generating CUSTOM Ordering");
            return getCustomEdgeBucketOrdering(num_partitions, buffer_capacity, randomly_assign_edge_buckets);
        default:
            SPDLOG_ERROR("Not implemented");
            std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> ret;
            return ret;
    }
}

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getNodePartitionOrdering(NodePartitionOrdering node_partition_ordering, Indices train_nodes,
                                                                                  int64_t total_num_nodes, int num_partitions, int buffer_capacity,
                                                                                  int fine_to_coarse_ratio, int num_cache_partitions) {
    switch (node_partition_ordering) {
        case NodePartitionOrdering::DISPERSED:
            SPDLOG_INFO("Generating Dispersed Ordering");
            return getDispersedNodePartitionOrdering(train_nodes, total_num_nodes, num_partitions, buffer_capacity, fine_to_coarse_ratio, num_cache_partitions);
        case NodePartitionOrdering::SEQUENTIAL:
            SPDLOG_INFO("Generating Sequential Ordering");
            return getSequentialNodePartitionOrdering(train_nodes, total_num_nodes, num_partitions, buffer_capacity);
        case NodePartitionOrdering::CUSTOM:
            return getCustomNodePartitionOrdering();
        default:
            SPDLOG_ERROR("Not implemented");
            std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> ret;
            return ret;
    }
}

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> convertEdgeBucketOrderToTensors(vector<vector<int>> buffer_states,
                                                                                         vector<vector<std::pair<int, int>>> edge_buckets_per_buffer) {
    vector<torch::Tensor> ret_buffer_states;
    vector<torch::Tensor> ret_edge_buckets_per_buffer;

    for (auto b : buffer_states) {
        ret_buffer_states.emplace_back(torch::tensor(b, torch::kInt64));
    }

    for (auto edge_buckets : edge_buckets_per_buffer) {
        torch::Tensor tmp = torch::zeros({(int64_t)edge_buckets.size(), 2}, torch::kInt64);

        for (int i = 0; i < edge_buckets.size(); i++) {
            tmp[i][0] = std::get<0>(edge_buckets[i]);
            tmp[i][1] = std::get<1>(edge_buckets[i]);
        }

        ret_edge_buckets_per_buffer.emplace_back(tmp);
    }

    return std::forward_as_tuple(ret_buffer_states, ret_edge_buckets_per_buffer);
}

vector<vector<int>> getBetaOrderingHelper(int num_partitions, int buffer_capacity) {
    vector<vector<int>> buffer_states;
    Indices all_partitions = torch::randperm(num_partitions, torch::kInt32);

    // get all buffer states
    Indices in_buffer = all_partitions.index_select(0, torch::arange(buffer_capacity));

    Indices combined = torch::cat({all_partitions, in_buffer});
    auto uniques = unique_with_counts_sorted(combined);
    auto vals = std::get<0>(uniques);
    auto counts = std::get<1>(uniques);
    Indices on_disk = vals.masked_select(counts == 1);

    int *data_ptr_ = (int *)in_buffer.data_ptr();
    buffer_states.emplace_back(vector<int>(data_ptr_, data_ptr_ + in_buffer.size(0)));

    while (on_disk.size(0) >= 1) {
        in_buffer = in_buffer.index_select(0, torch::randperm(in_buffer.size(0), torch::kInt64));
        on_disk = on_disk.index_select(0, torch::randperm(on_disk.size(0), torch::kInt64));

        for (int i = 0; i < on_disk.size(0); i++) {
            auto admit_id = on_disk[i].clone();

            on_disk[i] = in_buffer[-1];

            in_buffer[-1] = admit_id;

            data_ptr_ = (int *)in_buffer.data_ptr();
            buffer_states.emplace_back(vector<int>(data_ptr_, data_ptr_ + in_buffer.size(0)));
        }

        on_disk = on_disk.index_select(0, torch::randperm(on_disk.size(0), torch::kInt64));

        int num_replaced = 0;
        for (int i = 0; i < buffer_capacity - 1; i++) {
            if (i >= on_disk.size(0)) {
                break;
            }
            num_replaced++;
            in_buffer[i] = on_disk[i];

            data_ptr_ = (int *)in_buffer.data_ptr();
            buffer_states.emplace_back(vector<int>(data_ptr_, data_ptr_ + in_buffer.size(0)));
        }
        on_disk = on_disk.narrow(0, num_replaced, on_disk.size(0) - num_replaced);
    }

    return buffer_states;
}

vector<vector<std::pair<int, int>>> greedyAssignEdgeBucketsToBuffers(vector<vector<int>> buffer_states, int num_partitions) {
    vector<vector<std::pair<int, int>>> edge_buckets_per_buffer(buffer_states.size());
    torch::Tensor interacted = torch::zeros({num_partitions, num_partitions}, torch::kInt32);
    auto interacted_accessor = interacted.accessor<int32_t, 2>();

    for (int i = 0; i < buffer_states.size(); i++) {
        for (int j = 0; j < buffer_states[i].size(); j++) {
            for (int k = 0; k < buffer_states[i].size(); k++) {
                int32_t src_part = buffer_states[i][j];
                int32_t dst_part = buffer_states[i][k];
                if (interacted_accessor[src_part][dst_part] == 1) {
                    continue;
                }
                interacted_accessor[src_part][dst_part] = 1;
                edge_buckets_per_buffer[i].emplace_back(std::make_pair(src_part, dst_part));
            }
        }
    }

    return edge_buckets_per_buffer;
}

vector<vector<std::pair<int, int>>> randomlyAssignEdgeBucketsToBuffers(vector<vector<int>> buffer_states, int num_partitions) {
    // get edge buckets from buffer states
    Indices all_partitions = torch::arange(num_partitions, torch::kInt32);
    torch::Tensor left_col = all_partitions.repeat_interleave(num_partitions);
    torch::Tensor right_col = all_partitions.repeat({num_partitions});
    torch::Tensor all_buckets = torch::stack({left_col, right_col}, 1);
    auto all_buckets_accessor = all_buckets.accessor<int32_t, 2>();

    int num_buffers = buffer_states.size();
    int buffer_size = buffer_states[0].size();
    int num_buckets = all_buckets.size(0);

    torch::Tensor choices = torch::zeros({num_buckets, num_buffers}, torch::kInt32);
    int32_t *choices_mem = choices.data_ptr<int32_t>();

#pragma omp parallel for
    for (int i = 0; i < num_buffers; i++) {
        for (int j = 0; j < buffer_size; j++) {
            for (int k = j; k < buffer_size; k++) {
                int src_part = buffer_states[i][j];
                int dst_part = buffer_states[i][k];
                *(choices_mem + (src_part * num_partitions + dst_part) * num_buffers + i) = 1;
                *(choices_mem + (dst_part * num_partitions + src_part) * num_buffers + i) = 1;
            }
        }
    }

    torch::Tensor pick = torch::zeros({num_buckets}, torch::kInt32);
    torch::Tensor pick_one_hot = torch::zeros({num_buckets, num_buffers}, torch::kInt32);
    int32_t *pick_mem = pick.data_ptr<int32_t>();
    int32_t *pick_one_hot_mem = pick_one_hot.data_ptr<int32_t>();
    auto pick_accessor = pick.accessor<int32_t, 1>();

    // setup seeds
    unsigned int num_threads = 1;
#ifdef GEGE_OMP
#pragma omp parallel
    {
#pragma omp single
        num_threads = omp_get_num_threads();
    }
#endif
    std::vector<unsigned int> tid_seeds(num_threads);

    for (int i = 0; i < num_threads; i++) {
        tid_seeds[i] = rand();
    }

#pragma omp parallel
    {
#ifdef GEGE_OMP
        unsigned int seed = tid_seeds[omp_get_thread_num()];
#else
        unsigned int seed = tid_seeds[0];
#endif

#pragma omp for
        for (int i = 0; i < num_buckets; i++) {
            torch::Tensor buffer_choices = torch::nonzero(choices[i]);
            buffer_choices = torch::reshape(buffer_choices, {buffer_choices.size(0)});
            int32_t buffer_choice = buffer_choices[rand_r(&seed) % buffer_choices.size(0)].item<int32_t>();

            int32_t src_part = all_buckets_accessor[i][0];
            int32_t dst_part = all_buckets_accessor[i][1];
            *(pick_mem + (src_part * num_partitions + dst_part)) = buffer_choice;
            *(pick_one_hot_mem + (src_part * num_partitions + dst_part) * num_buffers + buffer_choice) = 1;
        }
    }

    torch::Tensor num_edge_buckets_per_buffer = torch::sum(pick_one_hot, 0);

    vector<vector<std::pair<int, int>>> edge_buckets_per_buffer(num_buffers);
    for (int i = 0; i < num_buffers; i++) {
        edge_buckets_per_buffer[i] = vector<std::pair<int, int>>(num_edge_buckets_per_buffer[i].item<int>());
    }

    vector<int> indices(num_buffers, 0);
    for (int i = 0; i < num_buckets; i++) {
        int32_t src_part = all_buckets_accessor[i][0];
        int32_t dst_part = all_buckets_accessor[i][1];
        std::pair<int, int> pair = std::make_pair(src_part, dst_part);

        int32_t buffer_choice = pick_accessor[i];

        edge_buckets_per_buffer[buffer_choice][indices[buffer_choice]] = pair;
        indices[buffer_choice] += 1;
    }

    return edge_buckets_per_buffer;
}

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getTwoLevelBetaOrdering(int num_partitions, int buffer_capacity, int fine_to_coarse_ratio,
                                                                                 int num_cache_partitions, bool randomly_assign_edge_buckets) {
    int coarse_num_partitions = num_partitions / fine_to_coarse_ratio;
    int coarse_buffer_capacity = buffer_capacity / fine_to_coarse_ratio;

    coarse_num_partitions = coarse_num_partitions - num_cache_partitions;
    coarse_buffer_capacity = coarse_buffer_capacity - num_cache_partitions;

    vector<vector<int>> coarse_buffer_states = getBetaOrderingHelper(coarse_num_partitions, coarse_buffer_capacity);

    int cached_fine_partitions = num_cache_partitions * fine_to_coarse_ratio;
    torch::Tensor fine_to_coarse_map = torch::arange(cached_fine_partitions, torch::kInt32);
    fine_to_coarse_map = torch::cat({fine_to_coarse_map, torch::randperm(num_partitions - cached_fine_partitions, torch::kInt32) + cached_fine_partitions});
    int *data_ptr_ = (int *)fine_to_coarse_map.data_ptr();

    for (int i = 0; i < coarse_buffer_states.size(); i++) {
        for (int j = 0; j < coarse_buffer_states[i].size(); j++) {
            coarse_buffer_states[i][j] += num_cache_partitions;
        }
        for (int j = 0; j < num_cache_partitions; j++) {
            coarse_buffer_states[i].emplace_back(j);
        }
    }

    // convert to fine buffer states
    vector<vector<int>> buffer_states;

    for (int i = 0; i < coarse_buffer_states.size(); i++) {
        vector<int> fine_buffer_state(buffer_capacity, 0);
        for (int j = 0; j < coarse_buffer_states[i].size(); j++) {
            int *start = (int *)data_ptr_ + coarse_buffer_states[i][j] * fine_to_coarse_ratio;
            int *end = (int *)data_ptr_ + (coarse_buffer_states[i][j] + 1) * fine_to_coarse_ratio;
            vector<int> fine_partitions = vector<int>(start, end);

            for (int k = j * fine_to_coarse_ratio; k < (j + 1) * fine_to_coarse_ratio; k++) {
                fine_buffer_state[k] = fine_partitions[k - j * fine_to_coarse_ratio];
            }
        }

        buffer_states.emplace_back(fine_buffer_state);
    }

    // assign edge buckets
    vector<vector<std::pair<int, int>>> edge_buckets_per_buffer;
    if (randomly_assign_edge_buckets) {
        edge_buckets_per_buffer = randomlyAssignEdgeBucketsToBuffers(buffer_states, num_partitions);
    } else {
        edge_buckets_per_buffer = greedyAssignEdgeBucketsToBuffers(buffer_states, num_partitions);
    }

    return convertEdgeBucketOrderToTensors(buffer_states, edge_buckets_per_buffer);
}

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getDispersedNodePartitionOrdering(Indices train_nodes, int64_t total_num_nodes, int num_partitions,
                                                                                           int buffer_capacity, int fine_to_coarse_ratio,
                                                                                           int num_cache_partitions) {
    int coarse_num_partitions = num_partitions / fine_to_coarse_ratio;
    int coarse_buffer_capacity = buffer_capacity / fine_to_coarse_ratio;

    coarse_num_partitions = coarse_num_partitions - num_cache_partitions;
    coarse_buffer_capacity = coarse_buffer_capacity - num_cache_partitions;

    // create coarse buffer states
    vector<torch::Tensor> coarse_buffer_states;
    Indices all_coarse_partitions = torch::randperm(coarse_num_partitions, torch::kInt32);
    Indices in_buffer = all_coarse_partitions.narrow(0, 0, coarse_buffer_capacity);
    Indices on_disk = all_coarse_partitions.narrow(0, coarse_buffer_capacity, coarse_num_partitions - coarse_buffer_capacity);
    coarse_buffer_states.emplace_back(in_buffer);

    while (on_disk.size(0) > 0) {
        in_buffer = in_buffer.index_select(0, torch::randperm(in_buffer.size(0), torch::kInt64));
        on_disk = on_disk.index_select(0, torch::randperm(on_disk.size(0), torch::kInt64));

        in_buffer[-1] = on_disk[0];
        coarse_buffer_states.emplace_back(in_buffer);
        on_disk = on_disk.narrow(0, 1, on_disk.size(0) - 1);
    }

    for (int i = 0; i < coarse_buffer_states.size(); i++) {
        coarse_buffer_states[i] =
            torch::cat({coarse_buffer_states[i] + num_cache_partitions, torch::arange(num_cache_partitions, coarse_buffer_states[i].options())});
    }

    // convert to fine buffer states
    torch::Tensor fine_to_coarse_map = torch::randperm(num_partitions, torch::kInt32);
    int *data_ptr_ = (int *)fine_to_coarse_map.data_ptr();

    vector<torch::Tensor> buffer_states;

    for (int i = 0; i < coarse_buffer_states.size(); i++) {
        vector<int> fine_buffer_state(buffer_capacity, 0);
        torch::Tensor coarse_buffer_state = coarse_buffer_states[i];
        auto coarse_buffer_state_accessor = coarse_buffer_state.accessor<int32_t, 1>();

        for (int j = 0; j < coarse_buffer_state.size(0); j++) {
            int *start = (int *)data_ptr_ + coarse_buffer_state_accessor[j] * fine_to_coarse_ratio;
            int *end = (int *)data_ptr_ + (coarse_buffer_state_accessor[j] + 1) * fine_to_coarse_ratio;
            vector<int> fine_partitions = vector<int>(start, end);

            for (int k = j * fine_to_coarse_ratio; k < (j + 1) * fine_to_coarse_ratio; k++) {
                fine_buffer_state[k] = fine_partitions[k - j * fine_to_coarse_ratio];
            }
        }

        buffer_states.emplace_back(torch::from_blob(fine_buffer_state.data(), {(int)fine_buffer_state.size()}, torch::kInt32).clone());
    }

    // randomly assign train nodes to buffers

    int64_t partition_size = ceil((double)total_num_nodes / num_partitions);
    torch::Tensor train_nodes_partition =
        torch::floor(train_nodes.to(torch::kFloat64).div(static_cast<double>(partition_size))).to(torch::kInt32);

    std::vector<std::vector<int>> partition_buffer_states(num_partitions);

    for (int i = 0; i < num_partitions; i++) {
        for (int j = 0; j < buffer_states.size(); j++) {
            bool partition_in_buffer = false;
            auto buffer_state_accessor = buffer_states[j].accessor<int32_t, 1>();

            for (int k = 0; k < buffer_capacity; k++) {
                if (buffer_state_accessor[k] == i) {
                    partition_in_buffer = true;
                    break;
                }
            }
            if (partition_in_buffer) {
                partition_buffer_states[i].emplace_back(j);
            }
        }
    }

    torch::Tensor train_nodes_buffer_choice = torch::zeros_like(train_nodes);
    std::vector<torch::Tensor> train_nodes_per_buffer(buffer_states.size());
    auto train_nodes_partition_accessor = train_nodes_partition.accessor<int32_t, 1>();  // todo

    for (int i = 0; i < train_nodes.size(0); i++) {
        int partition_id = train_nodes_partition_accessor[i];
        int rand_id = rand() % partition_buffer_states[partition_id].size();
        train_nodes_buffer_choice[i] = partition_buffer_states[partition_id][rand_id];
    }

    for (int i = 0; i < buffer_states.size(); i++) {
        train_nodes_per_buffer[i] = train_nodes.masked_select(train_nodes_buffer_choice == i);
    }

    return std::forward_as_tuple(buffer_states, train_nodes_per_buffer);
}

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getSequentialNodePartitionOrdering(Indices train_nodes, int64_t total_num_nodes, int num_partitions,
                                                                                            int buffer_capacity) {
    int64_t partition_size = ceil((double)total_num_nodes / num_partitions);
    torch::Tensor train_nodes_partition =
        torch::floor(train_nodes.to(torch::kFloat64).div(static_cast<double>(partition_size))).to(torch::kInt32);

    int32_t max_train_partition = torch::max(train_nodes_partition).item<int32_t>();
    int32_t num_train_partitions = max_train_partition + 1;
    SPDLOG_INFO("Num Train Partitions: {}", num_train_partitions);

    vector<torch::Tensor> buffer_states;
    Indices in_buffer = torch::arange(num_train_partitions, torch::kInt32);
    Indices on_disk = torch::arange(num_train_partitions, num_partitions, torch::kInt32);
    on_disk = on_disk.index_select(0, torch::randperm(on_disk.size(0), torch::kInt64));
    on_disk = on_disk.narrow(0, 0, buffer_capacity - num_train_partitions);

    buffer_states.emplace_back(torch::cat({in_buffer, on_disk}));

    std::vector<torch::Tensor> train_nodes_per_buffer;
    train_nodes_per_buffer.emplace_back(train_nodes.clone());

    return std::forward_as_tuple(buffer_states, train_nodes_per_buffer);
}

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getCustomNodePartitionOrdering() {
    SPDLOG_ERROR("Not implemented");
    std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> ret;
    return ret;
}

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getAccessAwareCustomEdgeBucketOrdering(int num_partitions, int buffer_capacity, int active_devices) {
    SPDLOG_INFO("Generating access-aware CUSTOM Ordering");
    return generate_access_aware_states(num_partitions, buffer_capacity, active_devices);
}

std::vector<int64_t> getDisjointBufferStatePermutation(const vector<torch::Tensor>& buffer_states, int active_devices) {
    auto groups = build_disjoint_groups(buffer_states, active_devices);
    if (groups.empty()) {
        std::vector<int64_t> identity(buffer_states.size());
        std::iota(identity.begin(), identity.end(), 0);
        return identity;
    }

    std::vector<int64_t> permutation;
    permutation.reserve(buffer_states.size());
    for (auto &group : groups) {
        for (auto state_idx : group) {
            permutation.emplace_back(state_idx);
        }
    }

    if (permutation.size() != buffer_states.size()) {
        std::vector<int64_t> identity(buffer_states.size());
        std::iota(identity.begin(), identity.end(), 0);
        return identity;
    }

    return permutation;
}

std::vector<int64_t> getAccessAwareDisjointBufferStatePermutation(const vector<torch::Tensor>& buffer_states,
                                                                  const vector<torch::Tensor>& edge_buckets_per_buffer,
                                                                  int active_devices) {
    if (active_devices <= 1 || buffer_states.size() <= 1 || edge_buckets_per_buffer.size() != buffer_states.size()) {
        return getDisjointBufferStatePermutation(buffer_states, active_devices);
    }

    std::vector<std::vector<int64_t>> state_partitions;
    state_partitions.reserve(buffer_states.size());
    for (auto &state : buffer_states) {
        state_partitions.emplace_back(tensor_to_partitions(state));
    }

    std::vector<std::vector<bool>> compatible(buffer_states.size(), std::vector<bool>(buffer_states.size(), false));
    for (std::size_t i = 0; i < buffer_states.size(); i++) {
        compatible[i][i] = true;
        for (std::size_t j = i + 1; j < buffer_states.size(); j++) {
            bool disjoint = states_disjoint(state_partitions[i], state_partitions[j]);
            compatible[i][j] = disjoint;
            compatible[j][i] = disjoint;
        }
    }

    auto summaries = build_state_access_summaries(buffer_states, edge_buckets_per_buffer);

    std::vector<int64_t> remaining(buffer_states.size());
    std::iota(remaining.begin(), remaining.end(), 0);
    std::vector<int64_t> permutation;
    permutation.reserve(buffer_states.size());

    std::vector<int64_t> previous_group;
    while (!remaining.empty()) {
        int target_group_size = std::min<int>(active_devices, remaining.size());
        GroupSearchResult best_group;
        std::vector<int64_t> current_group;
        current_group.reserve(target_group_size);
        search_best_disjoint_group(compatible, remaining, previous_group, summaries, target_group_size, 0, current_group, best_group);

        if (best_group.chosen_states.empty()) {
            return getDisjointBufferStatePermutation(buffer_states, active_devices);
        }

        for (auto state_idx : best_group.ordered_group) {
            permutation.emplace_back(state_idx);
        }

        previous_group = best_group.ordered_group;
        std::vector<int64_t> next_remaining;
        next_remaining.reserve(remaining.size() - best_group.chosen_states.size());
        for (auto state_idx : remaining) {
            if (std::find(best_group.chosen_states.begin(), best_group.chosen_states.end(), state_idx) == best_group.chosen_states.end()) {
                next_remaining.emplace_back(state_idx);
            }
        }
        remaining = std::move(next_remaining);
    }

    if (permutation.size() != buffer_states.size()) {
        std::vector<int64_t> identity(buffer_states.size());
        std::iota(identity.begin(), identity.end(), 0);
        return identity;
    }

    return permutation;
}

int32_t pow(int32_t a, int32_t x)
{
    int32_t ans = 1, temp = a;
    while(x) {
        if (x & 1) {
            ans = ans * temp;
        }
        temp *= temp;
        x >>= 1;
    }
    return ans;
}

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getCustomEdgeBucketOrdering(int num_partitions, int buffer_capacity, bool randomly_assign_edge_buckets)
{
    assert(buffer_capacity == 4);
    int32_t sub_chunk_per_perm = num_partitions / buffer_capacity;
    int32_t log2l = 0;

    while(pow(2, log2l) < num_partitions) {
        log2l += 1;
    }

    assert(pow(2, log2l) == num_partitions);

    std::vector<std::vector<std::vector<int>>> offset_supergroup = {
        {{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}},
        {{0, 1, 2, 3}, {1, 0, 3, 2}, {2, 3, 0, 1}, {3, 2, 1, 0}},
        {{0, 2, 3, 1}, {1, 3, 2, 0}, {2, 0, 1, 3}, {3, 1, 0, 2}},
        {{0, 3, 1, 2}, {1, 2, 0, 3}, {2, 1, 3, 0}, {3, 0, 2, 1}},
    };
    std::vector<std::vector<std::vector<int>>> p = {{{0, 1, 2, 3}}};

    for (int log4l_pre = 1; log4l_pre < log2l / 2; log4l_pre ++) {
        auto p_pre = p;
        p = std::vector<std::vector<std::vector<int>>>();
        for (auto& s : p_pre) {
            std::vector<std::vector<int>> s_cur;
            for (int offset = 0; offset < pow(4, log4l_pre + 1); offset += pow(4, log4l_pre)) {
                for (auto& g : s) {
                    std::vector<int> g_cur;
                    for(auto& x : g) {
                        g_cur.emplace_back(x + offset);
                    }
                    s_cur.emplace_back(g_cur);
                }
            }
            p.emplace_back(s_cur);
        }
        int32_t len = p_pre.size();
        for (int i = len - pow(4, log4l_pre - 1); i < len; i ++) {
            auto s = p_pre[i];
            for (auto& offset_s : offset_supergroup) {
                std::vector<std::vector<int>> s_cur;
                for (auto& g : s) {
                    for(auto& offset_g : offset_s){
                        std::vector<int> g_cur;
                        for (int j = 0; j < 4; j ++) {
                            g_cur.emplace_back(g[j] * 4 + offset_g[j]);
                        }
                        s_cur.emplace_back(g_cur);
                    }
                }
                p.emplace_back(s_cur);
            }
        }
    }
    std::vector<std::vector<std::vector<int>>> pairing_chunks = {
        {{0, 2}, {1, 3}},
        {{0, 3}, {1, 2}}
    };

    if (log2l % 2 == 1) {
        int32_t len_chunk = sub_chunk_per_perm;
        auto p_pre = p;
        p = std::vector<std::vector<std::vector<int>>>();
        
        for (auto& s: p_pre) {
            std::vector<std::vector<int>> s_cur;
            for(int i = 0; i < pow(2, log2l); i += pow(2, log2l - 1)) {
                for (auto& g : s) {
                    std::vector<int> g_cur;
                    for (auto& x : g) {
                        g_cur.emplace_back(x + i);
                    }
                    s_cur.emplace_back(g_cur);
                }
            }
            p.emplace_back(s_cur);
        }

        int32_t len = p_pre.size();
        for (int i = len - pow(2, log2l - 3); i < len; i ++) {
            std::vector<std::vector<int>> s = p_pre[i];
            for (auto& pairing_s : pairing_chunks) {
                std::vector<std::vector<int>> s_cur;
                for (auto& chunk_index : pairing_s) {
                    for (auto& g : s) {
                        std::vector<int> g_cur;
                        for (auto& x : g) {
                            g_cur.emplace_back(chunk_index[x / len_chunk] * len_chunk + x % len_chunk);
                        }
                        s_cur.emplace_back(g_cur);
                    }
                }
                p.emplace_back(s_cur);
            }

        }
    }
    std::vector<std::vector<int>> buffer_states;
    Indices all_partitions_map = torch::randperm(num_partitions, torch::kInt32);
    for (auto& p1 : p) {
        for(auto& p2 : p1) {
            buffer_states.emplace_back(p2);
        } 
    }
    for(int i = 0; i < buffer_states.size(); i ++){
        for(int j = 0; j < buffer_states[i].size(); j ++) {
            // std::cout << buffer_states[i][j] << " ";
            buffer_states[i][j] = all_partitions_map[buffer_states[i][j]].item<int>();
        }
    }

    Indices all_buffer_map = torch::randperm(buffer_states.size(), torch::kInt32);
    std::vector<std::vector<int>> shuffle_buffer_states;
    for (int i = 0; i < buffer_states.size(); i ++) {
        shuffle_buffer_states.push_back(buffer_states[all_buffer_map[i].item<int>()]);
    }
    buffer_states = shuffle_buffer_states;

    std::vector<std::vector<std::pair<int, int>>> edge_buckets_per_buffer;
    if (randomly_assign_edge_buckets) {
        edge_buckets_per_buffer = randomlyAssignEdgeBucketsToBuffers(buffer_states, num_partitions);
    } else {
        edge_buckets_per_buffer = greedyAssignEdgeBucketsToBuffers(buffer_states, num_partitions);
    }
    // for(auto const& edge_buckets : edge_buckets_per_buffer) {
        // std::cout << edge_buckets.size() << ": ";
        // for(auto const& bucket : edge_buckets) {
        //     std::cout << "(" << bucket.first << "," << bucket.second << ") "<< " ";
        // }
        // std::cout << std::endl;
    // }
    return convertEdgeBucketOrderToTensors(buffer_states, edge_buckets_per_buffer);

}

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getOptimizedCustomEdgeBucketOrdering(int num_partitions,
                                                                                               int buffer_capacity,
                                                                                               int active_devices,
                                                                                               int batch_size,
                                                                                               const vector<int64_t> &edge_bucket_sizes) {
    if (!optimized_custom_schedule_enabled()) {
        SPDLOG_INFO("GEGE_OPTIMIZED_CUSTOM_SCHEDULE disabled; falling back to standard CUSTOM ordering");
        return getCustomEdgeBucketOrdering(num_partitions, buffer_capacity, false);
    }

    if (buffer_capacity != 4 || active_devices != 4) {
        SPDLOG_INFO("Optimized CUSTOM ordering currently supports buffer_capacity=4 and active_devices=4; falling back to standard CUSTOM ordering");
        return getCustomEdgeBucketOrdering(num_partitions, buffer_capacity, false);
    }

    if (edge_bucket_sizes.size() != static_cast<size_t>(num_partitions * num_partitions)) {
        SPDLOG_WARN("Optimized CUSTOM ordering expected {} edge bucket sizes, found {}; falling back to standard CUSTOM ordering",
                    num_partitions * num_partitions, edge_bucket_sizes.size());
        return getCustomEdgeBucketOrdering(num_partitions, buffer_capacity, false);
    }

    return build_optimized_custom_edge_bucket_ordering(num_partitions, buffer_capacity, active_devices, batch_size, edge_bucket_sizes);
}
