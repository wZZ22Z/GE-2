#!/usr/bin/env python3

import argparse
from itertools import permutations
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class StateMetrics:
    partitions: Tuple[int, ...]
    weight: int
    batches: int
    bucket_count: int


@dataclass(frozen=True)
class ScheduleScore:
    worst_round_spread: int
    worst_batch_spread: int
    worst_state_weight: int
    total_round_spread: int
    continuity_hotness: int
    continuity_new_partitions: int
    total_abs_deviation: int

    def as_tuple(self) -> Tuple[int, int, int, int, int, int, int]:
        return (
            self.worst_round_spread,
            self.worst_batch_spread,
            self.worst_state_weight,
            self.total_round_spread,
            self.continuity_hotness,
            self.continuity_new_partitions,
            self.total_abs_deviation,
        )


@dataclass(frozen=True)
class EvaluatedSchedule:
    slot_to_partition: Tuple[int, ...]
    states: Tuple[StateMetrics, ...]
    rounds: Tuple[Tuple[StateMetrics, ...], ...]
    lane_rounds: Tuple[Tuple[StateMetrics, ...], ...]
    score: ScheduleScore


STATE_LINE_RE = re.compile(
    r"\[perf\]\[epoch (?P<epoch>\d+)]\[gpu (?P<gpu>\d+)]\[state (?P<round>\d+)] "
    r"phase=(?P<phase>\w+) state_idx=(?P<state_idx>\d+) resident_partitions=(?P<parts>[0-9,]+) "
    r"active_buckets=(?P<buckets>\d+) active_edges=(?P<edges>\d+) batches=(?P<batches>\d+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize the CUSTOM 4-GPU / 4-partition state schedule for a partitioned dataset by "
            "replacing the current random partition relabeling with a dataset-aware assignment."
        )
    )
    parser.add_argument("dataset_dir", type=Path, help="Dataset directory, e.g. datasets/twitter_16p")
    parser.add_argument("--num-partitions", type=int, default=16)
    parser.add_argument("--buffer-capacity", type=int, default=4)
    parser.add_argument("--active-devices", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument("--restarts", type=int, default=256)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--init-mode",
        choices=("random", "lite", "hybrid"),
        default="hybrid",
        help="Initial assignment strategy: random only, Lite-inspired constructive only, or both.",
    )
    parser.add_argument(
        "--compare-log",
        type=Path,
        default=None,
        help="Optional training log to summarize the currently observed schedules for comparison.",
    )
    return parser.parse_args()


def int_pow(a: int, x: int) -> int:
    ans = 1
    temp = a
    while x:
        if x & 1:
            ans *= temp
        temp *= temp
        x >>= 1
    return ans


def generate_custom_template(num_partitions: int, buffer_capacity: int) -> List[List[int]]:
    if buffer_capacity != 4:
        raise ValueError("This standalone optimizer currently mirrors the CUSTOM template only for buffer_capacity=4")

    log2l = 0
    while int_pow(2, log2l) < num_partitions:
        log2l += 1
    if int_pow(2, log2l) != num_partitions:
        raise ValueError("num_partitions must be a power of two")

    offset_supergroup = [
        [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]],
        [[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]],
        [[0, 2, 3, 1], [1, 3, 2, 0], [2, 0, 1, 3], [3, 1, 0, 2]],
        [[0, 3, 1, 2], [1, 2, 0, 3], [2, 1, 3, 0], [3, 0, 2, 1]],
    ]
    p: List[List[List[int]]] = [[[0, 1, 2, 3]]]

    for log4l_pre in range(1, log2l // 2):
        p_pre = p
        p = []
        for s in p_pre:
            s_cur: List[List[int]] = []
            step = int_pow(4, log4l_pre)
            for offset in range(0, int_pow(4, log4l_pre + 1), step):
                for g in s:
                    s_cur.append([x + offset for x in g])
            p.append(s_cur)

        length = len(p_pre)
        start = length - int_pow(4, log4l_pre - 1)
        for i in range(start, length):
            s = p_pre[i]
            for offset_s in offset_supergroup:
                s_cur = []
                for g in s:
                    for offset_g in offset_s:
                        s_cur.append([g[j] * 4 + offset_g[j] for j in range(4)])
                p.append(s_cur)

    pairing_chunks = [
        [[0, 2], [1, 3]],
        [[0, 3], [1, 2]],
    ]

    if log2l % 2 == 1:
        len_chunk = num_partitions // buffer_capacity
        p_pre = p
        p = []

        for s in p_pre:
            s_cur: List[List[int]] = []
            for offset in range(0, int_pow(2, log2l), int_pow(2, log2l - 1)):
                for g in s:
                    s_cur.append([x + offset for x in g])
            p.append(s_cur)

        length = len(p_pre)
        start = length - int_pow(2, log2l - 3)
        for i in range(start, length):
            s = p_pre[i]
            for pairing_s in pairing_chunks:
                s_cur = []
                for chunk_index in pairing_s:
                    for g in s:
                        g_cur = []
                        for x in g:
                            g_cur.append(chunk_index[x // len_chunk] * len_chunk + x % len_chunk)
                        s_cur.append(g_cur)
                p.append(s_cur)

    return [g for supergroup in p for g in supergroup]


def greedy_assign_edge_buckets_to_states(states: Sequence[Sequence[int]], num_partitions: int) -> List[List[Tuple[int, int]]]:
    interacted = [[False] * num_partitions for _ in range(num_partitions)]
    edge_buckets_per_state: List[List[Tuple[int, int]]] = []

    for state in states:
        state_buckets: List[Tuple[int, int]] = []
        for src_part in state:
            for dst_part in state:
                if interacted[src_part][dst_part]:
                    continue
                interacted[src_part][dst_part] = True
                state_buckets.append((src_part, dst_part))
        edge_buckets_per_state.append(state_buckets)

    return edge_buckets_per_state


def read_bucket_matrix(dataset_dir: Path, num_partitions: int) -> List[List[int]]:
    offsets_path = dataset_dir / "edges" / "train_partition_offsets.txt"
    values = [int(line.strip()) for line in offsets_path.read_text().splitlines() if line.strip()]
    expected = num_partitions * num_partitions
    if len(values) != expected:
        raise ValueError(f"Expected {expected} bucket sizes in {offsets_path}, found {len(values)}")

    return [values[i * num_partitions : (i + 1) * num_partitions] for i in range(num_partitions)]


def partition_hotness(bucket_matrix: Sequence[Sequence[int]]) -> List[int]:
    num_partitions = len(bucket_matrix)
    hotness: List[int] = []
    for partition in range(num_partitions):
        outgoing = sum(bucket_matrix[partition][other] for other in range(num_partitions))
        incoming = sum(bucket_matrix[other][partition] for other in range(num_partitions))
        hotness.append(outgoing + incoming - bucket_matrix[partition][partition])
    return hotness


def build_slot_pair_owners(template_states: Sequence[Sequence[int]], num_slots: int) -> List[List[int]]:
    owners = [[-1] * num_slots for _ in range(num_slots)]
    for state_idx, state in enumerate(template_states):
        for src_slot in state:
            for dst_slot in state:
                if owners[src_slot][dst_slot] == -1:
                    owners[src_slot][dst_slot] = state_idx

    for src_slot in range(num_slots):
        for dst_slot in range(num_slots):
            if owners[src_slot][dst_slot] == -1:
                raise RuntimeError(f"No owner state found for slot pair ({src_slot}, {dst_slot})")
    return owners


def lite_initial_assignment(
    template_states: Sequence[Sequence[int]],
    bucket_matrix: Sequence[Sequence[int]],
    hotness: Sequence[int],
    num_partitions: int,
    active_devices: int,
) -> List[int]:
    slot_pair_owners = build_slot_pair_owners(template_states, num_partitions)
    state_weights = [0] * len(template_states)
    slot_to_partition = [-1] * num_partitions
    assigned_slots: List[int] = []
    target_state_weight = sum(sum(row) for row in bucket_matrix) / len(template_states)

    sorted_partitions = sorted(range(num_partitions), key=lambda partition: (-hotness[partition], partition))

    for partition in sorted_partitions:
        best_slot: int | None = None
        best_key: Tuple[float, int, float, float, int] | None = None

        for slot in range(num_partitions):
            if slot_to_partition[slot] != -1:
                continue

            candidate_weights = state_weights.copy()
            diagonal_owner = slot_pair_owners[slot][slot]
            candidate_weights[diagonal_owner] += bucket_matrix[partition][partition]

            for other_slot in assigned_slots:
                other_partition = slot_to_partition[other_slot]
                if other_partition == -1:
                    continue
                forward_owner = slot_pair_owners[slot][other_slot]
                reverse_owner = slot_pair_owners[other_slot][slot]
                candidate_weights[forward_owner] += bucket_matrix[partition][other_partition]
                candidate_weights[reverse_owner] += bucket_matrix[other_partition][partition]

            round_spreads = []
            round_maxima = []
            total_over_target = 0.0
            total_abs_deviation = 0.0
            for round_start in range(0, len(candidate_weights), active_devices):
                round_weights = candidate_weights[round_start : round_start + active_devices]
                round_spreads.append(max(round_weights) - min(round_weights))
                round_maxima.append(max(round_weights))
            for weight in candidate_weights:
                total_over_target += max(weight - target_state_weight, 0.0)
                total_abs_deviation += abs(weight - target_state_weight)

            candidate_key = (
                max(round_maxima),
                max(round_spreads),
                total_over_target,
                total_abs_deviation,
                slot,
            )
            if best_key is None or candidate_key < best_key:
                best_key = candidate_key
                best_slot = slot

        if best_slot is None:
            raise RuntimeError("Failed to construct Lite-inspired initial assignment")

        slot_to_partition[best_slot] = partition
        diagonal_owner = slot_pair_owners[best_slot][best_slot]
        state_weights[diagonal_owner] += bucket_matrix[partition][partition]
        for other_slot in assigned_slots:
            other_partition = slot_to_partition[other_slot]
            if other_partition == -1:
                continue
            forward_owner = slot_pair_owners[best_slot][other_slot]
            reverse_owner = slot_pair_owners[other_slot][best_slot]
            state_weights[forward_owner] += bucket_matrix[partition][other_partition]
            state_weights[reverse_owner] += bucket_matrix[other_partition][partition]
        assigned_slots.append(best_slot)

    return slot_to_partition


def state_transition_cost(
    previous_state: StateMetrics, next_state: StateMetrics, hotness: Sequence[int]
) -> Tuple[int, int]:
    previous_partitions = set(previous_state.partitions)
    new_partitions = [partition for partition in next_state.partitions if partition not in previous_partitions]
    return (sum(hotness[partition] for partition in new_partitions), len(new_partitions))


def optimize_lane_assignment(
    rounds: Sequence[Tuple[StateMetrics, ...]], hotness: Sequence[int]
) -> Tuple[Tuple[Tuple[StateMetrics, ...], ...], int, int]:
    if not rounds:
        return tuple(), 0, 0

    all_permutations = [tuple(permutations(range(len(round_)))) for round_ in rounds]
    dp: List[dict[Tuple[int, ...], Tuple[int, int]]] = [{} for _ in rounds]
    backpointers: List[dict[Tuple[int, ...], Tuple[int, ...] | None]] = [{} for _ in rounds]

    for permutation in all_permutations[0]:
        dp[0][permutation] = (0, 0)
        backpointers[0][permutation] = None

    for round_idx in range(1, len(rounds)):
        for permutation in all_permutations[round_idx]:
            ordered_round = tuple(rounds[round_idx][index] for index in permutation)
            best_cost: Tuple[int, int] | None = None
            best_previous: Tuple[int, ...] | None = None

            for previous_permutation, previous_cost in dp[round_idx - 1].items():
                ordered_previous = tuple(rounds[round_idx - 1][index] for index in previous_permutation)
                transition_hotness = 0
                transition_new_partitions = 0
                for lane_idx in range(len(ordered_round)):
                    lane_hotness, lane_new_partitions = state_transition_cost(
                        ordered_previous[lane_idx], ordered_round[lane_idx], hotness
                    )
                    transition_hotness += lane_hotness
                    transition_new_partitions += lane_new_partitions

                candidate_cost = (
                    previous_cost[0] + transition_hotness,
                    previous_cost[1] + transition_new_partitions,
                )
                if best_cost is None or candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_previous = previous_permutation

            if best_cost is None or best_previous is None:
                raise RuntimeError("Failed to compute a valid lane assignment")
            dp[round_idx][permutation] = best_cost
            backpointers[round_idx][permutation] = best_previous

    final_permutation = min(dp[-1], key=lambda permutation: dp[-1][permutation])
    continuity_hotness, continuity_new_partitions = dp[-1][final_permutation]

    chosen_permutations: List[Tuple[int, ...] | None] = [None] * len(rounds)
    chosen_permutations[-1] = final_permutation
    for round_idx in range(len(rounds) - 1, 0, -1):
        chosen_permutations[round_idx - 1] = backpointers[round_idx][chosen_permutations[round_idx]]

    ordered_rounds = tuple(
        tuple(rounds[round_idx][index] for index in chosen_permutations[round_idx])
        for round_idx in range(len(rounds))
    )
    return ordered_rounds, continuity_hotness, continuity_new_partitions


def summarize_schedule(
    template_states: Sequence[Sequence[int]],
    slot_to_partition: Sequence[int],
    bucket_matrix: Sequence[Sequence[int]],
    hotness: Sequence[int],
    num_partitions: int,
    active_devices: int,
    batch_size: int,
) -> EvaluatedSchedule:
    mapped_states = [tuple(slot_to_partition[slot] for slot in state) for state in template_states]
    edge_buckets = greedy_assign_edge_buckets_to_states(mapped_states, num_partitions)

    state_metrics: List[StateMetrics] = []
    for partitions, buckets in zip(mapped_states, edge_buckets):
        weight = sum(bucket_matrix[src][dst] for src, dst in buckets)
        batches = (weight + batch_size - 1) // batch_size
        state_metrics.append(
            StateMetrics(partitions=tuple(partitions), weight=weight, batches=batches, bucket_count=len(buckets))
        )

    rounds = [tuple(state_metrics[i : i + active_devices]) for i in range(0, len(state_metrics), active_devices)]
    lane_rounds, continuity_hotness, continuity_new_partitions = optimize_lane_assignment(rounds, hotness)
    round_spreads = [max(state.weight for state in round_) - min(state.weight for state in round_) for round_ in rounds]
    batch_spreads = [max(state.batches for state in round_) - min(state.batches for state in round_) for round_ in rounds]
    total_abs_deviation = 0
    worst_state_weight = 0
    for round_ in rounds:
        mean_weight = sum(state.weight for state in round_) / len(round_)
        total_abs_deviation += sum(abs(state.weight - mean_weight) for state in round_)
        worst_state_weight = max(worst_state_weight, max(state.weight for state in round_))

    score = ScheduleScore(
        worst_round_spread=max(round_spreads),
        worst_batch_spread=max(batch_spreads),
        worst_state_weight=worst_state_weight,
        total_round_spread=sum(round_spreads),
        continuity_hotness=continuity_hotness,
        continuity_new_partitions=continuity_new_partitions,
        total_abs_deviation=int(total_abs_deviation),
    )

    return EvaluatedSchedule(
        slot_to_partition=tuple(slot_to_partition),
        states=tuple(state_metrics),
        rounds=tuple(rounds),
        lane_rounds=lane_rounds,
        score=score,
    )


def score_better(lhs: ScheduleScore, rhs: ScheduleScore) -> bool:
    return lhs.as_tuple() < rhs.as_tuple()


def steepest_descent(
    template_states: Sequence[Sequence[int]],
    initial_assignment: Sequence[int],
    bucket_matrix: Sequence[Sequence[int]],
    hotness: Sequence[int],
    num_partitions: int,
    active_devices: int,
    batch_size: int,
) -> EvaluatedSchedule:
    assignment = list(initial_assignment)
    current = summarize_schedule(
        template_states, assignment, bucket_matrix, hotness, num_partitions, active_devices, batch_size
    )

    improved = True
    while improved:
        improved = False
        best_candidate = current
        best_swap: Tuple[int, int] | None = None

        for i in range(len(assignment)):
            for j in range(i + 1, len(assignment)):
                candidate_assignment = assignment.copy()
                candidate_assignment[i], candidate_assignment[j] = candidate_assignment[j], candidate_assignment[i]
                candidate = summarize_schedule(
                    template_states,
                    candidate_assignment,
                    bucket_matrix,
                    hotness,
                    num_partitions,
                    active_devices,
                    batch_size,
                )
                if score_better(candidate.score, best_candidate.score):
                    best_candidate = candidate
                    best_swap = (i, j)

        if best_swap is not None:
            i, j = best_swap
            assignment[i], assignment[j] = assignment[j], assignment[i]
            current = best_candidate
            improved = True

    return current


def generate_initial_assignments(
    template_states: Sequence[Sequence[int]],
    bucket_matrix: Sequence[Sequence[int]],
    hotness: Sequence[int],
    num_partitions: int,
    active_devices: int,
    restarts: int,
    seed: int,
    init_mode: str,
) -> Iterable[List[int]]:
    rng = random.Random(seed)
    identity = list(range(num_partitions))

    if init_mode in ("lite", "hybrid"):
        yield lite_initial_assignment(template_states, bucket_matrix, hotness, num_partitions, active_devices)

    if init_mode in ("random", "hybrid"):
        yield identity

        for _ in range(max(restarts - 1, 0)):
            candidate = identity.copy()
            rng.shuffle(candidate)
            yield candidate


def optimize_schedule(
    template_states: Sequence[Sequence[int]],
    bucket_matrix: Sequence[Sequence[int]],
    hotness: Sequence[int],
    num_partitions: int,
    active_devices: int,
    batch_size: int,
    restarts: int,
    seed: int,
    init_mode: str,
) -> EvaluatedSchedule:
    best: EvaluatedSchedule | None = None

    for initial in generate_initial_assignments(
        template_states, bucket_matrix, hotness, num_partitions, active_devices, restarts, seed, init_mode
    ):
        candidate = steepest_descent(
            template_states, initial, bucket_matrix, hotness, num_partitions, active_devices, batch_size
        )
        if best is None or score_better(candidate.score, best.score):
            best = candidate

    if best is None:
        raise RuntimeError("No schedule candidates were generated")
    return best


def parse_log_rounds(log_path: Path) -> dict[int, list[list[StateMetrics]]]:
    per_epoch: dict[int, dict[int, list[StateMetrics]]] = {}
    for line in log_path.read_text().splitlines():
        match = STATE_LINE_RE.search(line)
        if not match:
            continue
        epoch = int(match.group("epoch"))
        round_idx = int(match.group("round"))
        parts = tuple(int(x) for x in match.group("parts").split(","))
        metrics = StateMetrics(
            partitions=parts,
            weight=int(match.group("edges")),
            batches=int(match.group("batches")),
            bucket_count=int(match.group("buckets")),
        )
        per_epoch.setdefault(epoch, {}).setdefault(round_idx, []).append(metrics)

    result: dict[int, list[list[StateMetrics]]] = {}
    for epoch, rounds in sorted(per_epoch.items()):
        ordered_rounds: list[list[StateMetrics]] = []
        for round_idx in sorted(rounds):
            ordered_rounds.append(sorted(rounds[round_idx], key=lambda state: state.partitions))
        result[epoch] = ordered_rounds
    return result


def format_state(state: StateMetrics) -> str:
    parts = ",".join(str(x) for x in state.partitions)
    return f"{{{parts}}} {state.weight / 1_000_000:.3f}M edges, {state.batches} batches, {state.bucket_count} buckets"


def print_schedule(title: str, schedule: EvaluatedSchedule) -> None:
    print(title)
    print(f"slot_to_partition: {list(schedule.slot_to_partition)}")
    print(
        "score: worst_round_spread={:.3f}M worst_batch_spread={} worst_state_weight={:.3f}M total_round_spread={:.3f}M continuity_hotness={:.3f}M continuity_new_partitions={}".format(
            schedule.score.worst_round_spread / 1_000_000,
            schedule.score.worst_batch_spread,
            schedule.score.worst_state_weight / 1_000_000,
            schedule.score.total_round_spread / 1_000_000,
            schedule.score.continuity_hotness / 1_000_000,
            schedule.score.continuity_new_partitions,
        )
    )
    for round_idx, round_ in enumerate(schedule.rounds, start=1):
        round_spread = max(state.weight for state in round_) - min(state.weight for state in round_)
        batch_spread = max(state.batches for state in round_) - min(state.batches for state in round_)
        print(f"Round {round_idx}: spread={round_spread / 1_000_000:.3f}M, batch_spread={batch_spread}")
        for lane_idx, state in enumerate(schedule.lane_rounds[round_idx - 1]):
            print(f"  GPU{lane_idx}: {format_state(state)}")
    print()


def print_observed_log_summary(log_path: Path) -> None:
    observed = parse_log_rounds(log_path)
    if not observed:
        print(f"No state lines found in {log_path}")
        return

    print(f"Observed schedules from {log_path}")
    for epoch, rounds in observed.items():
        worst_spread = 0
        worst_batch_spread = 0
        total_spread = 0
        for round_ in rounds:
            round_spread = max(state.weight for state in round_) - min(state.weight for state in round_)
            batch_spread = max(state.batches for state in round_) - min(state.batches for state in round_)
            worst_spread = max(worst_spread, round_spread)
            worst_batch_spread = max(worst_batch_spread, batch_spread)
            total_spread += round_spread
        print(
            "Epoch {}: worst_round_spread={:.3f}M worst_batch_spread={} total_round_spread={:.3f}M".format(
                epoch, worst_spread / 1_000_000, worst_batch_spread, total_spread / 1_000_000
            )
        )
    print()


def main() -> None:
    args = parse_args()
    bucket_matrix = read_bucket_matrix(args.dataset_dir, args.num_partitions)
    hotness = partition_hotness(bucket_matrix)
    template_states = generate_custom_template(args.num_partitions, args.buffer_capacity)

    print(
        f"Loaded {args.num_partitions}x{args.num_partitions} bucket matrix from "
        f"{args.dataset_dir / 'edges' / 'train_partition_offsets.txt'}"
    )
    print(f"Template states: {len(template_states)} ({len(template_states) // args.active_devices} rounds x {args.active_devices} GPUs)")
    print()

    optimized = optimize_schedule(
        template_states=template_states,
        bucket_matrix=bucket_matrix,
        hotness=hotness,
        num_partitions=args.num_partitions,
        active_devices=args.active_devices,
        batch_size=args.batch_size,
        restarts=args.restarts,
        seed=args.seed,
        init_mode=args.init_mode,
    )
    print_schedule("Optimized schedule", optimized)

    if args.compare_log is not None:
        print_observed_log_summary(args.compare_log)


if __name__ == "__main__":
    main()
