# Custom Schedule Optimizer Notes

This note records the standalone schedule optimizer used to analyze `CUSTOM` multi-GPU partition-buffer scheduling, along with the latest optimized schedules for `twitter_16p` and Freebase86M 16-way stratification.

Script:
- [optimize_custom_schedule.py](/home/smansou2/newCode/ge2/dandelion-dev/scripts/optimize_custom_schedule.py)

## Scope

The current optimizer mirrors the existing runtime structure rather than inventing a new scheduling family:

- `num_partitions = 16`
- `buffer_capacity = 4`
- `active_devices = 4`
- `20` states total
- `5` rounds
- `4` states per round

It is currently specialized to the existing `CUSTOM` template with `buffer_capacity=4`.

## What The Runtime Does Today

The current runtime behavior is:

1. build the abstract `CUSTOM` state template
2. randomly relabel real partitions onto template slots
3. greedily assign ordered edge buckets to the first state that covers them
4. group states into disjoint 4-state rounds

So the combinatorial template is structured, but the partition placement is random.

## Optimizer Objective

The standalone optimizer keeps the template fixed and replaces the random partition relabeling with a dataset-aware search.

For a state `S`, the exact state weight is:

\[
W(S)=\sum_{(i,j)\in E(S)} A[i][j]
\]

where:
- `A[i][j]` is the ordered training bucket size for partition pair `(i,j)`
- `E(S)` is the set of ordered buckets greedily assigned to that state

This counts both `(i,j)` and `(j,i)` because the runtime uses ordered buckets.

Batch proxy:

\[
B(S)=\left\lceil \frac{W(S)}{\text{batch\_size}} \right\rceil
\]

The primary schedule score is lexicographic:

1. worst round edge spread
2. worst round batch spread
3. worst state weight
4. total round spread

with:

\[
\text{round\_spread}=\max W(S)-\min W(S)
\]

computed over the 4 states in a round.

## GPU-Aware Continuity

The script also adds a GPU-lane continuity objective as a secondary tie-break.

For each partition `p`, define hotness:

\[
H(p)=\sum_q A[p][q]+\sum_q A[q][p]-A[p][p]
\]

For a same-lane transition from `S_prev` to `S_next`:

\[
T(S_{prev}, S_{next})=
\left(
\sum_{p \in P(S_{next}) \setminus P(S_{prev})} H(p),
\left|P(S_{next}) \setminus P(S_{prev})\right|
\right)
\]

The optimizer does **not** change round composition for continuity. It only picks the best GPU-lane ordering of the 4 states in each round.

Because there are only `4! = 24` possible lane orders per round, the best lane assignment is solved exactly with dynamic programming across rounds.

The full score tuple is:

1. `worst_round_spread`
2. `worst_batch_spread`
3. `worst_state_weight`
4. `total_round_spread`
5. `continuity_hotness`
6. `continuity_new_partitions`
7. `total_abs_deviation`

So balance remains primary and continuity is only a later tie-break.

## Search Method

The search over partition assignments currently uses:

- multiple random restarts
- pairwise partition swaps
- steepest-descent local search

This is heuristic, not a proof of global optimality. It has been good enough to produce tightly balanced schedules on the tested datasets.

## Commands Used

Twitter:

```bash
python3 -u /home/smansou2/newCode/ge2/dandelion-dev/scripts/optimize_custom_schedule.py \
  /home/smansou2/newCode/ge2/dandelion-dev/datasets/twitter_16p \
  --restarts 4 \
  --seed 12345
```

Freebase86M 16-way stratification:

```bash
python3 -u /home/smansou2/newCode/ge2/dandelion-dev/scripts/optimize_custom_schedule.py \
  /tmp/freebase_strat16_for_gege \
  --restarts 4 \
  --seed 12345
```

The Freebase matrix in `/tmp/freebase_strat16_for_gege` was derived from:

- [train.del](/home/smansou2/dist-kge/data/freebase/train.del)
- [entity_to_partitions.del](/home/smansou2/dist-kge/data/freebase/partitions/stratification/num_16/entity_to_partitions.del)

## Latest Twitter 16p Schedule

`slot_to_partition`

```text
[4, 8, 13, 6, 15, 2, 3, 9, 12, 1, 7, 5, 14, 0, 10, 11]
```

Score:

```text
worst_round_spread = 3.115M
worst_batch_spread = 63
worst_state_weight = 93.364M
total_round_spread = 14.466M
continuity_hotness = 8489.428M
continuity_new_partitions = 48
```

Rounds:

```text
Round 1
GPU0: {4,8,13,6}   90.629M edges, 1813 batches
GPU1: {12,1,7,5}   91.998M edges, 1840 batches
GPU2: {15,2,3,9}   91.131M edges, 1823 batches
GPU3: {14,0,10,11} 93.364M edges, 1868 batches

Round 2
GPU0: {13,3,7,10}  70.381M edges, 1408 batches
GPU1: {4,15,12,14} 67.330M edges, 1347 batches
GPU2: {8,2,1,0}    70.347M edges, 1407 batches
GPU3: {6,9,5,11}   67.280M edges, 1346 batches

Round 3
GPU0: {8,15,5,10}  67.459M edges, 1350 batches
GPU1: {13,9,12,0}  70.244M edges, 1405 batches
GPU2: {4,2,7,11}   70.357M edges, 1408 batches
GPU3: {6,3,1,14}   67.242M edges, 1345 batches

Round 4
GPU0: {8,9,7,14}   68.074M edges, 1362 batches
GPU1: {13,15,1,11} 68.120M edges, 1363 batches
GPU2: {6,2,12,10}  71.100M edges, 1423 batches
GPU3: {4,3,5,0}    68.013M edges, 1361 batches

Round 5
GPU0: {4,9,1,10}   67.392M edges, 1348 batches
GPU1: {8,3,12,11}  69.821M edges, 1397 batches
GPU2: {13,2,5,14}  69.724M edges, 1395 batches
GPU3: {6,15,7,0}   68.359M edges, 1368 batches
```

## Twitter 16p: Current CUSTOM vs Optimized

For a concrete "current CUSTOM" baseline, use the observed epoch-4 schedule from
[twitter16p_table5_relay_analysis.md](/home/smansou2/newCode/ge2/dandelion-dev/twitter16p_table5_relay_analysis.md).
That was the best current-custom schedule in the recent run log.

### Current CUSTOM Schedule

| Round | GPU0 | GPU1 | GPU2 | GPU3 | Spread |
|---|---:|---:|---:|---:|---:|
| 1 | 83.219M | 82.660M | 76.194M | 73.167M | 10.052M |
| 2 | 70.649M | 62.592M | 62.378M | 62.101M | 8.548M |
| 3 | 65.040M | 62.624M | 63.344M | 61.968M | 3.072M |
| 4 | 62.509M | 63.868M | 60.999M | 60.434M | 3.434M |
| 5 | 61.247M | 61.682M | 61.552M | 63.304M | 2.057M |

States:

```text
Round 1
GPU0: {12,8,13,14}
GPU1: {9,1,2,6}
GPU2: {11,7,4,5}
GPU3: {15,3,10,0}

Round 2
GPU0: {11,10,6,8}
GPU1: {4,15,1,14}
GPU2: {5,3,9,13}
GPU3: {7,0,2,12}

Round 3
GPU0: {7,15,6,13}
GPU1: {5,10,1,12}
GPU2: {11,3,2,14}
GPU3: {4,0,9,8}

Round 4
GPU0: {4,3,6,12}
GPU1: {5,15,2,8}
GPU2: {7,10,9,14}
GPU3: {11,0,1,13}

Round 5
GPU0: {11,15,9,12}
GPU1: {4,10,2,13}
GPU2: {7,3,1,8}
GPU3: {5,0,6,14}
```

### Optimized Schedule

| Round | GPU0 | GPU1 | GPU2 | GPU3 | Spread |
|---|---:|---:|---:|---:|---:|
| 1 | 90.629M | 91.998M | 91.131M | 93.364M | 2.735M |
| 2 | 70.381M | 67.330M | 70.347M | 67.280M | 3.101M |
| 3 | 67.459M | 70.244M | 70.357M | 67.242M | 3.115M |
| 4 | 68.074M | 68.120M | 71.100M | 68.013M | 3.087M |
| 5 | 67.392M | 69.821M | 69.724M | 68.359M | 2.428M |

States:

```text
Round 1
GPU0: {4,8,13,6}
GPU1: {12,1,7,5}
GPU2: {15,2,3,9}
GPU3: {14,0,10,11}

Round 2
GPU0: {13,3,7,10}
GPU1: {4,15,12,14}
GPU2: {8,2,1,0}
GPU3: {6,9,5,11}

Round 3
GPU0: {8,15,5,10}
GPU1: {13,9,12,0}
GPU2: {4,2,7,11}
GPU3: {6,3,1,14}

Round 4
GPU0: {8,9,7,14}
GPU1: {13,15,1,11}
GPU2: {6,2,12,10}
GPU3: {4,3,5,0}

Round 5
GPU0: {4,9,1,10}
GPU1: {8,3,12,11}
GPU2: {13,2,5,14}
GPU3: {6,15,7,0}
```

### Summary Comparison

| Metric | Current CUSTOM | Optimized |
|---|---:|---:|
| Worst round spread | 10.052M | 3.115M |
| Best round spread | 2.057M | 2.428M |
| Total round spread | 27.163M | 14.466M |

Interpretation:

- the optimized schedule is much more uniform across all 5 rounds
- the current schedule gets very good only in the later rounds
- the optimized schedule removes the two badly imbalanced early rounds

## Latest Freebase86M Stratification-16 Schedule

`slot_to_partition`

```text
[5, 9, 3, 14, 12, 7, 11, 10, 2, 6, 13, 1, 15, 0, 4, 8]
```

Score:

```text
worst_round_spread = 2.572M
worst_batch_spread = 52
worst_state_weight = 20.227M
total_round_spread = 12.556M
continuity_hotness = 1730.217M
continuity_new_partitions = 48
```

Rounds:

```text
Round 1
GPU0: {5,9,3,14}   18.374M edges, 368 batches
GPU1: {2,6,13,1}   17.769M edges, 356 batches
GPU2: {12,7,11,10} 19.862M edges, 398 batches
GPU3: {15,0,4,8}   20.227M edges, 405 batches

Round 2
GPU0: {5,12,2,15}  12.992M edges, 260 batches
GPU1: {3,11,13,4}  13.014M edges, 261 batches
GPU2: {14,10,1,8}  15.564M edges, 312 batches
GPU3: {9,7,6,0}    15.553M edges, 312 batches

Round 3
GPU0: {9,12,1,4}   13.645M edges, 273 batches
GPU1: {5,7,13,8}   13.747M edges, 275 batches
GPU2: {14,11,6,15} 13.588M edges, 272 batches
GPU3: {3,10,2,0}   16.140M edges, 323 batches

Round 4
GPU0: {9,10,13,15} 14.560M edges, 292 batches
GPU1: {3,12,6,8}   12.607M edges, 253 batches
GPU2: {14,7,2,4}   14.926M edges, 299 batches
GPU3: {5,11,1,0}   15.033M edges, 301 batches

Round 5
GPU0: {5,10,6,4}   14.003M edges, 281 batches
GPU1: {9,11,2,8}   13.436M edges, 269 batches
GPU2: {3,7,1,15}   13.704M edges, 275 batches
GPU3: {14,12,13,0} 15.983M edges, 320 batches
```

## Important Interpretation Notes

- `continuity_new_partitions = 48` is not a bug. Under the current exact-cover 5-round `CUSTOM` template, this is expected:
  - `4` GPUs
  - `4` round boundaries
  - best possible overlap is `1` kept partition out of `4`
  - so `3` new partitions per GPU per boundary
  - `4 * 4 * 3 = 48`
- The GPU-aware term is therefore selecting the **best lane ordering** under the current template, not changing the fact that the template itself only permits one kept partition across rounds.
- If stronger continuity is desired, the template family itself must change.
