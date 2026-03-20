# Twitter16P Table5 Relay Run Analysis

Run context:

- binary: `./build/gege/gege_train`
- config: `/tmp/twitter_16p_table5_approx_5e_noeval.yaml`
- devices: `CUDA_VISIBLE_DEVICES=0,1,2,3`
- env: `GEGE_PARTITION_BUFFER_PEER_RELAY=1`, `GEGE_UNIQUE_BACKEND=bitmap`
- negative sampling: `RNS`
- note: GPUs were not fully exclusive during the run, so absolute epoch times may be inflated by external load; the balance patterns below are still meaningful.

## How To Read The Timing

- `Epoch Runtime` is the real wall-clock epoch time.
- `[perf][epoch N] ..._sum_ms` values are sums across all 4 GPU threads.
- In the tables below, `per-GPU avg` means `epoch sum / 4`, which is easier to compare against wall-clock.
- `swap path` means `swap_barrier_wait + swap_update + swap_rebuild + swap_sync_wait`.
- `state 0` is the initial state before the epoch loop; `state 1..4` are swap-built states.

## Epoch Summary

| Epoch | Runtime (s) | Throughput (M edges/s) | Neg samp / GPU avg (s) | Map lookup / GPU avg (s) | Swap path / GPU avg (s) | Compute / GPU avg (s) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 272.874 | 4.843 | 177.365 | 15.482 | 65.357 | 7.366 |
| 2 | 258.522 | 5.112 | 177.571 | 15.506 | 56.999 | 7.047 |
| 3 | 259.745 | 5.088 | 176.770 | 15.280 | 57.508 | 7.216 |
| 4 | 243.858 | 5.419 | 178.445 | 15.957 | 40.898 | 7.334 |

## Main Findings

- Negative sampling dominates every epoch. It stays near `177-178s per GPU avg` across epochs.
- Compute is small, around `7s per GPU avg`.
- The big runtime improvement in epoch 4 comes from better balance and much lower barrier time, not from less total negative-sampling work.
- Sampler lock contention is not the issue:
  - `plan_lock_calls=0`
  - `plan_lock_wait_ms_total=0.000`
- The bottleneck in epochs 1-3 is lane imbalance: one lane gets more active edges, more batches, more negative-sampler calls, and then the other GPUs wait.

## Per-GPU Summary: Epoch 2

| GPU | Batches | Neg calls | Neg total (s) | Neg avg / call (ms) | Batch fetch (s) | Get next batch (s) | Edge sample (s) | Map lookup (s) | Compute (s) | Barrier wait (s) | Rebuild avg / state (s) | Active edges avg / state (M) | Batches avg / state |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 7437 | 14874 | 199.236 | 13.395 | 249.554 | 30.504 | 219.046 | 18.054 | 8.614 | 0.000 | 5.446 | 72.220 | 1444.8 |
| 1 | 6517 | 13034 | 173.932 | 13.344 | 251.332 | 60.711 | 190.617 | 15.020 | 6.329 | 30.126 | 2.455 | 63.610 | 1272.5 |
| 2 | 6230 | 12460 | 165.994 | 13.322 | 250.010 | 68.201 | 181.805 | 14.178 | 6.897 | 37.613 | 3.505 | 62.415 | 1248.8 |
| 3 | 6254 | 12508 | 165.711 | 13.248 | 251.467 | 69.356 | 182.108 | 14.772 | 6.348 | 38.750 | 2.806 | 62.507 | 1250.5 |

Interpretation:

- `gpu 0` is the heavy lane in epoch 2.
- Per-call negative-sampler time is nearly identical on all GPUs.
- `gpu 0` is slower because it has more state work and more batches, not because sampling is intrinsically slower on that GPU.

## Per-GPU Summary: Epoch 4

| GPU | Batches | Neg calls | Neg total (s) | Neg avg / call (ms) | Batch fetch (s) | Get next batch (s) | Edge sample (s) | Map lookup (s) | Compute (s) | Barrier wait (s) | Rebuild avg / state (s) | Active edges avg / state (M) | Batches avg / state |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6855 | 13710 | 186.709 | 13.618 | 235.447 | 29.663 | 205.780 | 17.398 | 7.862 | 0.000 | 3.871 | 64.861 | 1297.5 |
| 1 | 6671 | 13342 | 178.721 | 13.395 | 236.231 | 40.061 | 196.166 | 15.735 | 6.587 | 10.261 | 3.862 | 62.691 | 1254.2 |
| 2 | 6491 | 12982 | 173.030 | 13.328 | 235.166 | 45.529 | 189.632 | 14.873 | 7.665 | 15.714 | 3.885 | 62.068 | 1241.8 |
| 3 | 6423 | 12846 | 169.908 | 13.227 | 236.302 | 48.996 | 187.302 | 15.823 | 7.222 | 19.179 | 2.504 | 61.952 | 1239.8 |

Interpretation:

- Epoch 4 is much more balanced than epoch 2.
- `gpu 0` is still the critical path, but the gap is much smaller.
- That is why epoch 4 throughput improves and barrier wait drops.

## Superstep State Table: Epoch 2

### State 0 (`phase=init`)

| GPU | State idx | Resident partitions | Active buckets | Active edges | Batches |
|---|---:|---|---:|---:|---:|
| 0 | 0 | `6,10,3,8` | 16 | 82,865,686 | 1,658 |
| 1 | 1 | `4,12,13,1` | 14 | 71,340,687 | 1,427 |
| 2 | 2 | `0,11,9,14` | 12 | 61,744,827 | 1,235 |
| 3 | 3 | `7,2,15,5` | 12 | 62,570,662 | 1,252 |

### State 1 (`phase=swap`)

| GPU | State idx | Resident partitions | Active buckets | Active edges | Batches |
|---|---:|---|---:|---:|---:|
| 0 | 4 | `0,12,3,5` | 15 | 79,786,265 | 1,596 |
| 1 | 5 | `7,10,13,14` | 12 | 60,545,062 | 1,211 |
| 2 | 6 | `4,11,15,8` | 12 | 61,388,647 | 1,228 |
| 3 | 7 | `6,2,9,1` | 12 | 61,987,051 | 1,240 |

### State 2 (`phase=swap`)

| GPU | State idx | Resident partitions | Active buckets | Active edges | Batches |
|---|---:|---|---:|---:|---:|
| 0 | 8 | `4,2,3,14` | 15 | 80,095,016 | 1,602 |
| 1 | 9 | `7,12,9,8` | 14 | 70,897,390 | 1,418 |
| 2 | 10 | `6,11,13,5` | 12 | 60,867,715 | 1,218 |
| 3 | 11 | `0,10,15,1` | 12 | 61,782,565 | 1,236 |

### State 3 (`phase=swap`)

| GPU | State idx | Resident partitions | Active buckets | Active edges | Batches |
|---|---:|---|---:|---:|---:|
| 0 | 12 | `2,12,11,10` | 13 | 66,840,269 | 1,337 |
| 1 | 13 | `4,7,6,0` | 12 | 60,610,946 | 1,213 |
| 2 | 14 | `3,9,13,15` | 13 | 66,849,959 | 1,337 |
| 3 | 15 | `14,8,5,1` | 12 | 63,544,883 | 1,271 |

### State 4 (`phase=swap`)

| GPU | State idx | Resident partitions | Active buckets | Active edges | Batches |
|---|---:|---|---:|---:|---:|
| 0 | 16 | `4,10,9,5` | 12 | 62,158,862 | 1,244 |
| 1 | 17 | `0,2,13,8` | 12 | 62,386,574 | 1,248 |
| 2 | 18 | `7,11,3,1` | 12 | 60,553,760 | 1,212 |
| 3 | 19 | `6,12,15,14` | 12 | 62,711,837 | 1,255 |

Epoch 2 takeaway:

- `gpu 0` is overloaded in the first three states, especially states `0`, `1`, and `2`.
- This explains why `gpu 0` becomes the straggler and other GPUs accumulate barrier wait.

## Superstep State Table: Epoch 4

### State 0 (`phase=init`)

| GPU | State idx | Resident partitions | Active buckets | Active edges | Batches |
|---|---:|---|---:|---:|---:|
| 0 | 0 | `12,8,13,14` | 16 | 83,218,652 | 1,665 |
| 1 | 1 | `9,1,2,6` | 16 | 82,659,512 | 1,654 |
| 2 | 2 | `11,7,4,5` | 15 | 76,194,148 | 1,524 |
| 3 | 3 | `15,3,10,0` | 14 | 73,166,525 | 1,464 |

### State 1 (`phase=swap`)

| GPU | State idx | Resident partitions | Active buckets | Active edges | Batches |
|---|---:|---|---:|---:|---:|
| 0 | 4 | `11,10,6,8` | 14 | 70,648,836 | 1,413 |
| 1 | 5 | `4,15,1,14` | 12 | 62,592,311 | 1,252 |
| 2 | 6 | `5,3,9,13` | 12 | 62,377,606 | 1,248 |
| 3 | 7 | `7,0,2,12` | 12 | 62,101,142 | 1,243 |

### State 2 (`phase=swap`)

| GPU | State idx | Resident partitions | Active buckets | Active edges | Batches |
|---|---:|---|---:|---:|---:|
| 0 | 8 | `7,15,6,13` | 13 | 65,039,549 | 1,301 |
| 1 | 9 | `5,10,1,12` | 12 | 62,623,554 | 1,253 |
| 2 | 10 | `11,3,2,14` | 12 | 63,343,823 | 1,267 |
| 3 | 11 | `4,0,9,8` | 12 | 61,968,031 | 1,240 |

### State 3 (`phase=swap`)

| GPU | State idx | Resident partitions | Active buckets | Active edges | Batches |
|---|---:|---|---:|---:|---:|
| 0 | 12 | `4,3,6,12` | 12 | 62,508,725 | 1,251 |
| 1 | 13 | `5,15,2,8` | 12 | 63,868,015 | 1,278 |
| 2 | 14 | `7,10,9,14` | 12 | 60,999,481 | 1,220 |
| 3 | 15 | `11,0,1,13` | 12 | 60,434,486 | 1,209 |

### State 4 (`phase=swap`)

| GPU | State idx | Resident partitions | Active buckets | Active edges | Batches |
|---|---:|---|---:|---:|---:|
| 0 | 16 | `11,15,9,12` | 12 | 61,246,974 | 1,225 |
| 1 | 17 | `4,10,2,13` | 12 | 61,681,827 | 1,234 |
| 2 | 18 | `7,3,1,8` | 12 | 61,551,814 | 1,232 |
| 3 | 19 | `5,0,6,14` | 12 | 63,303,652 | 1,267 |

Epoch 4 takeaway:

- The state loads are much closer.
- This is exactly when runtime improves and barrier wait drops.

## Why The Run Improved In Epoch 4

Comparing epoch 2 to epoch 4:

| Metric | Epoch 2 | Epoch 4 | Change |
|---|---:|---:|---:|
| Runtime (s) | 258.522 | 243.858 | -14.664 |
| Throughput (M edges/s) | 5.112 | 5.419 | +0.307 |
| Swap barrier total (s) | 106.489 | 45.154 | -61.335 |
| Neg sampler total (s) | 704.873 | 708.368 | +3.495 |

Interpretation:

- Negative-sampler total work stayed basically flat.
- The runtime improvement came from lower imbalance, mainly much less barrier waiting.

## Overall Conclusion

The log supports this conclusion:

1. The main wall-clock cost is still negative sampling plus the mapping work it creates.
2. The main *performance problem* in the multi-GPU run is imbalance in per-GPU state assignment.
3. Epoch 4 shows that when state loads become more balanced, runtime improves substantially even though total negative-sampler work does not go down.
4. The next optimization target should be buffer-state scheduling and lane balancing, not sampler lock contention.
