# IO-Aware Exact LP Operator Design

## Goal

Turn the current link-prediction negative path into a single exact operator that
is worth a systems paper:

1. sampled score generation
2. online row-wise selection
3. selected score materialization
4. sparse gradient reduction into negative embeddings

The operator should reduce HBM traffic, reduce temporary tensor size, improve
warp-level work efficiency, and create a clean path to larger-scale and
multi-GPU execution.

This document is the implementation contract for that work. It is intentionally
more specific than a paper outline.

## Current GE2 LP Dataflow

For `DISTMULT` in `DNS` and `GAN`, the hot path is currently:

1. build candidate score tensors with `bmm`
2. select negatives row-wise with `tournament` or `topk`
3. compute selected-negative scores
4. backpropagate into query embeddings and selected negative embeddings

Relevant code paths:

- candidate score build:
  - `gege/src/cpp/include/data/samplers/negative.h`
- selection and selected-score orchestration:
  - `gege/src/cpp/src/nn/decoders/edge/decoder_methods.cpp`
- current selected-negative CUDA forward/backward:
  - `gege/src/cuda/src/nn/decoders/edge/distmult_selected_neg_cuda.cu`
- final storage update after sparse reduction:
  - `gege/src/cpp/src/data/batch.cpp`
  - `gege/src/cpp/src/storage/storage.cpp`

## Problem Statement

The current path still pays three avoidable costs:

1. full candidate-score materialization
2. redundant HBM traffic between scoring and selection
3. atomic-heavy negative-gradient accumulation

For a common training setup:

- `batch_size = 10000`
- `num_chunks = 10`
- `negatives_per_positive = 1000`
- `selected_ratio = 0.2`

the candidate tensor per side is:

- `chunk_num * num_per_chunk * negatives_per_positive`
- `10 * 1000 * 1000 = 10,000,000` scores
- with `float32`, that is about `40 MB` per side before any rereads

That is the wrong IO pattern for modern GPUs. The operator should move the
intermediate from `O(rows * negatives)` storage toward `O(rows * selected)`.

## Research Hypothesis

An exact LP operator that fuses:

- sampled score generation
- online selection
- selected-score forward
- sparse negative-gradient reduction

will reduce runtime and memory footprint more than isolated optimizations such
as:

- replacing `topk` alone
- replacing gather/scatter alone
- introducing standalone `SDDMM` or `SpMM` kernels without fused selection

The closest systems analogy is FlashAttention:

- exact semantics
- IO-aware tiling
- no materialization of the large intermediate

The closest graph-kernel prior is FusedMM, but the LP workload here is not a
plain `SDDMM -> SpMM` pipeline because row-wise negative selection is in the
middle.

## Operator Contract

The target operator for `DISTMULT` is:

`FlashLPScoreSelectReduce`

Forward interface:

- `query_sel`: selector query tensor `[chunk, row, dim]`
- `neg_sel`: selector negative tensor `[chunk, neg, dim]`
- `query_score`: scorer query tensor `[chunk, row, dim]`
- `neg_score`: scorer negative tensor `[chunk, neg, dim]`
- `mode`: `TOURNAMENT` or `TOPK`
- `selected_negatives_num`
- `tournament_size` if `mode == TOURNAMENT`

Forward outputs:

- `selected_indices`: `[chunk, row * selected_negatives_num]`
- `selected_scores`: `[chunk * row, selected_negatives_num]`

Backward outputs:

- `d_query_score`
- `d_neg_score`

Selector tensors receive no gradient in the current `DNS/GAN` execution because
selection runs under `NoGradGuard`.

## DistMult-Specific Math

For `DISTMULT`, each row score is a dot product:

- `score(q, n) = sum_d q[d] * n[d]`

where `q` is already relation-adjusted in the existing decoder path.

This is ideal for:

- tiled register accumulation
- tensor-core later, if we move to padded `dim = 128`
- sparse backward because each selected pair contributes an outer product

## Proposed Operator Structure

The operator has two execution modes:

### Mode A: Exact Tournament

This is the first implementation target.

For each row:

- negatives are partitioned into `selected_negatives_num` groups
- each group has `tournament_size = negatives / selected_negatives_num`
- the operator keeps the maximum score and index in each group

This is easier than exact `topk` because:

- selection state is fixed size per group
- no global row-wise heap or sort is needed
- all selection state fits in registers

### Mode B: Exact Top-k

This is the second target.

Replace per-group max with warp-level `WarpSelect` style state:

- each warp keeps a fixed-size row-local top-k structure in registers
- tiles stream through and update the running row state
- output is exact top-k without materializing the full score tensor

This mode is more complex and should come only after tournament mode is stable.

## Kernel Decomposition

The full operator is split into three kernels or kernel families.

### Kernel F1: Fused Forward Score + Select + Selected Score

Input:

- `query_sel`, `neg_sel`, `query_score`, `neg_score`

Output:

- `selected_indices`
- `selected_scores`

Responsibilities:

- tile rows, negatives, and embedding dimension
- compute selector scores on the fly
- update row-local selection state online
- compute selected scorer values for the winning entries
- never write the full candidate-score matrix

### Kernel B1: Dense Query Gradient

Input:

- `grad_selected_scores`
- `selected_indices`
- `neg_score`

Output:

- `d_query_score`

Responsibilities:

- load selected negative rows
- accumulate row-local dense gradients
- write one contiguous row gradient per query row

This part is dense and should be straightforward.

### Kernel B2: Sparse Negative Gradient Reduce

Input:

- `grad_selected_scores`
- `selected_indices`
- `query_score`

Output:

- `d_neg_score`

Responsibilities:

- compute pair contributions for selected negatives
- reduce duplicates inside the CTA before global writeback
- emit one reduced vector per unique negative row in the CTA

This is the `SpMM-like` part of the design.

## Recommended Thread-Block Geometry

The operator should not tile over full embedding rows in shared memory. It
should tile over `dim` as well, otherwise shared-memory footprint gets too
large for `dim = 100` or `128`.

Recommended initial geometry for `DISTMULT`, `FP32`, exact tournament:

- `CTA_THREADS = 128`
- `WARPS = 4`
- `ROW_TILE = 8`
- `GROUP_TILE = 8`
- `TOURNAMENT_SIZE = 5` when `selected_ratio = 0.2`
- `NEG_TILE = GROUP_TILE * TOURNAMENT_SIZE = 40`
- `DIM_TILE = 32`

Warp ownership:

- each warp owns `2` query rows
- lanes cooperate over `NEG_TILE` candidates and `DIM_TILE` dimensions
- row-local selection state stays in registers

Why this geometry:

- small enough shared-memory footprint
- enough rows per CTA to amortize negative-tile loads
- low register pressure for tournament state
- easy path to `dim = 128` later

Alternative geometry for BF16 or FP16 on `dim = 128`:

- `CTA_THREADS = 128 or 256`
- `ROW_TILE = 16`
- `GROUP_TILE = 16`
- `DIM_TILE = 32`

This mode is for tensor-core specialization later, not the first paper result.

## Shared-Memory Layout

Forward kernel shared-memory layout per CTA:

```text
q_sel_smem[2][ROW_TILE][DIM_TILE]
n_sel_smem[2][NEG_TILE][DIM_TILE]
q_score_smem[2][ROW_TILE][DIM_TILE]    // optional if different from selector
n_score_smem[2][NEG_TILE][DIM_TILE]    // optional if different from selector
```

Notes:

- `2` means double-buffered ping-pong buffers
- if selector and scorer tensors alias, reuse one buffer instead of two
- for `DNS`, selector and scorer are identical after detach, so one buffer path
  is valid
- for `GAN`, selector and scorer are different, so keep both

Approximate footprint for `FP32`, `ROW_TILE=8`, `NEG_TILE=40`, `DIM_TILE=32`:

- one query tile: `8 * 32 * 4 = 1 KB`
- one negative tile: `40 * 32 * 4 = 5 KB`
- one selector + scorer buffer pair: about `12 KB`
- double-buffered selector + scorer pair: about `24 KB`

This fits comfortably and leaves room for selection state.

## Register State

### Tournament Mode

Per row and group:

- `best_score[row_local][group_local]`
- `best_index[row_local][group_local]`

For the recommended geometry:

- `2 rows/warp`
- `8 groups/CTA`
- each warp needs only a small fixed register footprint

### Top-k Mode

Per row:

- fixed-size `k` score register array
- fixed-size `k` index register array

Use a warp-local `WarpSelect` style update rule after each score fragment.

## Forward Algorithm

For each `(chunk, row_tile)` CTA:

1. initialize row-local selection state in registers
2. for each negative-group tile:
   - for each `dim_tile`:
     - async load query and negative fragments into shared memory
     - accumulate partial scores in registers
   - finalize selector scores for the current negative tile
   - update tournament or top-k row state
3. when a group tile finishes:
   - write winning indices
   - compute scorer values for winners if selector and scorer differ
4. write `selected_indices` and `selected_scores`

Important:

- do not write partial score tiles to global memory
- only write the final selected outputs

## Backward Design

### Query Gradient

The query gradient is dense:

- for each row and selected negative
- `d_query += grad_score * neg_score[selected_idx]`

Kernel B1 should:

- reuse `selected_indices`
- tile selected negatives
- accumulate one dense output row per query row

This is effectively a small dense GEMM-like reduce and should not be the hard
part.

### Negative Gradient

The negative gradient is sparse:

- for each selected pair
- `d_neg[selected_idx] += grad_score * query_score[row]`

The current implementation uses global atomics directly. That is correct, but
too expensive.

#### Proposed On-Chip Reduction

For each CTA:

1. generate pair contributions for its `(row_tile, selected_negatives_num)` work
2. aggregate duplicate negative ids in shared memory before global writeback

Recommended structure:

- a small shared-memory hash table keyed by local negative id
- each slot stores:
  - `negative_idx`
  - partial gradient vector accumulator over `DIM_TILE`

Algorithm:

1. each warp computes pair contributions
2. insert or update shared-memory hash slot for the negative id
3. accumulate vector fragments per slot
4. after the tile finishes, flush one reduced vector per occupied slot to
   global memory

This changes the global write complexity from roughly:

- `O(row_tile * selected_negatives_num * dim)`

to:

- `O(unique_negatives_in_cta * dim)`

which is the right objective.

#### Why Not Full Sort-and-Segment Here

We already tried external sort-and-segment reduction on the backward path and it
lost badly because:

- sort cost dominated
- extra materialization cost dominated

The correct reduction boundary is on-chip, inside the fused operator.

## Complexity Story

Let:

- `R = chunk_num * num_per_chunk`
- `N = negatives_per_positive`
- `K = selected_negatives_num`
- `D = embedding_dim`

Current forward memory behavior:

- candidate scores written: `O(R * N)`
- selected scores written: `O(R * K)`

Proposed forward memory behavior:

- candidate scores written: `0`
- selected scores written: `O(R * K)`

Current negative-gradient update:

- global atomic vector updates: `O(R * K * D)`

Proposed negative-gradient update:

- global writes after CTA-local reduction: `O(U * D)`
- `U` is the number of CTA-local unique selected negatives

In workloads with repeated hard negatives inside a tile, `U << R * K`.

## Cache and Locality Extension

This is a secondary research extension, not the first implementation target.

### Superbatch Negative-Tile Cache

Idea:

- keep a short-lived device-side cache of negative embedding tiles for a
  superbatch of microbatches
- reuse loaded negative tiles across multiple query row tiles before evicting
  them

This is not a semantic hard-negative cache. It is a systems cache over already
sampled negative tiles.

Good cache key:

- `(side, chunk_local_pool_id, neg_tile_id, relation_bucket, version)`

Why it can help:

- `DNS/GAN` use the same sampled negative pool across many rows inside a chunk
- row tiles march over the same negative tiles repeatedly
- cache reuse should be high inside a chunk even if cross-batch reuse is low

This is a locality optimization. It does not change the learning objective.

## Scalability Path

The operator is chunk-local, which is favorable for scaling.

### Single GPU

Expected benefits:

- reduced HBM traffic
- reduced temporary tensor size
- fewer kernel launches
- better arithmetic intensity

### Multi-GPU

The operator composes well with partitioned execution because:

- selected negative state is chunk-local
- sparse negative-gradient writeback is chunk-local before storage update
- the operator does not require cross-GPU communication by itself

The later multi-GPU paper section should combine:

- this fused LP operator
- COVER or improved partition scheduling
- superstep or owner-compute sparse update routing if needed

But the operator paper contribution should stand on one GPU first.

## Implementation Roadmap

### Phase 0: Current Kept State

Already in tree:

- tiled tournament forward implemented in C++/ATen
- config-driven enablement
- exact validation hooks

This is a good baseline, not the end state.

### Phase 1: CUDA Forward Tournament Kernel

Implement:

- `DISTMULT + DNS/GAN + tournament_selection`
- exact forward score + select + selected score

Keep:

- current backward path unchanged

Success criterion:

- beat current tiled ATen forward path
- exact match on selected indices and selected scores

### Phase 2: CUDA Backward Sparse Reduce

Implement:

- dense query-gradient kernel
- shared-memory hash reduce for negative gradients

Success criterion:

- beat current atomic-heavy backward
- exact or `allclose` gradient match

### Phase 3: Exact Top-k Mode

Add:

- warp-level top-k state
- no tournament assumption

Success criterion:

- beat current `topk(sorted=false)` path

### Phase 4: Decoder Generalization

Extend to:

- `ComplEx`
- `TransE`

For `TransE`, use squared-distance form to keep the forward path LA-friendly.

## Validation Contract

The guarded implementation must satisfy:

- exact equality of selected indices against the current reference
- `allclose` on selected scores
- `allclose` on loss
- `allclose` on `d_query_score`
- `allclose` on `d_neg_score`
- stable end-to-end MRR over multi-epoch validation

If any of those fail, the guarded path stays off.

## Experimental Plan

### E1: Kernel Microbench

Measure:

- runtime
- DRAM bytes
- achieved occupancy
- shared-memory usage
- global atomic count

Compare:

- current ATen tiled tournament path
- CUDA forward-only tournament kernel
- full fused forward+backward kernel

### E2: End-to-End Single-GPU LP

Datasets:

- FB15K
- LiveJournal
- Twitter

Metrics:

- epoch runtime
- edges per second
- peak GPU memory
- MRR and Hits@K

### E3: Ablation Study

Variants:

1. baseline
2. tiled forward only
3. tiled forward + CUDA forward kernel
4. full fused forward + backward
5. full fused + cache

### E4: Sampler Sensitivity

Sweep:

- `negatives_per_positive`
- `negative_sampling_selected_ratio`
- `num_chunks`

Goal:

- show when IO reduction matters most

### E5: Architecture Sensitivity

Sweep:

- `dim = 100`
- `dim = 128`
- `FP32`
- `BF16/FP16 + FP32 accumulation`

Goal:

- separate the IO-aware gain from tensor-core gain

### E6: Multi-GPU

Only after single-GPU operator is stable:

- 1, 2, 4 GPUs
- partition-buffer and device-memory settings
- measure scaling efficiency and communication overhead

## Paper Positioning

The paper claim should be:

- existing KGE systems optimize sampling, partitioning, or storage
- existing graph kernels optimize `SDDMM` and `SpMM`
- but LP hard-negative training is bottlenecked by a dynamic
  `score -> select -> sparse update` operator
- we introduce an exact IO-aware fused operator for that workload

That is stronger than claiming:

- generic sparse kernel use
- generic cache tuning
- generic warp scheduling

## References To Anchor The Positioning

- FlashAttention
- FlashAttention-2
- FusedMM
- WarpSelect / GPU k-selection
- DGL-KE
- GraphVite
- Marius
- NSCaching

These should be used in the paper to position the operator as:

- exact
- IO-aware
- LP-specific
- dynamic-selection aware
