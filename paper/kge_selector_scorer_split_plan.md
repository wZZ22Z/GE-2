# KGE Selector/Scorer Split Plan

## Goal

Introduce an asymmetric KGE execution path:

- `selector` path: cheap, no-grad, allowed to be stale/compressed/cached
- `scorer` path: exact, fresh, gradient-carrying

This is the KGE-specific systems idea worth pursuing next. The selector sees
the wide candidate set; the scorer sees only the selected negatives.

## Why It Fits This Repo

Current negative sampling already has a natural phase split:

1. raw negative candidate generation
   - `gege/src/cpp/src/data/samplers/negative.cpp`
2. selector score build and negative selection
   - `gege/src/cpp/include/data/samplers/negative.h`
   - `gege/src/cpp/src/nn/decoders/edge/decoder_methods.cpp`
3. exact selected-negative scoring and backward
   - `gege/src/cpp/src/nn/decoders/edge/decoder_methods.cpp`
   - `gege/src/cpp/src/nn/decoders/edge/distmult_selected_neg_cuda.cpp`
   - `gege/src/cuda/src/nn/decoders/edge/distmult_selected_neg_cuda.cu`

The selector phase is already wrapped in `NoGradGuard` in the decoder path for
`DNS` and `GAN`. That is the asymmetry we should exploit.

## Proposed Architecture

### Phase A: Superbatch Negative Planner

Add a planning layer that precomputes raw candidate negative IDs for the next
`W` batches:

- planner lives beside `NegativeSamplingBase`
- planner output is a queue of raw candidate ID tensors
- planner is independent of filtering and final score computation

This is implemented first because it creates the control point needed for later
packing and caching without changing training semantics when disabled.

### Phase B: Packed Selector Slab

For the next `W` batches:

- union positive entity IDs and raw candidate negative IDs
- build a contiguous local slab of selector embeddings
- remap candidates to local slab indices

The selector reads only from this packed slab.

### Phase C: Stale/Compressed Selector Table

Replace the selector slab source with:

- stale GPU mirror
- or compressed/cached CPU tier

The scorer path still uses the exact fresh embeddings for selected negatives.

### Phase D: Negative-Pool-Aware Scheduling

Schedule batches/superbatches using overlap in:

- positive entity working set
- planned negative pools
- packed slab reuse

This is the multi-GPU extension point.

## Concrete Code Hooks

### Planner

- `NegativeSamplingBase`
  - own the raw candidate planning cache
  - expose reset on epoch / storage transitions
- `DataLoader::negativeSample`
  - remains the batch boundary
  - will consume planned raw negatives transparently

### Selector Slab

- extend `Batch` with optional selector-local negative IDs
- add a selector-specific gather/materialize path in
  - `decoder_methods.cpp`
- keep scorer path unchanged at first

### Exact Scorer

- keep current selected-negative exact scorer path:
  - `selected_neg_scores`
  - tiled tournament path

This reduces risk in the first selector/scorer split.

## Acceptance Criteria

Phase A:

- no correctness change with planner enabled
- no schema regressions
- planner disabled by default

Phase B:

- exact final scorer/update path unchanged
- reduced selector gather traffic
- stable accuracy on FB15K and LiveJournal

Phase C:

- selector-only staleness/compression
- exact scorer path remains unchanged

## Experimental Path

1. `FB15K + DISTMULT + DNS/GAN`
2. `LiveJournal + DISTMULT + GAN`
3. `Twitter` only after larger-dataset signal is clearly positive

The planner itself is infrastructure. The packed selector slab is the first
step expected to move runtime.
