# Bucket-Streaming KGE Plan

## Goal

Replace whole-state partition-buffer materialization with a bucket-streaming
execution path for large-scale KGE.

The current `twitter_16p` training path:

1. builds the full active state edge tensor
2. remaps the full state into local IDs
3. optionally builds sorted graph structures
4. extracts active buckets from the materialized state
5. shuffles the full active-edge tensor
6. batches from that shuffled tensor

This plan changes the execution boundary from **state** to **partition-pair edge
bucket**.

## Why This Fits GE2

GE2/COVER already optimizes communication under whole-partition movement. The
remaining overhead is local:

- state construction
- state remap
- state-local graph rebuilding
- state-wide shuffling

Bucket streaming does not try to beat COVER's communication lower bound. It
reduces the work that COVER does not model.

## Current Code Boundary

### State construction and remap

- `gege/src/cpp/src/storage/graph_storage.cpp`
  - `GraphModelStorage::initializeInMemorySubGraph`
  - `GraphModelStorage::updateInMemorySubGraph_`

These currently:

- enumerate active bucket IDs
- load every bucket into `all_in_memory_edges_`
- remap the whole state into `all_in_memory_mapped_edges_`
- optionally build `GegeGraph`

### Active edge materialization

- `gege/src/cpp/src/data/dataloader.cpp`
  - `DataLoader::setActiveEdges`

This currently:

- maps requested active bucket IDs to in-memory bucket offsets
- concatenates the selected buckets into `active_edges`
- applies a full-state `randperm`

## Proposed New Boundary

### New execution unit

`bucket_block = contiguous slice of one active partition-pair edge bucket`

Training should consume a stream of `bucket_block`s instead of one global
`active_edges` tensor per state.

### High-level dataflow

1. generate active bucket IDs from the current buffer state
2. for each bucket:
   - fetch the bucket or bucket block
   - remap endpoints arithmetically
   - batch directly from the remapped block
   - perform local negative sampling/scoring
3. optionally prefetch the next bucket block while the current one trains

This removes the need for:

- `all_in_memory_edges_`
- `all_in_memory_mapped_edges_`
- full-state shuffle

for LP training in the partition-buffer path.

## Phase 0: Measurement

Before changing execution:

measure these subphases in the current code:

### `initializeInMemorySubGraph`
- metadata generation
- edge-bucket load / concat
- remap
- merge-sort / graph build

### `updateInMemorySubGraph_`
- keep/new bucket bookkeeping
- old/new edge copy/load
- remap
- merge-sort / graph build

### `setActiveEdges`
- in-memory bucket lookup
- active-edge gather
- full-state shuffle

If `twitter_16p` shows that `setActiveEdges` plus full-state materialization is a
large fraction of epoch time, bucket streaming is justified.

## Phase 1: Bucket iterator

Add a new partition-buffer LP path:

- `DataLoader::setActiveEdges` no longer builds one concatenated tensor
- instead it builds a queue of bucket descriptors:
  - bucket ID
  - local start
  - local size

This path remains guarded.

Candidate names:

- `GEGE_BUCKET_STREAMING_LP=1`
- `GEGE_BUCKET_STREAMING_BLOCK_SIZE`

## Phase 2: Arithmetic bucket remap

For LP/unfiltered/no-neighbor-sampler:

- load one bucket or block
- remap endpoints from `(global_id -> local_id)` using:
  - `partition_id = global_id / partition_size`
  - `slot = partition_to_buffer_slot[partition_id]`
  - `local_id = slot * partition_size + intra_partition_offset`

No dense global map.

## Phase 3: Bucket-local batching

Batch directly from the bucket stream.

Options:

- fixed-size block batching within one bucket
- small-bucket coalescing across several buckets with the same active state

Important:

- preserve randomness without materializing a full-state permutation
- likely use:
  - random bucket order per epoch
  - random offset or intra-bucket block permutation

## Phase 4: Overlap

While the current bucket block trains:

- prefetch the next bucket block from host memory
- optionally precompute bucket-local negatives

Later extension:

- overlap next-state swap with tail bucket blocks in the current state

## Expected Benefits

### Time

- less state construction
- less remap work
- less graph rebuild work
- no full-state shuffle

### Memory

Current path stores:

- `all_in_memory_edges_`
- `all_in_memory_mapped_edges_`
- `active_edges`
- optional graph structures

Bucket streaming should move this toward:

- one or a few `bucket_block`s in flight

### Scalability

This should matter more on:

- `twitter_16p`
- future `Freebase86M`-scale partition-buffer runs

where state-local host/GPU data movement dominates more than decoder kernels.

## Risks

1. too many tiny buckets can increase control overhead
2. loss of full-state shuffle may hurt training quality if randomness is poor
3. bucket-level batching may reduce negative-pool reuse if blocks are too small

These should be evaluated explicitly.

## Evaluation Plan

### Systems

- epoch runtime
- `swap_update_ms`
- `swap_rebuild_ms`
- new bucket-stream timing breakdown
- peak resident memory

### Scalability

- `fb15k_16p` proxy
- `twitter_16p`
- multi-GPU later on ARC

### Model quality

- MRR / Hits@K against the current partition-buffer baseline

## Immediate Next Step

Do not code the streaming path first.

First:

- instrument the current pipeline
- quantify state-materialization cost on `twitter_16p`

Then:

- implement the guarded bucket iterator path only if the measurements justify
  the boundary change
