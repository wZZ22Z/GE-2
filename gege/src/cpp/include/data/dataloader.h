#pragma once

#include <atomic>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "batch.h"
#include "common/datatypes.h"
#include "configuration/config.h"
#include "data/samplers/edge.h"
#include "data/samplers/negative.h"
#include "data/samplers/neighbor.h"
#include "storage/graph_storage.h"
#include "storage/storage.h"

struct DataLoaderPerfStats {
    int64_t swap_barrier_wait_ns = 0;
    int64_t swap_update_ns = 0;
    int64_t swap_rebuild_ns = 0;
    int64_t swap_sync_wait_ns = 0;
    int64_t swap_count = 0;
    int64_t get_next_batch_ns = 0;
    int64_t edge_sample_ns = 0;
    int64_t edge_get_edges_ns = 0;
    int64_t edge_negative_sample_ns = 0;
    int64_t edge_map_collect_ids_ns = 0;
    int64_t edge_map_lookup_ns = 0;
    int64_t edge_map_verify_ns = 0;
    int64_t edge_remap_assign_ns = 0;
    int64_t edge_finalize_ns = 0;
    int64_t node_sample_ns = 0;
    int64_t load_cpu_parameters_ns = 0;
    int64_t get_batch_device_prepare_ns = 0;
    int64_t get_batch_perform_map_ns = 0;
    int64_t get_batch_overhead_ns = 0;
    NegativeSamplerPerfStats negative_sampler;
    std::vector<int64_t> device_swap_barrier_wait_ns;
    std::vector<int64_t> device_swap_update_ns;
    std::vector<int64_t> device_swap_rebuild_ns;
    std::vector<int64_t> device_swap_sync_wait_ns;
    std::vector<int64_t> device_swap_count;
    std::vector<int64_t> device_get_next_batch_ns;
    std::vector<int64_t> device_edge_sample_ns;
    std::vector<int64_t> device_edge_get_edges_ns;
    std::vector<int64_t> device_edge_negative_sample_ns;
    std::vector<int64_t> device_edge_map_collect_ids_ns;
    std::vector<int64_t> device_edge_map_lookup_ns;
    std::vector<int64_t> device_edge_map_verify_ns;
    std::vector<int64_t> device_edge_remap_assign_ns;
    std::vector<int64_t> device_edge_finalize_ns;
    std::vector<int64_t> device_node_sample_ns;
    std::vector<int64_t> device_load_cpu_parameters_ns;
    std::vector<int64_t> device_get_batch_device_prepare_ns;
    std::vector<int64_t> device_get_batch_perform_map_ns;
    std::vector<int64_t> device_get_batch_overhead_ns;
    std::vector<std::vector<int64_t>> device_swap_active_bucket_samples;
    std::vector<std::vector<int64_t>> device_swap_active_edge_samples;
    std::vector<std::vector<int64_t>> device_swap_batch_count_samples;
    std::vector<std::vector<int64_t>> device_swap_rebuild_samples_ns;
};

class DataLoader {
   public:
    bool train_;
    int epochs_processed_;
    int64_t batches_processed_;
    int64_t current_edge_;
    std::mutex *sampler_lock_;
    std::vector<std::vector<shared_ptr<Batch>>> all_batches_;
    int batch_size_;

    bool single_dataset_;

    int batch_id_offset_;
    std::atomic<int32_t> async_barrier;
    std::vector<std::vector<shared_ptr<Batch>>::iterator> batch_iterators_;
    
    int loaded_subgraphs;
    int32_t false_negative_edges;

    std::atomic<int32_t> swap_tasks_completed;
    std::mutex *batch_lock_;
    std::condition_variable *batch_cv_;
    bool waiting_for_batches_;
    std::vector<int> batches_left_;
    int total_batches_processed_;
    std::vector<bool> all_reads_;
    
    // record the number of devices that have finished the current batches.
    std::atomic<int64_t> activate_devices_;

    vector<torch::Tensor> buffer_states_;

    vector<torch::Device> devices_;

    // Link prediction
    vector<torch::Tensor> edge_buckets_per_buffer_;
    vector<vector<torch::Tensor>::iterator> edge_buckets_per_buffer_iterators_;

    // Node classification
    vector<torch::Tensor> node_ids_per_buffer_;
    vector<torch::Tensor>::iterator node_ids_per_buffer_iterator_;

    shared_ptr<NeighborSampler> training_neighbor_sampler_;
    shared_ptr<NeighborSampler> evaluation_neighbor_sampler_;

    shared_ptr<NegativeSampler> training_negative_sampler_;
    shared_ptr<NegativeSampler> evaluation_negative_sampler_;

    Timestamp timestamp_;

    shared_ptr<GraphModelStorage> graph_storage_;

    shared_ptr<EdgeSampler> edge_sampler_;
    shared_ptr<NegativeSampler> negative_sampler_;
    shared_ptr<NeighborSampler> neighbor_sampler_;

    shared_ptr<TrainingConfig> training_config_;
    shared_ptr<EvaluationConfig> evaluation_config_;
    bool only_root_features_;
    bool use_inverse_relations_;
    std::atomic<int64_t> swap_barrier_wait_ns_{0};
    std::atomic<int64_t> swap_update_ns_{0};
    std::atomic<int64_t> swap_rebuild_ns_{0};
    std::atomic<int64_t> swap_sync_wait_ns_{0};
    std::atomic<int64_t> swap_count_{0};
    std::atomic<int64_t> get_next_batch_ns_{0};
    std::atomic<int64_t> edge_sample_ns_{0};
    std::atomic<int64_t> edge_get_edges_ns_{0};
    std::atomic<int64_t> edge_negative_sample_ns_{0};
    std::atomic<int64_t> edge_map_collect_ids_ns_{0};
    std::atomic<int64_t> edge_map_lookup_ns_{0};
    std::atomic<int64_t> edge_map_verify_ns_{0};
    std::atomic<int64_t> edge_remap_assign_ns_{0};
    std::atomic<int64_t> edge_finalize_ns_{0};
    std::atomic<int64_t> node_sample_ns_{0};
    std::atomic<int64_t> load_cpu_parameters_ns_{0};
    std::atomic<int64_t> get_batch_device_prepare_ns_{0};
    std::atomic<int64_t> get_batch_perform_map_ns_{0};
    std::atomic<int64_t> get_batch_overhead_ns_{0};
    std::vector<int64_t> device_swap_barrier_wait_ns_;
    std::vector<int64_t> device_swap_update_ns_;
    std::vector<int64_t> device_swap_rebuild_ns_;
    std::vector<int64_t> device_swap_sync_wait_ns_;
    std::vector<int64_t> device_swap_count_;
    std::vector<int64_t> device_get_next_batch_ns_;
    std::vector<int64_t> device_edge_sample_ns_;
    std::vector<int64_t> device_edge_get_edges_ns_;
    std::vector<int64_t> device_edge_negative_sample_ns_;
    std::vector<int64_t> device_edge_map_collect_ids_ns_;
    std::vector<int64_t> device_edge_map_lookup_ns_;
    std::vector<int64_t> device_edge_map_verify_ns_;
    std::vector<int64_t> device_edge_remap_assign_ns_;
    std::vector<int64_t> device_edge_finalize_ns_;
    std::vector<int64_t> device_node_sample_ns_;
    std::vector<int64_t> device_load_cpu_parameters_ns_;
    std::vector<int64_t> device_get_batch_device_prepare_ns_;
    std::vector<int64_t> device_get_batch_perform_map_ns_;
    std::vector<int64_t> device_get_batch_overhead_ns_;
    std::vector<std::vector<int64_t>> device_swap_active_bucket_samples_;
    std::vector<std::vector<int64_t>> device_swap_active_edge_samples_;
    std::vector<std::vector<int64_t>> device_swap_batch_count_samples_;
    std::vector<std::vector<int64_t>> device_swap_rebuild_samples_ns_;
    std::vector<int64_t> device_current_state_index_;
    std::vector<int64_t> device_current_active_bucket_count_;
    std::vector<int64_t> device_current_active_edge_count_;
    std::vector<std::string> device_current_state_partitions_;
    std::vector<int64_t> device_state_build_sequence_;

    LearningTask learning_task_;

    NegativeSamplingMethod negative_sampling_method_;

    DataLoader(shared_ptr<GraphModelStorage> graph_storage, LearningTask learning_task, shared_ptr<TrainingConfig> training_config,
               shared_ptr<EvaluationConfig> evaluation_config, shared_ptr<EncoderConfig> encoder_config, vector<torch::Device> devices,
               NegativeSamplingMethod nsm = NegativeSamplingMethod::OTHER, bool use_inverse_relations = true);

    DataLoader(shared_ptr<GraphModelStorage> graph_storage, LearningTask learning_task, int batch_size, shared_ptr<NegativeSampler> negative_sampler = nullptr,
               shared_ptr<NeighborSampler> neighbor_sampler = nullptr, bool train = false);

    ~DataLoader();

    void setBufferOrdering();

    void setActiveEdges(int32_t device_idx = 0);

    void setActiveNodes();

    void initializeBatches(bool prepare_encode = false, int32_t device_idx = 0);

    void refreshGraphStorageMode();

    void clearBatches();

    /**
     * Check to see whether another batch exists.
     * @return True if batch exists, false if not
     */
    bool hasNextBatch(int32_t device_idx = 0);

    shared_ptr<Batch> getNextBatch(int32_t device_idx = 0);

    /**
     * Notify that the batch has been completed. Used for concurrency control.
     */
    void finishedBatch(int32_t device_idx = 0);

    /**
     * Gets the next batch.
     * Loads edges from storage
     * Constructs negative negative edges
     * Loads CPU embedding parameters
     * @return The next batch
     */
    shared_ptr<Batch> getBatch(at::optional<torch::Device> device = c10::nullopt, bool perform_map = false, int32_t device_idx = 0);

    /**
     * Loads edges and samples negatives to construct a batch
     * @param batch: Batch object to load edges into.
     */
    void edgeSample(shared_ptr<Batch> batch, int32_t device_idx = 0);

    /**
     * Creates a mapping from global node ids into batch local node ids
     * @param batch: Batch to map
     */
    void mapEdges(shared_ptr<Batch> batch, bool use_negs, bool use_nbrs, bool set_map);

    /**
     * Loads edges and samples negatives to construct a batch
     * @param batch: Batch object to load nodes into.
     */
    void nodeSample(shared_ptr<Batch> batch, int32_t device_idx = 0);

    /**
     * Samples negatives for the batch using the dataloader's negative sampler
     * @param batch: Batch object to load negative samples into.
     */
    void negativeSample(shared_ptr<Batch> batch, int32_t device_idx = 0);

    /**
     * Loads CPU parameters into batch
     * @param batch: Batch object to load parameters into.
     */
    void loadCPUParameters(shared_ptr<Batch> batch);

    /**
     * Loads GPU parameters into batch
     * @param batch Batch object to load parameters into.
     */
    void loadGPUParameters(shared_ptr<Batch> batch, int device_idx = 0);

    /**
     * Applies gradient updates to underlying storage
     * @param batch: Batch object to apply updates from.
     * @param gpu: If true, only the gpu parameters will be updated.
     */
    void updateEmbeddings(shared_ptr<Batch> batch, bool gpu, int32_t device_idx = 0);

    void updateEmbeddingsG(shared_ptr<Batch> batch, bool gpu, int32_t device_idx = 0);

    /**
     * Notify that the epoch has been completed. Prepares dataset for a new epoch.
     */
    void nextEpoch();

    void resetPerfStats();

    DataLoaderPerfStats getPerfStats() const;

    /**
     * Load graph from storage.
     */
    void loadStorage();

    bool epochComplete(int32_t device_idx = 0) { return (batches_left_[device_idx] == 0) && all_reads_[device_idx]; }

    /**
     * Unload graph from storage.
     * @param write Set to true to write embedding table state to disk
     */
    void unloadStorage(bool write = false) { graph_storage_->unload(write); }

    /**
     * Gets the number of edges from the graph storage.
     * @return Number of edges in the graph
     */
    int64_t getNumEdges() { return graph_storage_->getNumEdges(); }

    int64_t getEpochsProcessed() { return epochs_processed_; }

    int64_t getBatchesProcessed() { return batches_processed_; }

    bool isTrain() { return train_; }

    /**
     * Sets graph storage, negative sampler, and neighbor sampler to training set.
     */
    void setTrainSet() {
        if (single_dataset_) {
            throw GegeRuntimeException("This dataloader only has a single dataset and cannot switch");
        } else {
            batch_size_ = training_config_->batch_size;
            train_ = true;
            loaded_subgraphs = 0;
            async_barrier = 0;
            graph_storage_->setTrainSet();
            negative_sampler_ = training_negative_sampler_;
            neighbor_sampler_ = training_neighbor_sampler_;
            refreshGraphStorageMode();
            loadStorage();
        }
    }

    /**
     * Sets graph storage, negative sampler, and neighbor sampler to validation set.
     */
    void setValidationSet() {
        if (single_dataset_) {
            throw GegeRuntimeException("This dataloader only has a single dataset and cannot switch");
        } else {
            batch_size_ = evaluation_config_->batch_size;
            train_ = false;
            graph_storage_->setValidationSet();
            negative_sampler_ = evaluation_negative_sampler_;
            neighbor_sampler_ = evaluation_neighbor_sampler_;
            refreshGraphStorageMode();
            loadStorage();
        }
    }

    void setTestSet() {
        if (single_dataset_) {
            throw GegeRuntimeException("This dataloader only has a single dataset and cannot switch");
        } else {
            batch_size_ = evaluation_config_->batch_size;
            train_ = false;
            graph_storage_->setTestSet();
            negative_sampler_ = evaluation_negative_sampler_;
            neighbor_sampler_ = evaluation_neighbor_sampler_;
            refreshGraphStorageMode();
            loadStorage();
        }
    }

    void setEncode() {
        if (single_dataset_) {
            loadStorage();
            initializeBatches(true);
        } else {
            batch_size_ = evaluation_config_->batch_size;
            train_ = false;
            graph_storage_->setTrainSet();
            neighbor_sampler_ = evaluation_neighbor_sampler_;
            negative_sampler_ = nullptr;
            refreshGraphStorageMode();
            loadStorage();
            initializeBatches(true);
        }
    }
};
