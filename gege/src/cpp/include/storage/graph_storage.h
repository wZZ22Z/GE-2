#pragma once

#include "configuration/constants.h"
#include "storage/storage.h"

struct GraphModelStoragePtrs {
    shared_ptr<Storage> edges = nullptr;
    shared_ptr<Storage> train_edges = nullptr;
    shared_ptr<Storage> validation_edges = nullptr;
    shared_ptr<Storage> test_edges = nullptr;
    shared_ptr<Storage> nodes = nullptr;
    shared_ptr<Storage> train_nodes = nullptr;
    shared_ptr<Storage> valid_nodes = nullptr;
    shared_ptr<Storage> test_nodes = nullptr;
    shared_ptr<Storage> node_features = nullptr;
    shared_ptr<Storage> node_labels = nullptr;
    shared_ptr<Storage> relation_features = nullptr;
    shared_ptr<Storage> relation_labels = nullptr;
    shared_ptr<Storage> node_embeddings = nullptr;
    shared_ptr<Storage> node_embeddings_g = nullptr;
    shared_ptr<Storage> encoded_nodes = nullptr;
    shared_ptr<Storage> node_optimizer_state = nullptr;
    shared_ptr<Storage> node_optimizer_state_g = nullptr;
    std::vector<shared_ptr<Storage>> filter_edges;
};

struct InMemorySubgraphState {
    EdgeList all_in_memory_edges_;
    EdgeList all_in_memory_mapped_edges_;
    torch::Tensor in_memory_partition_ids_;
    torch::Tensor in_memory_edge_bucket_ids_;
    torch::Tensor in_memory_edge_bucket_sizes_;
    torch::Tensor in_memory_edge_bucket_starts_;
    torch::Tensor global_to_local_index_map_;
    shared_ptr<GegeGraph> in_memory_subgraph_;
};

class GraphModelStorage {
   private:
    void _load(shared_ptr<Storage> storage);

    void _unload(shared_ptr<Storage> storage, bool write);

    bool shouldUsePartitionBufferLPFastPath_();

    torch::Tensor getPartitionToBufferSlotMap_(int32_t device_idx = 0);

    int64_t getPartitionSize_(int32_t device_idx = 0);

    torch::Tensor getGlobalToLocalMapForValidation_(bool get_current, int32_t device_idx = 0);

    torch::Tensor mapEdgesWithDenseMap_(torch::Tensor edges, torch::Tensor global_to_local_index_map, torch::Device device);

    torch::Tensor mapEdgesWithPartitionSlots_(torch::Tensor edges, torch::Tensor partition_to_buffer_slot, int64_t partition_size,
                                              torch::Device device);

    int64_t num_nodes_;
    int64_t num_edges_;
    bool partition_buffer_lp_fast_path_enabled_;

   protected:
    bool train_;

    shared_ptr<InMemory> in_memory_embeddings_;
    shared_ptr<InMemory> in_memory_features_;

   public:
    // In memory subgraph for partition buffer

    std::vector<EdgeList> active_edges_;
    std::vector<torch::Device> devices_;
    Indices active_nodes_;
    torch::Tensor perm_;

    std::mutex *subgraph_lock_;
    std::condition_variable *subgraph_cv_;
    shared_ptr<InMemorySubgraphState> current_subgraph_state_;
    std::vector<shared_ptr<InMemorySubgraphState>> current_subgraph_states_;
    shared_ptr<InMemorySubgraphState> next_subgraph_state_;
    bool prefetch_;
    bool prefetch_complete_;

    GraphModelStoragePtrs storage_ptrs_;
    bool full_graph_evaluation_;

    GraphModelStorage(GraphModelStoragePtrs storage_ptrs, shared_ptr<StorageConfig> storage_config);

    GraphModelStorage(GraphModelStoragePtrs storage_ptrs, bool prefetch = false);

    ~GraphModelStorage();

    void load();

    void load_g();

    void unload(bool write);

    void initializeInMemorySubGraph(torch::Tensor buffer_state, torch::Device device = torch::kCPU, int32_t device_idx = 0);

    void updateInMemorySubGraph_(shared_ptr<InMemorySubgraphState> subgraph, std::pair<std::vector<int>, std::vector<int>> swap_ids, int32_t device_idx = 0);

    void updateInMemorySubGraph(int32_t device_idx = 0);

    void getNextSubGraph();

    EdgeList merge_sorted_edge_buckets(EdgeList edges, torch::Tensor starts, int buffer_size, bool src);

    void setEdgesStorage(shared_ptr<Storage> edge_storage);

    void setNodesStorage(shared_ptr<Storage> node_storage);

    EdgeList getEdges(Indices indices, int32_t device_idx = 0);

    EdgeList getEdgesRange(int64_t start, int64_t size, int32_t device_idx = 0);

    Indices getRandomNodeIds(int64_t size);

    Indices getNodeIdsRange(int64_t start, int64_t size);

    void shuffleEdges();

    torch::Tensor getNodeEmbeddings(Indices indices, int32_t device_idx = 0);

    torch::Tensor getNodeEmbeddingsG(Indices indices, int32_t device_idx = 0);

    torch::Tensor getNodeEmbeddingsRange(int64_t start, int64_t size);

    torch::Tensor getNodeFeatures(Indices indices);

    torch::Tensor getNodeFeaturesRange(int64_t start, int64_t size);

    torch::Tensor getEncodedNodes(Indices indices);

    torch::Tensor getEncodedNodesRange(int64_t start, int64_t size);

    torch::Tensor getNodeLabels(Indices indices);

    torch::Tensor getNodeLabelsRange(int64_t start, int64_t size);

    void updatePutNodeEmbeddings(Indices indices, torch::Tensor values);

    void updateAddNodeEmbeddings(Indices indices, torch::Tensor values, int32_t device_idx = 0);

    void updateAddNodeEmbeddingsG(Indices indices, torch::Tensor values, int32_t device_idx = 0);

    void updatePutEncodedNodes(Indices indices, torch::Tensor values);

    void updatePutEncodedNodesRange(int64_t start, int64_t size, torch::Tensor values);

    OptimizerState getNodeEmbeddingState(Indices indices, int32_t device_idx = 0);

    OptimizerState getNodeEmbeddingStateG(Indices indices, int32_t device_idx = 0);

    OptimizerState getNodeEmbeddingStateRange(int64_t start, int64_t size);

    void updatePutNodeEmbeddingState(Indices indices, OptimizerState state);

    void updateAddNodeEmbeddingState(Indices indices, torch::Tensor values, int32_t device_idx = 0);

    void updateAddNodeEmbeddingStateG(Indices indices, torch::Tensor values, int32_t device_idx = 0);

    bool embeddingsOffDevice();

    bool embeddingsOffDeviceG();

    void sortAllEdges(int32_t device_idx = 0);

    int getNumPartitions() {
        int num_partitions = 1;

        if (useInMemorySubGraph()) {
            if (instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_features)) {
                num_partitions = std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_features)->options_->num_partitions;
            }

            // assumes both the node features and node embeddings have the same number of partitions
            if (instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_embeddings)) {
                num_partitions = std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_embeddings)->options_->num_partitions;
            }
            if (instance_of<Storage, MemPartitionBufferStorage>(storage_ptrs_.node_embeddings)) {
                num_partitions = std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_embeddings)->options_->num_partitions;
            }
        }

        return num_partitions;
    }

    void rePartition() {
        if (instance_of<Storage, MemPartitionBufferStorage>(storage_ptrs_.node_embeddings)) { 
            auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
            torch::Tensor perm = torch::randperm(getNumNodes(), opts);
            auto tup = torch::sort(perm); 
            torch::Tensor pos = std::get<1>(tup);
            std::dynamic_pointer_cast<InMemory>(storage_ptrs_.edges)->rePartition(perm, getNumNodes(), getNumPartitions());
            std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_embeddings)->rePartition(perm, pos);
            std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_optimizer_state)->rePartition(perm, pos);
        }
    }

    bool useInMemorySubGraph() {
        bool embeddings_buffered = instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_embeddings);
        embeddings_buffered = embeddings_buffered || instance_of<Storage, MemPartitionBufferStorage>(storage_ptrs_.node_embeddings);
        bool features_buffered = instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_features);

        return (embeddings_buffered || features_buffered) && (train_ || (!full_graph_evaluation_));
    }

    void setPartitionBufferLPFastPathEnabled(bool enabled) { partition_buffer_lp_fast_path_enabled_ = enabled; }

    bool partitionBufferLPFastPathEnabled() { return shouldUsePartitionBufferLPFastPath_(); }

    bool hasSwap(int32_t device_idx = 0) {
        if (storage_ptrs_.node_embeddings != nullptr) {
            if (instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_embeddings)) {
                return std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_embeddings)->hasSwap();
            }
            if (instance_of<Storage, MemPartitionBufferStorage>(storage_ptrs_.node_embeddings)) {
                return std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_embeddings)->hasSwap(device_idx);
            }
        }

        if (storage_ptrs_.node_features != nullptr) {
            return std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_features)->hasSwap();
        }

        return false;
    }

    std::pair<std::vector<int>, std::vector<int>> getNextSwapIds(int32_t device_idx = 0) {
        std::vector<int> evict_ids;
        std::vector<int> admit_ids;

        if (storage_ptrs_.node_embeddings != nullptr && instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_embeddings)) {
            evict_ids = std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_embeddings)->getNextEvict();
            admit_ids = std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_embeddings)->getNextAdmit();
        } else if (storage_ptrs_.node_features != nullptr && instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_features)) {
            evict_ids = std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_features)->getNextEvict();
            admit_ids = std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_features)->getNextAdmit();
        } else if (storage_ptrs_.node_embeddings != nullptr && instance_of<Storage, MemPartitionBufferStorage>(storage_ptrs_.node_embeddings)) {
            evict_ids = std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_embeddings)->getNextEvict(device_idx);
            admit_ids = std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_embeddings)->getNextAdmit(device_idx);
        }

        return std::make_pair(evict_ids, admit_ids);
    }

    void performSwap(int32_t device_idx = 0) {
        if (storage_ptrs_.node_embeddings != nullptr && instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_embeddings)) {
            std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_embeddings)->performNextSwap();
            if (storage_ptrs_.node_optimizer_state != nullptr && train_) {
                std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_optimizer_state)->performNextSwap();
            }
        }

        if (storage_ptrs_.node_embeddings != nullptr && instance_of<Storage, MemPartitionBufferStorage>(storage_ptrs_.node_embeddings)) {
            std::vector<std::thread> threads;
            threads.push_back(std::thread(&MemPartitionBufferStorage::performNextSwap, std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_embeddings), device_idx));
            if (storage_ptrs_.node_optimizer_state != nullptr && train_) {
                threads.push_back(std::thread(&MemPartitionBufferStorage::performNextSwap, std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_optimizer_state), device_idx));
            }
            for(auto& thread : threads) {
                thread.join();
            }

            // std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_embeddings)->performNextSwap(device_idx);
            // if (storage_ptrs_.node_optimizer_state != nullptr && train_) {
            //     std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_optimizer_state)->performNextSwap(device_idx);
            // }
        }

        if (storage_ptrs_.node_embeddings_g != nullptr && instance_of<Storage, MemPartitionBufferStorage>(storage_ptrs_.node_embeddings_g)) {
            std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_embeddings_g)->performNextSwap(device_idx);
            if (storage_ptrs_.node_optimizer_state_g != nullptr && train_) {
                std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_optimizer_state_g)->performNextSwap(device_idx);
            }
        }

        if (storage_ptrs_.node_features != nullptr && instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_features)) {
            std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_features)->performNextSwap();
        }
    }

    void setBufferOrdering(vector<torch::Tensor> buffer_states) {
        if (storage_ptrs_.node_embeddings != nullptr && (instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_embeddings))) {
            std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_embeddings)->setBufferOrdering(buffer_states);
            if (storage_ptrs_.node_optimizer_state != nullptr && train_) {
                std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_optimizer_state)->setBufferOrdering(buffer_states);
            }
        }
        if (storage_ptrs_.node_embeddings != nullptr && (instance_of<Storage, MemPartitionBufferStorage>(storage_ptrs_.node_embeddings))) {
            std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_embeddings)->setBufferOrdering(buffer_states);
            if (storage_ptrs_.node_optimizer_state != nullptr && train_) {
                std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_optimizer_state)->setBufferOrdering(buffer_states);
            }
        }
        if (storage_ptrs_.node_embeddings_g != nullptr && (instance_of<Storage, MemPartitionBufferStorage>(storage_ptrs_.node_embeddings_g))) {
            std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_embeddings_g)->setBufferOrdering(buffer_states);
            if (storage_ptrs_.node_optimizer_state_g != nullptr && train_) {
                std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_optimizer_state_g)->setBufferOrdering(buffer_states);
            }
        }
        if (storage_ptrs_.node_features != nullptr && instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_features)) {
            std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_features)->setBufferOrdering(buffer_states);
        }
    }

    void setActiveEdges(torch::Tensor active_edges, int32_t device_idx) { 
        active_edges_[device_idx] = active_edges;
    }

    void setActiveNodes(torch::Tensor node_ids) { active_nodes_ = node_ids; }

    int64_t getNumActiveEdges(int device_idx = 0) {
        if (active_edges_[device_idx].defined()) {
            return active_edges_[device_idx].size(0);
        } else {
            return storage_ptrs_.edges->getDim0();
        }
    }

    int64_t getNumActiveNodes() {
        if (active_nodes_.defined()) {
            return active_nodes_.size(0);
        } else {
            return storage_ptrs_.nodes->getDim0();
        }
    }

    int64_t getNumEdges() { return storage_ptrs_.edges->getDim0(); }

    int64_t getNumNodes() {
        if (storage_ptrs_.node_embeddings != nullptr) {
            return storage_ptrs_.node_embeddings->getDim0();
        }

        if (storage_ptrs_.node_features != nullptr) {
            return storage_ptrs_.node_features->getDim0();
        }

        return num_nodes_;
    }

    int64_t getNumNodesInMemory(int32_t device_idx = 0) {
        if (storage_ptrs_.node_embeddings != nullptr) {
            if (useInMemorySubGraph()) {
                if (instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_embeddings))
                    return std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_embeddings)->getNumInMemory();
                if (instance_of<Storage, MemPartitionBufferStorage>(storage_ptrs_.node_embeddings))
                    return std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_embeddings)->getNumInMemory(device_idx);
            }
        }

        if (storage_ptrs_.node_features != nullptr) {
            if (useInMemorySubGraph()) {
                return std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_features)->getNumInMemory();
            }
        }

        return getNumNodes();
    }

    void setTrainSet() {
        train_ = true;
        
        if (instance_of<Storage, MemPartitionBufferStorage>(storage_ptrs_.node_embeddings)) {
            storage_ptrs_.node_embeddings->device_ = torch::kCUDA;
        }

        if (storage_ptrs_.node_embeddings_g != nullptr && instance_of<Storage, MemPartitionBufferStorage>(storage_ptrs_.node_embeddings_g)) {
            storage_ptrs_.node_embeddings_g->device_ = torch::kCUDA;
        }

        if (storage_ptrs_.train_edges != nullptr) {
            setEdgesStorage(storage_ptrs_.train_edges);
        }

        if (storage_ptrs_.train_nodes != nullptr) {
            setNodesStorage(storage_ptrs_.train_nodes);
        }
    }

    void setValidationSet() {
        train_ = false;

        if (instance_of<Storage, MemPartitionBufferStorage>(storage_ptrs_.node_embeddings)) {
            storage_ptrs_.node_embeddings->device_ = torch::kCPU;
        }

        if (storage_ptrs_.validation_edges != nullptr) {
            setEdgesStorage(storage_ptrs_.validation_edges);
        }

        if (storage_ptrs_.valid_nodes != nullptr) {
            setNodesStorage(storage_ptrs_.valid_nodes);
        }
    }

    void setTestSet() {
        train_ = false;
        
        if (instance_of<Storage, MemPartitionBufferStorage>(storage_ptrs_.node_embeddings)) {
            storage_ptrs_.node_embeddings->device_ = torch::kCPU;
        }

        if (storage_ptrs_.test_edges != nullptr) {
            setEdgesStorage(storage_ptrs_.test_edges);
        }

        if (storage_ptrs_.test_nodes != nullptr) {
            setNodesStorage(storage_ptrs_.test_nodes);
        }
    }

    void setFilterEdges(std::vector<shared_ptr<Storage>> filter_edges) { storage_ptrs_.filter_edges = filter_edges; }

    void addFilterEdges(shared_ptr<Storage> filter_edges) { storage_ptrs_.filter_edges.emplace_back(filter_edges); }
};
