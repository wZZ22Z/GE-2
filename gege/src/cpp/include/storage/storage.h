#pragma once

#include <fstream>
#include <string>
#include <tuple>
#include <vector>
#include <iostream>
#include <condition_variable>
#include <memory>
#include <mutex>

#include "common/datatypes.h"
#include "data/batch.h"
#include "storage/buffer.h"

using std::list;
using std::shared_ptr;
using std::string;
using std::unordered_map;
using std::vector;

#define MAX_SHUFFLE_SIZE 4E8
#define MAX_SORT_SIZE 4E8

void renameFile(string old_filename, string new_filename);

void copyFile(string src_filename, string dst_filename);

bool fileExists(string filename);

void createDir(string path, bool exist_ok);

/** Abstract storage class */
class Storage {
   public:
    int64_t dim0_size_;
    int64_t dim1_size_;
    torch::Dtype dtype_;
    bool initialized_;
    vector<int64_t> edge_bucket_sizes_;
    torch::Tensor data_;
    torch::Device device_;
    string filename_;

    Storage();

    virtual ~Storage(){};

    virtual torch::Tensor indexRead(Indices indices) = 0;

    virtual void indexAdd(Indices indices, torch::Tensor values) = 0;

    virtual torch::Tensor range(int64_t offset, int64_t n) = 0;

    virtual void indexPut(Indices indices, torch::Tensor values) = 0;

    virtual void rangePut(int64_t offset, int64_t n, torch::Tensor values) = 0;

    virtual void load() = 0;

    virtual void write() = 0;

    virtual void unload(bool write = false) = 0;

    virtual void shuffle() = 0;

    virtual void sort(bool src) = 0;

    int64_t getDim0() { return dim0_size_; }

    bool isInitialized() { return initialized_; }

    void setInitialized(bool init) { initialized_ = init; }

    void readPartitionSizes(string filename) {
        std::ifstream partition_file(filename);
        edge_bucket_sizes_.clear();
        int64_t size;
        while (partition_file >> size) {
            edge_bucket_sizes_.push_back(size);
        }
    }

    vector<int64_t> getEdgeBucketSizes() { return edge_bucket_sizes_; }
};

class ReusableBarrier {
   public:
    explicit ReusableBarrier(int parties) : parties_(parties), arrived_(0), generation_(0) {}

    void arrive_and_wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        int generation = generation_;
        arrived_++;
        if (arrived_ == parties_) {
            arrived_ = 0;
            generation_++;
            lock.unlock();
            cv_.notify_all();
            return;
        }
        cv_.wait(lock, [&] { return generation_ != generation; });
    }

   private:
    int parties_;
    int arrived_;
    int generation_;
    std::mutex mutex_;
    std::condition_variable cv_;
};


class MemPartitionBufferStorage : public Storage {
   public:
    bool loaded_;

    std::vector<MemPartitionBuffer*> buffers_;
    std::vector<torch::Device> devices_;

    shared_ptr<PartitionBufferOptions> options_;

    MemPartitionBufferStorage(string filename, int64_t dim0_size, int64_t dim1_size, shared_ptr<PartitionBufferOptions> options, std::vector<torch::Device> devices);
    
    ~MemPartitionBufferStorage();

    void rangePut(int64_t offset, torch::Tensor values);

    void append(torch::Tensor values);

    void load() override;

    void unload(bool perform_write) override;

    void unload(bool perform_write, int32_t device_idx);

    void write() override;

    torch::Tensor indexRead(Indices indices) override;

    void indexAdd(Indices indices, torch::Tensor values) override;

    torch::Tensor range(int64_t offset, int64_t n) override;

    void indexPut(Indices indices, torch::Tensor values) override;

    void rangePut(int64_t offset, int64_t n, torch::Tensor values) override;

    void shuffle() override;

    void sort(bool src) override;

    torch::Tensor indexRead(Indices indices, int32_t device_idx);

    void indexAdd(Indices indices, torch::Tensor values, int32_t device_idx);

    Indices getRandomIds(int64_t size, int32_t device_idx = 0) { return buffers_[device_idx]->getRandomIds(size); }

    bool hasSwap(int32_t device_idx = 0) { return buffers_[device_idx]->hasSwap(); }

    void performNextSwap(int32_t device_idx);

    torch::Tensor getGlobalToLocalMap(bool get_current, int32_t device_idx = 0) { return buffers_[device_idx]->getGlobalToLocalMap(get_current); }

    torch::Tensor getPartitionToBufferSlotMap(int32_t device_idx = 0) { return buffers_[device_idx]->getPartitionToBufferSlotMap(); }

    void sync(int device_idx = 0) { buffers_[device_idx]->sync(); }

    void setBufferOrdering(vector<torch::Tensor> buffer_states) { 
        for (int i = 0; i < devices_.size(); i ++) {
            buffers_[i]->setBufferOrdering(buffer_states); 
        }
    }

    void rePartition(torch::Tensor perm, torch::Tensor pos) {
        for(int i = 0; i < devices_.size(); i ++) {
            buffers_[i]->setPermutation(perm, pos);
        }
    }

    std::vector<int> getNextAdmit(int32_t device_idx = 0) { return buffers_[device_idx]->getNextAdmit(); }

    std::vector<int> getNextEvict(int32_t device_idx = 0) { return buffers_[device_idx]->getNextEvict(); }

    int64_t getNumInMemory(int32_t device_idx = 0) { return buffers_[device_idx]->getNumInMemory(); }

    int64_t getPartitionSize(int32_t device_idx = 0) { return buffers_[device_idx]->getPartitionSize(); }
   private:
    int fd_;
    bool peer_relay_runtime_enabled_;
    std::once_flag peer_relay_init_once_;
    std::unique_ptr<ReusableBarrier> peer_relay_ready_barrier_;
    std::unique_ptr<ReusableBarrier> peer_relay_build_barrier_;
    std::vector<torch::Tensor> peer_relay_next_states_;
    std::vector<torch::Tensor> peer_relay_staged_views_;
    void initializePeerRelay_();
    bool peerRelayEnabled_();

};

/** Storage which uses the partition buffer, used for node embeddings and optimizer state */
class PartitionBufferStorage : public Storage {
   public:
    bool loaded_;

    PartitionBuffer *buffer_;

    shared_ptr<PartitionBufferOptions> options_;

    PartitionBufferStorage(string filename, int64_t dim0_size, int64_t dim1_size, shared_ptr<PartitionBufferOptions> options);

    PartitionBufferStorage(string filename, torch::Tensor data, shared_ptr<PartitionBufferOptions> options);

    PartitionBufferStorage(string filename, shared_ptr<PartitionBufferOptions> options);

    ~PartitionBufferStorage();

    void rangePut(int64_t offset, torch::Tensor values);

    void append(torch::Tensor values);

    void load() override;

    void unload(bool perform_write) override;

    void write() override;

    torch::Tensor indexRead(Indices indices) override;

    void indexAdd(Indices indices, torch::Tensor values) override;

    torch::Tensor range(int64_t offset, int64_t n) override;

    void indexPut(Indices indices, torch::Tensor values) override;

    void rangePut(int64_t offset, int64_t n, torch::Tensor values) override;

    void shuffle() override;

    void sort(bool src) override;

    Indices getRandomIds(int64_t size) { return buffer_->getRandomIds(size); }

    bool hasSwap() { return buffer_->hasSwap(); }

    void performNextSwap() { buffer_->performNextSwap(); }

    torch::Tensor getGlobalToLocalMap(bool get_current) { return buffer_->getGlobalToLocalMap(get_current); }

    torch::Tensor getPartitionToBufferSlotMap() { return buffer_->getPartitionToBufferSlotMap(); }

    void sync() { buffer_->sync(); }

    void setBufferOrdering(vector<torch::Tensor> buffer_states) { buffer_->setBufferOrdering(buffer_states); }

    std::vector<int> getNextAdmit() { return buffer_->getNextAdmit(); }

    std::vector<int> getNextEvict() { return buffer_->getNextEvict(); }

    int64_t getNumInMemory() { return buffer_->getNumInMemory(); }

    int64_t getPartitionSize() { return buffer_->getPartitionSize(); }
};

/** Flat File storage used for data that only requires sequential access. Can be used to store and access large amounts of edges. */
class FlatFile : public Storage {
   private:
    int fd_;

    bool loaded_;

   public:
    FlatFile(string filename, int64_t dim0_size, int64_t dim1_size, torch::Dtype dtype, bool alloc = false);

    FlatFile(string filename, torch::Tensor data);

    FlatFile(string filename, torch::Dtype dtype);

    ~FlatFile(){};

    void rangePut(int64_t offset, torch::Tensor values);

    void append(torch::Tensor values);

    void load() override;

    void write() override;

    void unload(bool perform_write) override;

    torch::Tensor indexRead(Indices indices) override;

    void indexAdd(Indices indices, torch::Tensor values) override;

    torch::Tensor range(int64_t offset, int64_t n) override;

    void indexPut(Indices indices, torch::Tensor values) override;

    void rangePut(int64_t offset, int64_t n, torch::Tensor values) override;

    void shuffle() override;

    void sort(bool src) override;

    void move(string new_filename);

    void copy(string new_filename, bool rename);

    void mem_load();

    void mem_unload(bool write);
};

/** In memory storage for data which fits in either GPU or CPU memory. */
class InMemory : public Storage {
   private:
    int fd_;

    bool loaded_;

   public:
    InMemory(string filename, int64_t dim0_size, int64_t dim1_size, torch::Dtype dtype, torch::Device device);

    InMemory(string filename, torch::Tensor data, torch::Device device);

    InMemory(string filename, torch::Dtype dtype);

    InMemory(torch::Tensor data);

    ~InMemory(){};

    void load() override;

    void write() override;

    void unload(bool perform_write) override;

    torch::Tensor indexRead(Indices indices) override;

    void indexAdd(Indices indices, torch::Tensor values) override;

    torch::Tensor range(int64_t offset, int64_t n) override;

    void indexPut(Indices indices, torch::Tensor values) override;

    void rangePut(int64_t offset, int64_t n, torch::Tensor values) override;

    void shuffle() override;

    void sort(bool src) override;
    
    void rePartition(torch::Tensor permutation, int64_t num_nodes, int64_t num_partitions) {
        int64_t partition_size = ceil((double)num_nodes / num_partitions);
 
        auto src_partitions = torch::floor(
                                  permutation.index_select(0, data_.select(1, 0).squeeze())
                                      .to(torch::kFloat64)
                                      .div(static_cast<double>(partition_size)))
                                  .to(torch::kInt64);
        auto dst_partitions = torch::floor(
                                  permutation.index_select(0, data_.select(1, -1).squeeze())
                                      .to(torch::kFloat64)
                                      .div(static_cast<double>(partition_size)))
                                  .to(torch::kInt64);

        auto tup = torch::sort(dst_partitions, -1, true);
        torch::Tensor dst_args = std::get<1>(tup);
        tup = torch::sort(src_partitions.index_select(0, dst_args), -1, true);
        torch::Tensor src_args = std::get<1>(tup);
        data_.copy_(data_.index_select(0, dst_args.index_select(0, src_args)));

        auto edge_bucket_ids_src = torch::floor(
                                       permutation.index_select(0, data_.select(1, 0).squeeze())
                                           .to(torch::kFloat64)
                                           .div(static_cast<double>(partition_size)))
                                       .to(torch::kInt64);
        auto edge_bucket_ids_dst = torch::floor(
                                       permutation.index_select(0, data_.select(1, -1).squeeze())
                                           .to(torch::kFloat64)
                                           .div(static_cast<double>(partition_size)))
                                       .to(torch::kInt64);


        torch::Tensor offsets = torch::zeros({num_partitions, num_partitions}, torch::kInt64);

        auto tup1 = torch::unique_consecutive(edge_bucket_ids_src, false, true);
        torch::Tensor unique_src = std::get<0>(tup1);
        torch::Tensor counts = std::get<2>(tup1);

        torch::Tensor num_source_offsets = torch::cumsum(counts, 0) - counts;
        int32_t curr_src_unique = 0;

        for(int i = 0; i < num_partitions; i ++) {
            if (curr_src_unique < unique_src.size(0) && unique_src[curr_src_unique].item<int64_t>() == i) {
                int64_t offset = num_source_offsets[curr_src_unique].item<int64_t>();
                int64_t num_edges = counts[curr_src_unique].item<int64_t>();
                torch::Tensor dst_ids = edge_bucket_ids_dst.narrow(0, offset, num_edges);
                tup1 = torch::unique_consecutive(dst_ids, false, true);
                torch::Tensor unique_dst = std::get<0>(tup1);
                torch::Tensor counts_dst = std::get<2>(tup1);
                offsets.index_put_({unique_src[curr_src_unique], unique_dst}, counts_dst);
                curr_src_unique += 1;
            }
        }

        offsets = offsets.reshape({-1});
        edge_bucket_sizes_ = std::vector<int64_t>(offsets.data_ptr<int64_t>(), offsets.data_ptr<int64_t>() + offsets.numel());
        // std::cout << "edge_bucket_sizes_ size() " << edge_bucket_sizes_.size() << std::endl;
        // for(int i = 0; i < 16; i ++) {
        //     for(int j = 0; j < 16; j ++) {
        //         std::cout << edge_bucket_sizes_[i * 16 + j] << " ";
        //     }
        //     std::cout << std::endl;
        // }
    }
};
