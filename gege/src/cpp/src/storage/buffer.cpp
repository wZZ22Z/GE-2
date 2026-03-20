#include "storage/buffer.h"

#include <common/util.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <fcntl.h>
#include <unistd.h>
#include <cstdlib>
#include <iostream>
#include <string>

#include <functional>
#include <future>
#include <shared_mutex>
#ifdef GEGE_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#include "pytorch_scatter/segment_sum.h"
#endif

#include "configuration/constants.h"
#include "reporting/logger.h"

#if defined(GEGE_CUDA)
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define GEGE_HAS_NVTX 1
#elif __has_include(<nvToolsExt.h>)
#include <nvToolsExt.h>
#define GEGE_HAS_NVTX 1
#else
#define GEGE_HAS_NVTX 0
#endif
#else
#define GEGE_HAS_NVTX 0
#endif

namespace {

bool parse_env_flag(const char *name, bool default_value) {
    const char *raw = std::getenv(name);
    if (raw == nullptr) {
        return default_value;
    }

    std::string value(raw);
    if (value == "0" || value == "false" || value == "False" || value == "FALSE") {
        return false;
    }

    if (value == "1" || value == "true" || value == "True" || value == "TRUE") {
        return true;
    }

    return default_value;
}

bool csr_update_enabled() {
    static bool enabled = parse_env_flag("GEGE_CSR_UPDATE", true);
    return enabled;
}

bool csr_update_reduce_enabled() {
    static bool enabled = parse_env_flag("GEGE_CSR_UPDATE_REDUCE", false);
    return enabled;
}

bool csr_nvtx_enabled() {
    static bool enabled = parse_env_flag("GEGE_CSR_NVTX", false);
    return enabled;
}

int64_t parse_env_int(const char *name, int64_t default_value);

bool partition_buffer_swap_timing_enabled() {
    static bool enabled = parse_env_flag("GEGE_PARTITION_BUFFER_SWAP_TIMING", false);
    return enabled;
}

int64_t partition_buffer_swap_timing_max() {
    static int64_t max_swaps = std::max<int64_t>(parse_env_int("GEGE_PARTITION_BUFFER_SWAP_TIMING_MAX", 128), 0);
    return max_swaps;
}

std::atomic<int64_t> &partition_buffer_swap_timing_counter() {
    static std::atomic<int64_t> counter{0};
    return counter;
}

bool should_log_partition_buffer_swap_timing(int64_t &timing_id) {
    if (!partition_buffer_swap_timing_enabled()) {
        return false;
    }
    timing_id = partition_buffer_swap_timing_counter().fetch_add(1);
    return timing_id < partition_buffer_swap_timing_max();
}

int64_t parse_env_int(const char *name, int64_t default_value) {
    const char *raw = std::getenv(name);
    if (raw == nullptr) {
        return default_value;
    }

    try {
        return std::stoll(std::string(raw));
    } catch (...) {
        return default_value;
    }
}

bool stage_debug_enabled() {
    static bool enabled = parse_env_flag("GEGE_STAGE_DEBUG", false);
    return enabled;
}

int64_t stage_debug_max_updates() {
    static int64_t max_updates = parse_env_int("GEGE_STAGE_DEBUG_MAX_UPDATES", 40);
    return std::max<int64_t>(max_updates, 0);
}

std::atomic<int64_t> &stage_debug_counter() {
    static std::atomic<int64_t> counter{0};
    return counter;
}

bool should_run_stage_debug(int64_t &debug_update_id) {
    if (!stage_debug_enabled()) {
        return false;
    }
    debug_update_id = stage_debug_counter().fetch_add(1);
    return debug_update_id < stage_debug_max_updates();
}

double elapsed_ms(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

class ScopedNvtxRange {
   public:
    explicit ScopedNvtxRange(const char *name) {
        active_ = false;
#if GEGE_HAS_NVTX
        if (csr_nvtx_enabled()) {
            nvtxRangePushA(name);
            active_ = true;
        }
#endif
    }

    ~ScopedNvtxRange() {
#if GEGE_HAS_NVTX
        if (active_) {
            nvtxRangePop();
        }
#endif
    }

   private:
    bool active_;
};

#ifdef GEGE_CUDA
std::tuple<torch::Tensor, torch::Tensor> reduce_updates_with_csr(torch::Tensor indices, torch::Tensor values) {
    ScopedNvtxRange nvtx_scope("buffer.reduce_updates_with_csr");

    if (!indices.defined() || !values.defined() || indices.numel() == 0 || values.numel() == 0) {
        return std::forward_as_tuple(indices, values);
    }

    torch::Tensor indices64 = indices.to(torch::kInt64);
    torch::Tensor permutation = torch::argsort(indices64);
    torch::Tensor sorted_indices = indices64.index_select(0, permutation);

    auto unique_tup = torch::unique_consecutive(sorted_indices, false, true);
    torch::Tensor unique_indices = std::get<0>(unique_tup);
    torch::Tensor counts = std::get<2>(unique_tup).to(torch::kInt64);

    if (unique_indices.numel() == sorted_indices.numel()) {
        return std::forward_as_tuple(indices64, values);
    }

    torch::Tensor sorted_values = values.index_select(0, permutation);
    auto indptr_opts = torch::TensorOptions().dtype(torch::kInt64).device(indices.device());
    torch::Tensor indptr = torch::zeros({unique_indices.numel() + 1}, indptr_opts);
    if (counts.numel() > 0) {
        indptr.narrow(0, 1, counts.numel()).copy_(counts.cumsum(0));
    }

    torch::Tensor reduced_values = segment_sum_csr(sorted_values, indptr, torch::nullopt);
    return std::forward_as_tuple(unique_indices, reduced_values);
}
#endif
}  // namespace

Partition::Partition(int partition_id, int64_t partition_size, int embedding_size, torch::Dtype dtype, int64_t idx_offset, int64_t file_offset) {
    lock_ = new std::mutex();
    cv_ = new std::condition_variable();
    data_ptr_ = nullptr;
    partition_id_ = partition_id;

    present_ = false;

    partition_size_ = partition_size;
    embedding_size_ = embedding_size;
    dtype_ = dtype;
    dtype_size_ = get_dtype_size_wrapper(dtype_);
    total_size_ = partition_size_ * embedding_size_ * dtype_size_;

    idx_offset_ = idx_offset;
    file_offset_ = file_offset;
    buffer_idx_ = -1;

    tensor_ = torch::Tensor();

    evicting_ = false;
}

Partition::~Partition() {
    delete lock_;
    delete cv_;
    tensor_ = torch::Tensor();
}

torch::Tensor Partition::indexRead(Indices indices) {
    if (indices.sizes().size() != 1) {
        // TODO: throw invalid input to func exception
        throw std::runtime_error("");
    }

    lock_->lock();

    torch::Tensor ret = tensor_.index_select(0, indices - idx_offset_);

    lock_->unlock();
    cv_->notify_all();

    return ret;
}

PartitionedFile::PartitionedFile(string filename, int num_partitions, int64_t partition_size, int embedding_size, int64_t total_embeddings,
                                 torch::Dtype dtype) {
    num_partitions_ = num_partitions;
    partition_size_ = partition_size;
    embedding_size_ = embedding_size;
    total_embeddings_ = total_embeddings;
    dtype_ = dtype;
    dtype_size_ = get_dtype_size_wrapper(dtype_);

    filename_ = filename;

    int flags = O_RDWR | IO_FLAGS;
    fd_ = open(filename_.c_str(), flags);
    if (fd_ == -1) {
        SPDLOG_ERROR("Unable to open {}\nError: {}", filename_, errno);
        throw std::runtime_error("");
    }
}

void PartitionedFile::readPartition(void *addr, Partition *partition) {
    if (addr == NULL || partition == NULL) {
        // TODO: throw null ptr exception
        throw std::runtime_error("");
    }

    memset_wrapper(addr, 0, partition->total_size_);
    if (pread_wrapper(fd_, addr, partition->total_size_, partition->file_offset_) == -1) {
        SPDLOG_ERROR("Unable to read Block: {}\nError: {}", partition->partition_id_, errno);
        throw std::runtime_error("");
    }
    partition->data_ptr_ = addr;
    partition->tensor_ = torch::from_blob(addr, {partition->partition_size_, embedding_size_}, dtype_);
}

// writePartition accesses data pointed to by p->data_ptr_. Address p->data_ptr_ is expected to contain
// same data as that of p->tensor_.
void PartitionedFile::writePartition(Partition *partition, bool clear_mem) {
    if (partition == NULL || partition->data_ptr_ == nullptr) {
        // TODO: throw null ptr exception
        throw std::runtime_error("");
    }

    if (pwrite_wrapper(fd_, partition->data_ptr_, partition->total_size_, partition->file_offset_) == -1) {
        throw GegeRuntimeException(fmt::format("Unable to write partition: {}\nError: {}", partition->partition_id_, errno));
    }

    if (clear_mem) {
        memset_wrapper(partition->data_ptr_, 0, partition->total_size_);
        partition->data_ptr_ = nullptr;
        partition->tensor_ = torch::Tensor();
    }
}

LookaheadBlock::LookaheadBlock(int64_t total_size, PartitionedFile *partitioned_file, int num_per_lookahead) {
    total_size_ = total_size;
    partitioned_file_ = partitioned_file;
    partitions_ = {};
    lock_ = new std::mutex();

    mems_ = std::vector<void *>(num_per_lookahead);

    for (int i = 0; i < num_per_lookahead; i++) {
        if (posix_memalign(&mems_[i], 4096, total_size_)) {
            SPDLOG_ERROR("Unable to allocate lookahead memory\nError: {}", errno);
            throw std::runtime_error("");
        }
        memset_wrapper(mems_[i], 0, total_size_);
    }

    done_ = false;
    present_ = false;
    thread_ = nullptr;
}

LookaheadBlock::~LookaheadBlock() {
    delete lock_;

    for (void *mem : mems_) {
        free(mem);
    }
}

void LookaheadBlock::run() {
    while (!done_) {
        // wait until block is empty
        std::unique_lock lock(*lock_);
        cv_.wait(lock, [this] { return present_ == false; });

        if (partitions_.empty()) {
            break;
        }

#pragma omp parallel for
        for (int i = 0; i < partitions_.size(); i++) {
            Partition *partition = partitions_[i];
            std::unique_lock partition_lock(*partition->lock_);
            partition->cv_->wait(partition_lock, [partition] { return partition->evicting_ == false; });
            partitioned_file_->readPartition(mems_[i], partition);
            partition_lock.unlock();
            partition->cv_->notify_all();
        }

        present_ = true;
        lock.unlock();
        cv_.notify_all();
    }
}

void LookaheadBlock::start(std::vector<Partition *> first_partitions) {
    partitions_ = first_partitions;
    if (thread_ == nullptr) {
        thread_ = new std::thread(&LookaheadBlock::run, this);
    }
}

void LookaheadBlock::stop() {
    if (thread_ != nullptr) {
        if (thread_->joinable()) {
            done_ = true;
            present_ = false;
            cv_.notify_all();
            thread_->join();
        }
        delete thread_;
    }
}

void LookaheadBlock::move_to_buffer(std::vector<void *> buff_addrs, std::vector<int64_t> buffer_idxs, std::vector<Partition *> next_partitions) {
    if (partitions_.size() > buff_addrs.size() || partitions_.size() > buffer_idxs.size()) {
        // TODO: throw invalid inputs for function exception
        throw std::runtime_error("");
    }
    // wait until block is populated
    std::unique_lock lock(*lock_);
    cv_.wait(lock, [this] { return present_ == true; });

#pragma omp parallel for
    for (int i = 0; i < partitions_.size(); i++) {
        Partition *partition = partitions_[i];
        void *addr = buff_addrs[i];
        int64_t buffer_idx = buffer_idxs[i];
        memcpy_wrapper(addr, mems_[i], partition->total_size_);
        memset_wrapper(mems_[i], 0, partition->total_size_);

        partition->data_ptr_ = addr;
        partition->tensor_ = torch::from_blob(partition->data_ptr_, {partition->partition_size_, partition->embedding_size_}, partition->dtype_);
        partition->buffer_idx_ = buffer_idx;
        partition->present_ = true;
    }

    // next partition will be prefetched automatically
    partitions_ = next_partitions;
    present_ = false;
    lock.unlock();
    cv_.notify_all();
}

AsyncWriteBlock::AsyncWriteBlock(int64_t total_size, PartitionedFile *partitioned_file, int num_per_evict) {
    total_size_ = total_size;
    partitioned_file_ = partitioned_file;

    lock_ = new std::mutex();

    mems_ = std::vector<void *>(num_per_evict);

    for (int i = 0; i < num_per_evict; i++) {
        if (posix_memalign(&mems_[i], 4096, total_size_)) {
            SPDLOG_ERROR("Unable to allocate lookahead memory\nError: {}", errno);
            throw std::runtime_error("");
        }
        memset_wrapper(mems_[i], 0, total_size_);
    }

    done_ = false;
    present_ = false;
    thread_ = nullptr;
}

AsyncWriteBlock::~AsyncWriteBlock() {
    delete lock_;

    for (void *mem : mems_) {
        free(mem);
    }
}

void AsyncWriteBlock::run() {
    while (!done_) {
        // wait until block is empty
        std::unique_lock lock(*lock_);
        cv_.wait(lock, [this] { return present_ == true; });

        if (done_) {
            return;
        }

#pragma omp parallel for
        for (int i = 0; i < partitions_.size(); i++) {
            Partition *partition = partitions_[i];
            partitioned_file_->writePartition(partition);
            partition->present_ = false;
            partition->evicting_ = false;
            partition->cv_->notify_all();
        }

        present_ = false;
        lock.unlock();
        cv_.notify_all();
    }
}

void AsyncWriteBlock::start() {
    if (thread_ == nullptr) {
        thread_ = new std::thread(&AsyncWriteBlock::run, this);
    }
}

void AsyncWriteBlock::stop() {
    if (thread_ != nullptr) {
        if (thread_->joinable()) {
            done_ = true;
            present_ = true;
            cv_.notify_all();
            thread_->join();
        }
        delete thread_;
    }
}

void AsyncWriteBlock::async_write(std::vector<Partition *> partitions) {
    if (partitions.size() > mems_.size()) {
        // TODO: throw invalid inputs for function exception
        throw std::runtime_error("");
    }

    // wait until block is empty
    std::unique_lock lock(*lock_);
    cv_.wait(lock, [this] { return present_ == false; });

    partitions_ = partitions;

#pragma omp parallel for
    for (int i = 0; i < partitions_.size(); i++) {
        void *mem = mems_[i];
        Partition *partition = partitions_[i];

        memcpy_wrapper(mem, partition->data_ptr_, total_size_);
        memset_wrapper(partition->data_ptr_, 0, total_size_);

        partition->data_ptr_ = mem;
        partition->evicting_ = true;
    }

    present_ = true;

    lock.unlock();
    cv_.notify_all();
}

PartitionBuffer::PartitionBuffer(int capacity, int num_partitions, int fine_to_coarse_ratio, int64_t partition_size, int embedding_size,
                                 int64_t total_embeddings, torch::Dtype dtype, string filename, bool prefetching) {
    capacity_ = capacity;
    size_ = 0;
    num_partitions_ = num_partitions;
    partition_size_ = partition_size;
    fine_to_coarse_ratio_ = fine_to_coarse_ratio;
    dtype_ = dtype;
    dtype_size_ = get_dtype_size_wrapper(dtype_);
    embedding_size_ = embedding_size;
    total_embeddings_ = total_embeddings;

    filename_ = filename;
    partition_table_ = std::vector<Partition *>();

    prefetching_ = prefetching;

    int64_t curr_idx_offset = 0;
    int64_t curr_file_offset = 0;
    int64_t curr_partition_size = partition_size_;
    int64_t curr_total_size = curr_partition_size * embedding_size_ * dtype_size_;
    for (int64_t i = 0; i < num_partitions_; i++) {
        // the last partition might be slightly smaller
        if (i == num_partitions_ - 1) {
            curr_partition_size = total_embeddings_ - curr_idx_offset;
            curr_total_size = curr_partition_size * embedding_size_ * dtype_size_;
        }
        Partition *curr_part = new Partition(i, curr_partition_size, embedding_size_, dtype_, curr_idx_offset, curr_file_offset);
        partition_table_.push_back(curr_part);

        curr_file_offset += curr_total_size;
        curr_idx_offset += curr_partition_size;
    }

    filename_ = filename;
    partitioned_file_ = new PartitionedFile(filename_, num_partitions_, partition_size_, embedding_size_, total_embeddings_, dtype_);

    tensor_mem_ = nullptr;
    loaded_ = false;
}

PartitionBuffer::~PartitionBuffer() {
    unload(true);

    delete partitioned_file_;
    for (int64_t i = 0; i < num_partitions_; i++) {
        delete partition_table_[i];
    }
}

void PartitionBuffer::load() {
    if (!loaded_) {
        if (posix_memalign(&buff_mem_, 4096, capacity_ * partition_size_ * embedding_size_ * dtype_size_)) {
            SPDLOG_ERROR("Unable to allocate buffer memory\nError: {}", errno);
            throw std::runtime_error("");
        }
        memset_wrapper(buff_mem_, 0, capacity_ * partition_size_ * embedding_size_ * dtype_size_);
        buffer_tensor_view_ = torch::from_blob(buff_mem_, {capacity_ * partition_size_, embedding_size_}, dtype_);

        // initialize buffer
        int partition_id;

        int64_t num_nodes = 0;

        for (int i = 0; i < buffer_state_.size(0); i++) {
            partition_id = buffer_state_[i].item<int>();
            Partition *partition = partition_table_[partition_id];
            void *buff_addr = (char *)buff_mem_ + (i * partition_size_ * embedding_size_ * dtype_size_);
            partitioned_file_->readPartition(buff_addr, partition);
            partition->present_ = true;
            partition->buffer_idx_ = i;
            num_nodes += partition->partition_size_;
        }

        in_buffer_ids_ = torch::empty({num_nodes}, torch::kInt64);

        if (prefetching_) {
            lookahead_block_ = new LookaheadBlock(partition_size_ * embedding_size_ * dtype_size_, partitioned_file_, fine_to_coarse_ratio_);
            async_write_block_ = new AsyncWriteBlock(partition_size_ * embedding_size_ * dtype_size_, partitioned_file_, fine_to_coarse_ratio_);
            startThreads();
        }

        loaded_ = true;
    }
}

void PartitionBuffer::unload(bool write) {

    if (loaded_) {
        if (write) {
            sync();
        }
        buffer_tensor_view_ = torch::Tensor();
        free(buff_mem_);
        buff_mem_ = nullptr;

        if (prefetching_) {
            stopThreads();
            delete lookahead_block_;
            delete async_write_block_;
        }

        size_ = 0;
        loaded_ = false;
    }
}

torch::Tensor PartitionBuffer::getBufferState() { return buffer_state_; }

// indices a relative to the local node ids
torch::Tensor PartitionBuffer::indexRead(torch::Tensor indices) {
    if (indices.sizes().size() != 1) {
        // TODO: throw invalid input to func exception
        throw std::runtime_error("");
    }
    return buffer_tensor_view_.index_select(0, indices);
}

Indices PartitionBuffer::getRandomIds(int64_t size) { return torch::randint(in_buffer_ids_.size(0), size, torch::kInt64); }

// indices must contain unique values, else there is a possibility of a race condition
void PartitionBuffer::indexAdd(torch::Tensor indices, torch::Tensor values) {
    if (!values.defined() || indices.sizes().size() != 1 || indices.size(0) != values.size(0) || buffer_tensor_view_.size(1) != values.size(1)) {
        // TODO: throw invalid inputs for function error
        throw std::runtime_error("");
    }

    // assumes this operation is only used on float valued data, and this op takes place on the CPU
    auto data_accessor = buffer_tensor_view_.accessor<float, 2>();
    auto ids_accessor = indices.accessor<int64_t, 1>();
    auto values_accessor = values.accessor<float, 2>();

    int d = values.size(1);
    int64_t size = indices.size(0);
#pragma omp parallel for
    for (int64_t i = 0; i < size; i++) {
        for (int j = 0; j < d; j++) {
            data_accessor[ids_accessor[i]][j] += values_accessor[i][j];
        }
    }
}

void PartitionBuffer::setBufferOrdering(std::vector<torch::Tensor> buffer_states) {
    buffer_states_ = buffer_states;
    buffer_state_iterator_ = buffer_states_.begin();
    buffer_state_ = *buffer_state_iterator_++;
    
    if (loaded_) {
        SPDLOG_INFO("Buffer ordering changed, unloading buffer");
        unload(true);
        load();
    }
}

bool PartitionBuffer::hasSwap() { return buffer_state_iterator_ != buffer_states_.end(); }

void PartitionBuffer::performNextSwap() {
    if (!buffer_state_.defined() || buffer_state_iterator_ == buffer_states_.end()) {
        return;
    }

    // get evicted and admitted partitions
    std::vector<int> evict_ids = getNextEvict();
    std::vector<int> admit_ids = getNextAdmit();

    std::vector<Partition *> admit_partitions;
    std::vector<Partition *> evict_partitions;
    std::vector<int64_t> evict_buffer_idxs;
    for (int admit_id : admit_ids) {
        admit_partitions.emplace_back(partition_table_[admit_id]);
    }
    for (int evict_id : evict_ids) {
        evict_partitions.emplace_back(partition_table_[evict_id]);
        evict_buffer_idxs.emplace_back(partition_table_[evict_id]->buffer_idx_);
    }

    buffer_state_ = *buffer_state_iterator_++;

    // evict partition
    auto t1 = std::chrono::high_resolution_clock::now();
    evict(evict_partitions);
    auto t2 = std::chrono::high_resolution_clock::now();
    SPDLOG_INFO("evict time: {}", std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());
    // admit partition
    t1 = std::chrono::high_resolution_clock::now();
    admit(admit_partitions, evict_buffer_idxs);
    t2 = std::chrono::high_resolution_clock::now();
    SPDLOG_INFO("admit time: {}", std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());

    int64_t num_nodes = 0;

    int partition_id;
    for (int i = 0; i < buffer_state_.size(0); i++) {
        partition_id = buffer_state_[i].item<int>();
        num_nodes += partition_table_[partition_id]->partition_size_;
    }

    in_buffer_ids_ = torch::empty({num_nodes}, torch::kInt64);

    //    int64_t offset = 0;
    //    for (int i = 0; i < buffer_state_.size(0); i++) {
    //        partition_id = buffer_state_[i].item<int>();
    //        Partition *partition = partition_table_[partition_id];
    //        int64_t partition_offset = partition->idx_offset_;
    //
    //        in_buffer_ids_.slice(0, offset, offset + partition->partition_size_) = torch::arange(partition_offset, partition_offset +
    //        partition->partition_size_); offset += partition->partition_size_;
    //    }
}

std::vector<int> PartitionBuffer::getNextAdmit() {
    std::vector<int> admit_ids;
    bool admitted;

    if (buffer_state_iterator_ != buffer_states_.end()) {
        for (int i = 0; i < buffer_state_iterator_->size(0); i++) {
            admitted = true;
            for (int j = 0; j < buffer_state_.size(0); j++) {
                if ((*buffer_state_iterator_)[i].item<int>() == (buffer_state_)[j].item<int>()) {
                    admitted = false;
                }
            }
            if (admitted) {
                admit_ids.emplace_back((*buffer_state_iterator_)[i].item<int>());
            }
        }
    }
    return admit_ids;
}

std::vector<int> PartitionBuffer::getNextEvict() {
    std::vector<int> evict_ids;
    bool evicted;

    for (int i = 0; i < buffer_state_.size(0); i++) {
        evicted = true;
        for (int j = 0; j < buffer_state_iterator_->size(0); j++) {
            if ((*buffer_state_iterator_)[j].item<int>() == buffer_state_[i].item<int>()) {
                evicted = false;
            }
        }
        if (evicted) {
            evict_ids.emplace_back(buffer_state_[i].item<int>());
        }
    }
    return evict_ids;
}

torch::Tensor PartitionBuffer::getGlobalToLocalMap(bool get_current) {

    torch::Tensor buffer_index_map = -torch::ones({total_embeddings_}, torch::kInt64);
    torch::Tensor buffer_state;

    if (get_current) {
        buffer_state = buffer_state_;
#pragma omp parallel for
        for (int i = 0; i < buffer_state.size(0); i++) {
            int partition_id = buffer_state[i].item<int>();
            Partition *partition = partition_table_[partition_id];
            int64_t partition_offset = partition->idx_offset_;
            int64_t buffer_offset = partition->buffer_idx_ * partition_size_;
            buffer_index_map.slice(0, partition_offset, partition_offset + partition->partition_size_) =
                torch::arange(buffer_offset, buffer_offset + partition->partition_size_);
        }
    } else {
        // get mapping for next swap
        buffer_state = *buffer_state_iterator_;

        // get evicted and admitted partitions
        std::vector<int> evict_ids = getNextEvict();
        std::vector<int> admit_ids = getNextAdmit();

        // get mapping for the partitions that will still be in the buffer
#pragma omp parallel for
        for (int i = 0; i < buffer_state.size(0); i++) {
            int partition_id = buffer_state[i].item<int>();
            Partition *partition = partition_table_[partition_id];
            int64_t partition_offset = partition->idx_offset_;

            if (partition->buffer_idx_ != -1) {
                int64_t buffer_offset = partition->buffer_idx_ * partition_size_;
                buffer_index_map.slice(0, partition_offset, partition_offset + partition->partition_size_) =
                    torch::arange(buffer_offset, buffer_offset + partition->partition_size_);
            }
        }

// get mapping for the partitions that will be admitted
#pragma omp parallel for
        for (int i = 0; i < evict_ids.size(); i++) {
            Partition *admit_partition = partition_table_[admit_ids[i]];
            Partition *evict_partition = partition_table_[evict_ids[i]];
            int64_t partition_offset = admit_partition->idx_offset_;
            int64_t buffer_offset = evict_partition->buffer_idx_ * partition_size_;
            buffer_index_map.slice(0, partition_offset, partition_offset + admit_partition->partition_size_) =
                torch::arange(buffer_offset, buffer_offset + admit_partition->partition_size_);
        }
    }
    return buffer_index_map;
}

torch::Tensor PartitionBuffer::getPartitionToBufferSlotMap() {
    torch::Tensor partition_to_buffer_slot = -torch::ones({num_partitions_}, torch::kInt64);
    auto slot_accessor = partition_to_buffer_slot.accessor<int64_t, 1>();

#pragma omp parallel for
    for (int partition_id = 0; partition_id < num_partitions_; partition_id++) {
        Partition *partition = partition_table_[partition_id];
        if (partition->present_) {
            slot_accessor[partition_id] = partition->buffer_idx_;
        }
    }

    return partition_to_buffer_slot;
}

void PartitionBuffer::evict(std::vector<Partition *> evict_partitions) {
    if (prefetching_) {
        async_write_block_->async_write(evict_partitions);
    } else {
#pragma omp parallel for
        for (int i = 0; i < evict_partitions.size(); i++) {
            partitioned_file_->writePartition(evict_partitions[i]);
        }
    }

#pragma omp parallel for
    for (int i = 0; i < evict_partitions.size(); i++) {
        evict_partitions[i]->present_ = false;
    }
}

void PartitionBuffer::admit(std::vector<Partition *> admit_partitions, std::vector<int64_t> buffer_idxs) {
    if (admit_partitions.size() > buffer_idxs.size()) {
        // TODO: throw invalid inputs for function error
        throw std::runtime_error("");
    }

    std::vector<void *> buff_addrs(buffer_idxs.size());
    
#pragma omp parallel for
    for (int i = 0; i < buffer_idxs.size(); i++) {
        void *buff_addr = (char *)buff_mem_ + (buffer_idxs[i] * partition_size_ * embedding_size_ * dtype_size_);
        buff_addrs[i] = buff_addr;
    }

    if (prefetching_) {
        std::vector<int> next_admit_ids = getNextAdmit();
        std::vector<Partition *> next_partitions;
        if (!next_admit_ids.empty()) {
            for (int admit_id : next_admit_ids) {
                next_partitions.emplace_back(partition_table_[admit_id]);
            }
        }
        lookahead_block_->move_to_buffer(buff_addrs, buffer_idxs, next_partitions);
    } else {
#pragma omp parallel for
        for (int i = 0; i < admit_partitions.size(); i++) {
            Partition *partition = admit_partitions[i];
            partitioned_file_->readPartition(buff_addrs[i], partition);
            partition->present_ = true;
            partition->buffer_idx_ = buffer_idxs[i];
        }
    }
}

void PartitionBuffer::sync() {
    SPDLOG_DEBUG("Synchronizing buffer");
    Partition *curr_partition;
    for (int i = 0; i < num_partitions_; i++) {
        curr_partition = partition_table_[i];
        if (curr_partition->present_) {
            partitioned_file_->writePartition(curr_partition, true);
            curr_partition->present_ = false;
            curr_partition->buffer_idx_ = -1;
        }
    }
}

void PartitionBuffer::startThreads() {
    SPDLOG_DEBUG("Starting prefetching threads");
    std::vector<Partition *> partitions;
    std::vector<int> admit_ids = getNextAdmit();
    for (int admit_id : admit_ids) {
        partitions.emplace_back(partition_table_[admit_id]);
    }
    lookahead_block_->start(partitions);
    async_write_block_->start();
}

void PartitionBuffer::stopThreads() {
    SPDLOG_DEBUG("Stopping prefetching threads");
    lookahead_block_->stop();
    async_write_block_->stop();
}

MemPartitionBuffer::MemPartitionBuffer(int capacity, int num_partitions, int fine_to_coarse_ratio, int64_t partition_size, int embedding_size, int64_t total_embeddings,
                    torch::Dtype dtype, string filename, bool prefetching, torch::Device device, int buffer_sizes)
    :PartitionBuffer(capacity, num_partitions, fine_to_coarse_ratio, partition_size, embedding_size, total_embeddings, dtype, filename, prefetching) {
    buffer_sizes_ = buffer_sizes;
    device_ = device;

    // if (posix_memalign(&buff_mem_, 4096, capacity_ * partition_size_ * embedding_size_ * dtype_size_)) {
    //     SPDLOG_ERROR("Unable to allocate buffer memory\nError: {}", errno);
    //     throw std::runtime_error("");
    // }
    // memset_wrapper(buff_mem_, 0, capacity_ * partition_size_ * embedding_size_ * dtype_size_);

    torch::TensorOptions options = torch::TensorOptions().dtype(dtype_).device(torch::kCPU).pinned_memory(true);
    buffer_tensor_view_ = torch::zeros({capacity_ * partition_size_, embedding_size_}, options);
    buff_mem_ = buffer_tensor_view_.data_ptr();
    buffer_tensor_gpu_view_ = torch::empty({capacity_ * partition_size_, embedding_size_}, dtype_).to(device_);
    // buffer_tensor_gpu_view_ = buffer_tensor_view_.to(device_);
    perm_ = torch::arange(0, total_embeddings_, torch::kInt64);
    pos_  = torch::arange(0, total_embeddings_, torch::kInt64);
}

MemPartitionBuffer::~MemPartitionBuffer() {
    unload(true);
    // free(buff_mem_);
    buffer_tensor_view_ = torch::Tensor();
    buffer_tensor_gpu_view_ = torch::Tensor();
}


torch::Tensor MemPartitionBuffer::getGlobalToLocalMap(bool get_current) {

    torch::Tensor buffer_index_map = -torch::ones({total_embeddings_}, torch::kInt64);
    torch::Tensor buffer_state;

    buffer_state = buffer_state_;
#pragma omp parallel for
    for (int i = 0; i < buffer_state.size(0); i++) {
        int partition_id = buffer_state[i].item<int>();
        Partition *partition = partition_table_[partition_id];
        int64_t partition_offset = partition->idx_offset_;
        int64_t buffer_offset = partition->buffer_idx_ * partition_size_;
        buffer_index_map.index_put_({pos_.slice(0, partition_offset, partition_offset + partition->partition_size_)}, 
                                                torch::arange(buffer_offset, buffer_offset + partition->partition_size_));
    }
    return buffer_index_map;
}

void MemPartitionBuffer::load(torch::Tensor data_storage) {
    if (!loaded_) {
        auto t1 = std::chrono::high_resolution_clock::now();
        int64_t num_nodes = 0;
        // buffer_tensor_view_ = torch::from_blob(buff_mem_, {capacity_ * partition_size_, embedding_size_}, dtype_);
        data_storage_ = data_storage;
        void* tensor_mem_ = data_storage_.data_ptr();

#pragma omp parallel for
        for (int i = 0; i < 4; i++) {
            int partition_id = buffer_state_[i].item<int>();
            Partition *partition = partition_table_[partition_id];
            // void *buff_addr = (char *)buff_mem_ + (i * partition_size_ * embedding_size_ * dtype_size_);
            // void *tensor_addr = (char *)tensor_mem_ + (partition->idx_offset_ * embedding_size_ * dtype_size_);
            partition->present_ = true;
            partition->buffer_idx_ = i;
            int64_t buffer_offset = partition->buffer_idx_ * partition_size_;
            // memset_wrapper(buff_addr, 0, partition->total_size_);
            // memcpy_wrapper(buff_addr, tensor_addr, partition->total_size_);
            buffer_tensor_view_.slice(0, buffer_offset, buffer_offset + partition->partition_size_) = 
                    data_storage_.index_select(0, pos_.slice(0, partition->idx_offset_, partition->idx_offset_ + partition->partition_size_));
            // partition->data_ptr_ = buff_addr;
            num_nodes += partition->partition_size_;
        }

        loaded_ = true;
        buffer_tensor_gpu_view_.copy_(buffer_tensor_view_);
        auto t2 = std::chrono::high_resolution_clock::now();
        // SPDLOG_INFO("Loaded {} nodes in {} ms", num_nodes, std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());
    }
}

void MemPartitionBuffer::indexAdd(torch::Tensor indices, torch::Tensor values) {
    if (!values.defined() || indices.sizes().size() != 1 || indices.size(0) != values.size(0) || buffer_tensor_gpu_view_.size(1) != values.size(1)) {
        // TODO: throw invalid input to func exception
        throw std::runtime_error("");
    }
    int64_t debug_update_id = -1;
    bool run_stage_debug = should_run_stage_debug(debug_update_id);
    auto index_add_start = std::chrono::high_resolution_clock::now();
    auto step_start = index_add_start;
    
    if (values.device().is_cuda()) {
#ifdef GEGE_CUDA
        if (csr_update_enabled()) {
            static bool logged = false;
            if (!logged) {
                SPDLOG_INFO("MemPartitionBuffer::indexAdd using direct CSR update path");
                logged = true;
            }
            torch::Tensor update_indices = indices;
            torch::Tensor update_values = values;
            if (csr_update_reduce_enabled()) {
                std::tie(update_indices, update_values) = reduce_updates_with_csr(indices, values);
                if (run_stage_debug) {
                    auto now = std::chrono::high_resolution_clock::now();
                    SPDLOG_INFO("[stage-debug][buffer.indexAdd][update {}][step 1] csr_reduce ms={:.3f} in_rows={} out_rows={}",
                                debug_update_id, elapsed_ms(step_start, now), indices.numel(), update_indices.numel());
                    step_start = now;
                }
            }
            buffer_tensor_gpu_view_.index_add_(0, update_indices, update_values);
            if (run_stage_debug) {
                auto now = std::chrono::high_resolution_clock::now();
                SPDLOG_INFO("[stage-debug][buffer.indexAdd][update {}][step 2] index_add_cuda ms={:.3f} rows={} dim={}",
                            debug_update_id, elapsed_ms(step_start, now), update_indices.numel(), update_values.size(1));
            }
        } else {
            buffer_tensor_gpu_view_.index_add_(0, indices, values);
            if (run_stage_debug) {
                auto now = std::chrono::high_resolution_clock::now();
                SPDLOG_INFO("[stage-debug][buffer.indexAdd][update {}][step 1] index_add_cuda ms={:.3f} rows={} dim={} csr_update={}",
                            debug_update_id, elapsed_ms(step_start, now), indices.numel(), values.size(1), false);
            }
        }
#else
        buffer_tensor_gpu_view_.index_add_(0, indices, values);
#endif
    } else {
        // assumes this operation is only used on float valued data.
        auto data_accessor = buffer_tensor_view_.accessor<float, 2>();
        auto ids_accessor = indices.accessor<int64_t, 1>();
        auto values_accessor = values.accessor<float, 2>();

        int d = values.size(1);
        int64_t size = indices.size(0);
#pragma omp parallel for
        for (int64_t i = 0; i < size; i++) {
            for (int j = 0; j < d; j++) {
                data_accessor[ids_accessor[i]][j] += values_accessor[i][j];
            }
        }
        if (run_stage_debug) {
            auto now = std::chrono::high_resolution_clock::now();
            SPDLOG_INFO("[stage-debug][buffer.indexAdd][update {}][step 1] index_add_cpu ms={:.3f} rows={} dim={}",
                        debug_update_id, elapsed_ms(step_start, now), size, d);
        }
    }

    if (run_stage_debug) {
        auto now = std::chrono::high_resolution_clock::now();
        SPDLOG_INFO("[stage-debug][buffer.indexAdd][update {}][step 9] total_ms={:.3f} device={}",
                    debug_update_id, elapsed_ms(index_add_start, now), values.device().str());
    }
}


void MemPartitionBuffer::performNextSwap() {
    if (!buffer_state_.defined() || buffer_state_iterator_ == buffer_states_.end()) {
        return;
    }

    int64_t timing_id = -1;
    bool log_timing = should_log_partition_buffer_swap_timing(timing_id);
    auto total_start = std::chrono::high_resolution_clock::now();
    auto t1 = total_start;
    double unload_ms = 0.0;
    double load_ms = 0.0;

    // evict partition
    unload(true);
    auto t2 = std::chrono::high_resolution_clock::now();
    unload_ms = elapsed_ms(t1, t2);
#ifdef GEGE_CUDA
    c10::cuda::CUDACachingAllocator::emptyCache();
#endif
    
    buffer_state_ = *buffer_state_iterator_;

    for(int i = 0; i < buffer_sizes_; i ++) {
        if (buffer_states_.end() != buffer_state_iterator_)
            buffer_state_iterator_++;
    }

    // admit partition
    t1 = std::chrono::high_resolution_clock::now();
    load(data_storage_);
    t2 = std::chrono::high_resolution_clock::now();
    load_ms = elapsed_ms(t1, t2);
#ifdef GEGE_CUDA
    c10::cuda::CUDACachingAllocator::emptyCache();
#endif

    int64_t num_nodes = 0;

    int partition_id;
    for (int i = 0; i < buffer_state_.size(0); i++) {
        partition_id = buffer_state_[i].item<int>();
        num_nodes += partition_table_[partition_id]->partition_size_;
    }

    if (log_timing) {
        auto total_end = std::chrono::high_resolution_clock::now();
        SPDLOG_INFO(
            "[partition-buffer-swap][swap {}] this={} device={} active_parts={} nodes={} unload_ms={:.3f} load_ms={:.3f} total_ms={:.3f}",
            timing_id, fmt::ptr(this), device_.str(), buffer_state_.size(0), num_nodes, unload_ms, load_ms, elapsed_ms(total_start, total_end));
    }
}

void MemPartitionBuffer::sync() {
    int64_t timing_id = -1;
    bool log_timing = should_log_partition_buffer_swap_timing(timing_id);
    auto t1 = std::chrono::high_resolution_clock::now();
    void* tensor_mem_ = data_storage_.data_ptr();
#pragma omp parallel for
    for (int i = 0; i < 4; i++) {
        int partition_id = buffer_state_[i].item<int>();
        Partition *partition = partition_table_[partition_id];
        int64_t buffer_offset = partition->buffer_idx_ * partition_size_;
        // void *tensor_addr = (char *)tensor_mem_ + (partition->idx_offset_ * embedding_size_ * dtype_size_);
        // void *buff_addr = (char *)buff_mem_unload_ + (i * partition_size_ * embedding_size_ * dtype_size_);

        // memcpy_wrapper(tensor_addr, buff_addr, partition->total_size_);
        // memset_wrapper(buff_addr, 0, partition->total_size_);
        // auto t1 = std::chrono::high_resolution_clock::now();
        data_storage_.index_put_({pos_.slice(0, partition->idx_offset_, partition->idx_offset_ + partition->partition_size_)}, 
                buffer_tensor_view_.slice(0, buffer_offset, buffer_offset + partition->partition_size_));
        // auto t2 = std::chrono::high_resolution_clock::now();
        // SPDLOG_INFO("Sync time: {}", std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());
        partition->data_ptr_ = nullptr;
        // partition->tensor_ = torch::Tensor();
        partition->present_ = false;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    if (log_timing) {
        SPDLOG_INFO("[partition-buffer-swap][sync {}] this={} device={} parts={} sync_ms={:.3f}",
                    timing_id, fmt::ptr(this), device_.str(), buffer_state_.defined() ? buffer_state_.size(0) : 0,
                    elapsed_ms(t1, t2));
    }
}



void MemPartitionBuffer::setBufferOrdering(std::vector<torch::Tensor> buffer_states) {
    buffer_states_ = buffer_states;
    buffer_state_iterator_ = buffer_states_.begin();

    for(int i = 0; i < device_.index(); i ++) 
        buffer_state_iterator_++;
    
    buffer_state_ = *buffer_state_iterator_;
    for(int i = 0; i < buffer_sizes_; i ++ ) 
        if (buffer_state_iterator_ != buffer_states_.end())
            buffer_state_iterator_++;
    loaded_ = false;
}

void MemPartitionBuffer::unload(bool write) {

    if (loaded_) {
        int64_t timing_id = -1;
        bool log_timing = should_log_partition_buffer_swap_timing(timing_id);
        auto total_start = std::chrono::high_resolution_clock::now();
        auto phase_start = total_start;
        double gpu_to_cpu_ms = 0.0;
        if (buffer_tensor_gpu_view_.device().is_cuda()) {
            buffer_tensor_view_.copy_(buffer_tensor_gpu_view_.detach());
            if (log_timing) {
                auto now = std::chrono::high_resolution_clock::now();
                gpu_to_cpu_ms = elapsed_ms(phase_start, now);
                phase_start = now;
            }
            buff_mem_unload_ = buffer_tensor_view_.data_ptr();

        }
        
        sync();
        double sync_ms = 0.0;
        if (log_timing) {
            auto now = std::chrono::high_resolution_clock::now();
            sync_ms = elapsed_ms(phase_start, now);
            SPDLOG_INFO(
                "[partition-buffer-swap][unload {}] this={} device={} loaded_parts={} gpu_to_cpu_ms={:.3f} sync_ms={:.3f} total_ms={:.3f}",
                timing_id, fmt::ptr(this), device_.str(), buffer_state_.defined() ? buffer_state_.size(0) : 0, gpu_to_cpu_ms, sync_ms,
                elapsed_ms(total_start, now));
        }
        // buffer_tensor_view_ = torch::Tensor();
        // buff_mem_unload_ = nullptr;

        size_ = 0;
        loaded_ = false;
    }
}

void MemPartitionBuffer::evict(std::vector<Partition *> evict_partitions) {
    if (prefetching_) {
        async_write_block_->async_write(evict_partitions);
    } else {
        unload(true);
    }
}

void MemPartitionBuffer::admit(std::vector<Partition *> admit_partitions, std::vector<int64_t> buffer_idxs) {
    if (admit_partitions.size() > buffer_idxs.size()) {
        // TODO: throw invalid inputs for function error
        throw std::runtime_error("");
    }

    if (prefetching_) {
        std::vector<void *> buff_addrs(buffer_idxs.size());

#pragma omp parallel for
        for (int i = 0; i < buffer_idxs.size(); i++) {
            void *buff_addr = (char *)buff_mem_ + (buffer_idxs[i] * partition_size_ * embedding_size_ * dtype_size_);
            buff_addrs[i] = buff_addr;
        }

        std::vector<int> next_admit_ids = getNextAdmit();
        std::vector<Partition *> next_partitions;
        if (!next_admit_ids.empty()) {
            for (int admit_id : next_admit_ids) {
                next_partitions.emplace_back(partition_table_[admit_id]);
            }
        }
        lookahead_block_->move_to_buffer(buff_addrs, buffer_idxs, next_partitions);
    } else {
        load(data_storage_);
    }
}

torch::Tensor MemPartitionBuffer::indexRead(torch::Tensor indices) {
    if (indices.sizes().size() != 1) {
        // TODO: throw invalid input to func exception
        throw std::runtime_error("");
    }
    
    return buffer_tensor_gpu_view_.index_select(0, indices);
}
