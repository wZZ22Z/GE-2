#include "storage/storage.h"

#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>

#include <iostream>
#include <string>

#include "common/util.h"
#include "configuration/constants.h"
#include "reporting/logger.h"

#if defined(GEGE_CUDA)
#include "pytorch_scatter/segment_sum.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
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

using std::ios;
using std::ios_base;

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

bool partition_buffer_peer_relay_enabled() {
    static bool enabled = parse_env_flag("GEGE_PARTITION_BUFFER_PEER_RELAY", false);
    return enabled;
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
    ScopedNvtxRange nvtx_scope("storage.reduce_updates_with_csr");

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

void renameFile(string old_filename, string new_filename) {
    int result = rename(old_filename.c_str(), new_filename.c_str());
    if (result != 0) {
        SPDLOG_ERROR("Unable to rename {}\nError: {}", old_filename, errno);
        throw std::runtime_error("");
    }
}

void copyFile(string src_file, string dst_file) {
    std::ifstream src;
    std::ofstream dst;

    src.open(src_file, ios::in | ios::binary);
    dst.open(dst_file, ios::out | ios::binary);

    dst << src.rdbuf();

    src.close();
    dst.close();
}

bool fileExists(string filename) {
    if (FILE *file = fopen(filename.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }
}

void createDir(string path, bool exist_ok) {
    if (mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1) {
        if (errno == EEXIST) {
            if (exist_ok) {
                SPDLOG_DEBUG("{} directory already exists", path);
            } else {
                SPDLOG_ERROR("{} directory already exists", path);
                throw std::runtime_error("");
            }
        } else {
            SPDLOG_ERROR("Failed to create {}\nError: {}", path, errno);
            throw std::runtime_error("");
        }
    }
}

Storage::Storage() : device_(torch::kCPU) {}

PartitionBufferStorage::PartitionBufferStorage(string filename, int64_t dim0_size, int64_t dim1_size, shared_ptr<PartitionBufferOptions> options) {
    filename_ = filename;
    dim0_size_ = dim0_size;
    dim1_size_ = dim1_size;
    options_ = options;
    dtype_ = options_->dtype;
    initialized_ = true;
    loaded_ = false;
    int64_t partition_size = ceil((double)dim0_size_ / options_->num_partitions);
    device_ = torch::kCPU;

    buffer_ = new PartitionBuffer(options_->buffer_capacity, options_->num_partitions, options_->fine_to_coarse_ratio, partition_size, dim1_size_, dim0_size_,
                                  dtype_, filename_, options_->prefetching);
}

PartitionBufferStorage::PartitionBufferStorage(string filename, torch::Tensor data, shared_ptr<PartitionBufferOptions> options) {
    filename_ = filename;
    dim0_size_ = 0;
    dim1_size_ = data.size(1);
    options_ = options;
    dtype_ = options_->dtype;
    append(data);
    initialized_ = true;
    loaded_ = false;
    int64_t partition_size = ceil((double)dim0_size_ / options_->num_partitions);
    device_ = torch::kCPU;

    buffer_ = new PartitionBuffer(options_->buffer_capacity, options_->num_partitions, options_->fine_to_coarse_ratio, partition_size, dim1_size_, dim0_size_,
                                  dtype_, filename_, options_->prefetching);
}

PartitionBufferStorage::PartitionBufferStorage(string filename, shared_ptr<PartitionBufferOptions> options) {
    filename_ = filename;
    dim0_size_ = 0;
    initialized_ = false;
    loaded_ = false;
    options_ = options;
    dtype_ = options_->dtype;
    int64_t partition_size = ceil((double)dim0_size_ / options_->num_partitions);
    device_ = torch::kCPU;

    buffer_ = new PartitionBuffer(options_->buffer_capacity, options_->num_partitions, options_->fine_to_coarse_ratio, partition_size, dim1_size_, dim0_size_,
                                  dtype_, filename_, options_->prefetching);
}

void PartitionBufferStorage::rangePut(int64_t offset, torch::Tensor values) {
    int fd = open(filename_.c_str(), O_RDWR | IO_FLAGS);
    if (fd == -1) {
        SPDLOG_ERROR("Unable to open {}\nError: {}", filename_, errno);
        throw std::runtime_error("");
    }

    int64_t dtype_size = get_dtype_size_wrapper(dtype_);
    int64_t ptr_offset = offset * dim1_size_ * dtype_size;

    if (pwrite_wrapper(fd, values.data_ptr(), values.size(0) * dim1_size_ * dtype_size, ptr_offset) == -1) {
        SPDLOG_ERROR("Unable to write {}\nError: {}", filename_, errno);
        throw std::runtime_error("");
    }

    close(fd);
}

void PartitionBufferStorage::append(torch::Tensor values) {
    ios::openmode flags;

    if (dim0_size_ == 0) {
        flags = ios::trunc | ios::binary;
    } else {
        flags = ios::binary | ios_base::app;
    }

    dim0_size_ += values.size(0);
    dim1_size_ = values.size(1);
    dtype_ = values.scalar_type();

    std::ofstream outfile(filename_, flags);

    int dtype_size = get_dtype_size_wrapper(dtype_);

    outfile.write((char *)values.data_ptr(), values.size(0) * values.size(1) * dtype_size);

    outfile.close();
}

PartitionBufferStorage::~PartitionBufferStorage() { delete buffer_; }

void PartitionBufferStorage::load() {
    if (!loaded_ && initialized_) {
        buffer_->load();
        loaded_ = true;
    }
}

void PartitionBufferStorage::write() {
    if (loaded_) {
        buffer_->sync();
    }
}

void PartitionBufferStorage::unload(bool perform_write) {
    if (loaded_) {
        buffer_->unload(perform_write);
        loaded_ = false;
    }
}

torch::Tensor PartitionBufferStorage::indexRead(Indices indices) { return buffer_->indexRead(indices); }

void PartitionBufferStorage::indexAdd(Indices indices, torch::Tensor values) { return buffer_->indexAdd(indices, values); }

torch::Tensor PartitionBufferStorage::range(int64_t offset, int64_t n) {
    SPDLOG_ERROR("Unsupported operation for PartitionBufferStorage");
    throw std::runtime_error("");
}

void PartitionBufferStorage::indexPut(Indices indices, torch::Tensor values) {
    SPDLOG_ERROR("Unsupported operation for PartitionBufferStorage");
    throw std::runtime_error("");
}

void PartitionBufferStorage::rangePut(int64_t offset, int64_t n, torch::Tensor values) {
    SPDLOG_ERROR("Unsupported operation for PartitionBufferStorage");
    throw std::runtime_error("");
}

void PartitionBufferStorage::shuffle() {
    SPDLOG_ERROR("Shuffle not supported for PartitionBufferStorage");
    throw std::runtime_error("");
};

void PartitionBufferStorage::sort(bool src) {
    SPDLOG_ERROR("Sort not supported for PartitionBufferStorage");
    throw std::runtime_error("");
};

MemPartitionBufferStorage::MemPartitionBufferStorage(string filename, int64_t dim0_size, int64_t dim1_size, shared_ptr<PartitionBufferOptions> options, std::vector<torch::Device> devices) {
    filename_ = filename;
    dim0_size_ = dim0_size;
    dim1_size_ = dim1_size;
    options_ = options;
    dtype_ = options_->dtype;
    initialized_ = true;
    loaded_ = false;
    peer_relay_runtime_enabled_ = false;
    int64_t partition_size = ceil((double)dim0_size_ / options_->num_partitions);
    device_ = torch::kCUDA;
    devices_ = devices;
    for (int i = 0; i < devices_.size(); i ++) {
        MemPartitionBuffer* buffer = new MemPartitionBuffer(options_->buffer_capacity, options_->num_partitions, options_->fine_to_coarse_ratio, partition_size, dim1_size_, dim0_size_,
                                  dtype_, filename_, options_->prefetching, devices_[i], devices_.size());
        buffers_.emplace_back(buffer);
    }
}

void MemPartitionBufferStorage::initializePeerRelay_() {
    peer_relay_runtime_enabled_ = false;

    if (!partition_buffer_peer_relay_enabled()) {
        return;
    }

#if !defined(GEGE_CUDA)
    SPDLOG_WARN("GEGE_PARTITION_BUFFER_PEER_RELAY requested but GEGE_CUDA is not enabled; falling back to CPU-backed swaps");
    return;
#else
    if (devices_.size() <= 1) {
        SPDLOG_INFO("GEGE_PARTITION_BUFFER_PEER_RELAY requested but only {} physical CUDA device is active; falling back to CPU-backed swaps", devices_.size());
        return;
    }

    if (options_ != nullptr && options_->prefetching) {
        SPDLOG_WARN("GEGE_PARTITION_BUFFER_PEER_RELAY does not support partition prefetching; falling back to CPU-backed swaps");
        return;
    }

    for (std::size_t src = 0; src < devices_.size(); src++) {
        if (!devices_[src].is_cuda()) {
            SPDLOG_WARN("GEGE_PARTITION_BUFFER_PEER_RELAY requires CUDA devices only; falling back to CPU-backed swaps");
            return;
        }
        for (std::size_t dst = 0; dst < devices_.size(); dst++) {
            if (src == dst) {
                continue;
            }
            int can_access = 0;
            cudaError_t status = cudaDeviceCanAccessPeer(&can_access, devices_[src].index(), devices_[dst].index());
            if (status != cudaSuccess || can_access == 0) {
                SPDLOG_WARN("GEGE_PARTITION_BUFFER_PEER_RELAY disabled: CUDA peer access unavailable between device {} and {}",
                            devices_[src].index(), devices_[dst].index());
                return;
            }
        }
    }

    for (std::size_t src = 0; src < devices_.size(); src++) {
        c10::cuda::CUDAGuard device_guard(devices_[src]);
        for (std::size_t dst = 0; dst < devices_.size(); dst++) {
            if (src == dst) {
                continue;
            }
            cudaError_t status = cudaDeviceEnablePeerAccess(devices_[dst].index(), 0);
            if (status == cudaErrorPeerAccessAlreadyEnabled) {
                // cudaDeviceEnablePeerAccess may leave a sticky runtime error even
                // though repeated enable attempts are semantically harmless.
                // Clear it here so later kernel launch checks do not trip on this
                // stale status.
                cudaGetLastError();
                continue;
            }
            if (status != cudaSuccess) {
                SPDLOG_WARN("GEGE_PARTITION_BUFFER_PEER_RELAY disabled: failed to enable peer access from device {} to {} ({})",
                            devices_[src].index(), devices_[dst].index(), cudaGetErrorString(status));
                return;
            }
        }
    }

    // Ensure no sticky peer-access status leaks into unrelated CUDA work.
    cudaGetLastError();

    peer_relay_ready_barrier_ = std::make_unique<ReusableBarrier>(static_cast<int>(devices_.size()));
    peer_relay_build_barrier_ = std::make_unique<ReusableBarrier>(static_cast<int>(devices_.size()));
    peer_relay_next_states_.resize(devices_.size());
    peer_relay_staged_views_.resize(devices_.size());
    peer_relay_runtime_enabled_ = true;
    SPDLOG_INFO("Enabled GEGE_PARTITION_BUFFER_PEER_RELAY for {} CUDA devices; CPU backing store will be synchronized at unload/eval boundaries", devices_.size());
#endif
}

bool MemPartitionBufferStorage::peerRelayEnabled_() {
    std::call_once(peer_relay_init_once_, [this]() { initializePeerRelay_(); });
    return peer_relay_runtime_enabled_;
}


void MemPartitionBufferStorage::rangePut(int64_t offset, torch::Tensor values) {
    int fd = open(filename_.c_str(), O_RDWR | IO_FLAGS);
    if (fd == -1) {
        SPDLOG_ERROR("Unable to open {}\nError: {}", filename_, errno);
        throw std::runtime_error("");
    }

    int64_t dtype_size = get_dtype_size_wrapper(dtype_);
    int64_t ptr_offset = offset * dim1_size_ * dtype_size;

    if (pwrite_wrapper(fd, values.data_ptr(), values.size(0) * dim1_size_ * dtype_size, ptr_offset) == -1) {
        SPDLOG_ERROR("Unable to write {}\nError: {}", filename_, errno);
        throw std::runtime_error("");
    }

    close(fd);
}

void MemPartitionBufferStorage::append(torch::Tensor values) {
    ios::openmode flags;

    if (dim0_size_ == 0) {
        flags = ios::trunc | ios::binary;
    } else {
        flags = ios::binary | ios_base::app;
    }

    dim0_size_ += values.size(0);
    dim1_size_ = values.size(1);
    dtype_ = values.scalar_type();

    std::ofstream outfile(filename_, flags);

    int dtype_size = get_dtype_size_wrapper(dtype_);

    outfile.write((char *)values.data_ptr(), values.size(0) * values.size(1) * dtype_size);

    outfile.close();
}

MemPartitionBufferStorage::~MemPartitionBufferStorage() { 
    for(int i = 0; i < devices_.size(); i ++) {
        delete buffers_[i];
    }
}

void MemPartitionBufferStorage::load() {
    // SPDLOG_INFO("MemPartitionBufferStorage Loading {}", filename_);
    if (!loaded_ && !filename_.empty()) {
        fd_ = open((filename_).c_str(), O_RDWR);
        if (fd_ == -1) {
            SPDLOG_DEBUG("Unable to open {}\nError: {}", filename_, errno);
            return;
        }

        int64_t dtype_size = get_dtype_size_wrapper(dtype_);

        data_ = torch::empty({dim0_size_, dim1_size_}, dtype_);
        void* data_ptr_ = data_.data_ptr();
        int64_t offset = 0;
        int64_t read_size = dim0_size_ * dim1_size_ * dtype_size;

        if (pread_wrapper(fd_, data_.data_ptr(), read_size, offset) == -1) {
            SPDLOG_ERROR("Unable to read {}\nError: {}", filename_, errno);
            throw std::runtime_error("");
        }
        loaded_ = true;
    }
    
    for (int i = 0; i < buffers_.size(); i ++)
        buffers_[i]->load(data_);
}

void MemPartitionBufferStorage::write() {
    if (loaded_ && !filename_.empty()) { 
        int64_t dtype_size = get_dtype_size_wrapper(dtype_);

        torch::Tensor data = data_;
        data = data_.to(torch::kCPU);


        int64_t offset = 0;
        int64_t read_size = dim0_size_ * dim1_size_ * dtype_size;

        if (pwrite_wrapper(fd_, data.data_ptr(), read_size, offset) == -1) {
            SPDLOG_ERROR("Unable to read {}\nError: {}", filename_, errno);
            throw std::runtime_error("");
        }
    }
}

void MemPartitionBufferStorage::unload(bool perform_write) {
    if (loaded_) {
        for (int i = 0; i < buffers_.size(); i ++)
            buffers_[i]->unload(perform_write);

        if (perform_write) {
            write();
            close(fd_);
            data_ = torch::Tensor();
            loaded_ = false;
        }
    }
}

void MemPartitionBufferStorage::unload(bool perform_write, int32_t device_idx) {
    if (loaded_) {
        buffers_[device_idx]->unload(perform_write);
        if (perform_write) {
            write();
            close(fd_);
            data_ = torch::Tensor();
            loaded_ = false;
        }
    }
}

void MemPartitionBufferStorage::performNextSwap(int32_t device_idx) {
    if (!peerRelayEnabled_()) {
        buffers_[device_idx]->performNextSwap();
        return;
    }

#if !defined(GEGE_CUDA)
    buffers_[device_idx]->performNextSwap();
#else
    auto *buffer = buffers_[device_idx];
    if (!buffer->buffer_state_.defined() || buffer->buffer_state_iterator_ == buffer->buffer_states_.end()) {
        return;
    }

    {
        c10::cuda::CUDAGuard device_guard(buffer->device_);
        cudaDeviceSynchronize();
    }

    peer_relay_next_states_[device_idx] = (*buffer->buffer_state_iterator_).clone();
    peer_relay_staged_views_[device_idx] = torch::Tensor();
    peer_relay_ready_barrier_->arrive_and_wait();

    std::vector<int> current_owner(options_->num_partitions, -1);
    for (int src_dev = 0; src_dev < static_cast<int>(buffers_.size()); src_dev++) {
        auto *src_buffer = buffers_[src_dev];
        auto current_state = src_buffer->buffer_state_.to(torch::kCPU).to(torch::kInt64).contiguous();
        auto *state_ptr = current_state.data_ptr<int64_t>();
        for (int64_t i = 0; i < current_state.numel(); i++) {
            current_owner[state_ptr[i]] = src_dev;
        }
    }

    auto next_state = peer_relay_next_states_[device_idx].to(torch::kCPU).to(torch::kInt64).contiguous();
    // Every slot in the next buffer state is fully overwritten by relay copies,
    // so zero-filling the staging tensor just burns swap time.
    torch::Tensor staged_view = torch::empty_like(buffer->buffer_tensor_gpu_view_);
    {
        c10::cuda::CUDAGuard device_guard(buffer->device_);
        auto comm_stream = c10::cuda::getStreamFromPool(false, buffer->device_.index());
        c10::cuda::CUDAStreamGuard stream_guard(comm_stream);

        auto *next_state_ptr = next_state.data_ptr<int64_t>();
        for (int64_t slot = 0; slot < next_state.numel(); slot++) {
            int partition_id = static_cast<int>(next_state_ptr[slot]);
            int src_dev = current_owner[partition_id];
            if (src_dev < 0) {
                throw GegeRuntimeException(fmt::format("Peer relay could not find current owner for partition {}", partition_id));
            }

            auto *src_buffer = buffers_[src_dev];
            Partition *src_partition = src_buffer->partition_table_[partition_id];
            Partition *dst_partition = buffer->partition_table_[partition_id];
            int64_t src_slot = src_partition->buffer_idx_;
            int64_t rows = dst_partition->partition_size_;
            int64_t bytes = rows * dim1_size_ * get_dtype_size_wrapper(dtype_);

            void *dst_ptr = static_cast<char *>(staged_view.data_ptr()) + (slot * buffer->partition_size_ * dim1_size_ * get_dtype_size_wrapper(dtype_));
            void *src_ptr = static_cast<char *>(src_buffer->buffer_tensor_gpu_view_.data_ptr()) +
                            (src_slot * src_buffer->partition_size_ * dim1_size_ * get_dtype_size_wrapper(dtype_));

            cudaError_t status;
            if (src_dev == device_idx) {
                status = cudaMemcpyAsync(dst_ptr, src_ptr, bytes, cudaMemcpyDeviceToDevice, comm_stream.stream());
            } else {
                status = cudaMemcpyPeerAsync(dst_ptr, buffer->device_.index(), src_ptr, src_buffer->device_.index(), bytes, comm_stream.stream());
            }

            if (status != cudaSuccess) {
                throw GegeRuntimeException(fmt::format("Peer relay copy failed for partition {} from device {} to {}: {}",
                                                      partition_id, src_buffer->device_.index(), buffer->device_.index(),
                                                      cudaGetErrorString(status)));
            }
        }

        cudaStreamSynchronize(comm_stream.stream());
    }

    peer_relay_staged_views_[device_idx] = staged_view;
    peer_relay_build_barrier_->arrive_and_wait();

    auto previous_state = buffer->buffer_state_.to(torch::kCPU).to(torch::kInt64).contiguous();
    auto *previous_state_ptr = previous_state.data_ptr<int64_t>();
    for (int64_t i = 0; i < previous_state.numel(); i++) {
        int partition_id = static_cast<int>(previous_state_ptr[i]);
        Partition *partition = buffer->partition_table_[partition_id];
        partition->present_ = false;
        partition->buffer_idx_ = -1;
        partition->data_ptr_ = nullptr;
    }

    buffer->buffer_tensor_gpu_view_ = peer_relay_staged_views_[device_idx];
    buffer->buffer_state_ = *buffer->buffer_state_iterator_;
    for (int i = 0; i < buffer->buffer_sizes_; i++) {
        if (buffer->buffer_state_iterator_ != buffer->buffer_states_.end()) {
            buffer->buffer_state_iterator_++;
        }
    }

    int64_t num_rows = 0;
    auto current_state = buffer->buffer_state_.to(torch::kCPU).to(torch::kInt64).contiguous();
    auto *current_state_ptr = current_state.data_ptr<int64_t>();
    for (int64_t slot = 0; slot < current_state.numel(); slot++) {
        int partition_id = static_cast<int>(current_state_ptr[slot]);
        Partition *partition = buffer->partition_table_[partition_id];
        partition->present_ = true;
        partition->buffer_idx_ = slot;
        partition->data_ptr_ = nullptr;
        num_rows += partition->partition_size_;
    }
    buffer->size_.store(num_rows);
    buffer->loaded_ = true;
#endif
}

torch::Tensor MemPartitionBufferStorage::indexRead(Indices indices) { 
    if(device_ == torch::kCUDA) {
        return buffers_[0]->indexRead(indices);
    } else { 
        if (indices.sizes().size() != 1) {
            // TODO: throw invalid input to func exception
            throw std::runtime_error("");
        }

        if (data_.defined()) {
            return data_.index_select(0, indices.to(devices_[0]));
        } else {
            return torch::Tensor();
        }
    }
}

torch::Tensor MemPartitionBufferStorage::indexRead(Indices indices, int32_t device_idx) { 
    if(device_ == torch::kCUDA) {
        return buffers_[device_idx]->indexRead(indices);
    } else { 
        if (indices.sizes().size() != 1) {
            // TODO: throw invalid input to func exception
            throw std::runtime_error("");
        }
        // std::cout << data_.device() << std::endl;
        if (data_.defined()) {
            return data_.index_select(0, indices);
        } else {
            return torch::Tensor();
        }
    }
}


void MemPartitionBufferStorage::indexAdd(Indices indices, torch::Tensor values) { 
    return buffers_[0]->indexAdd(indices, values); 
}

void MemPartitionBufferStorage::indexAdd(Indices indices, torch::Tensor values, int32_t device_idx) { 
    return buffers_[device_idx]->indexAdd(indices, values); 
}

void MemPartitionBufferStorage::rangePut(int64_t offset, int64_t n, torch::Tensor values) {
    SPDLOG_ERROR("Unsupported operation for MemPartitionBufferStorage");
    throw std::runtime_error("");
}

torch::Tensor MemPartitionBufferStorage::range(int64_t offset, int64_t n) {
    SPDLOG_ERROR("Unsupported operation for MemPartitionBufferStorage");
    throw std::runtime_error("");
}

void MemPartitionBufferStorage::indexPut(Indices indices, torch::Tensor values) {
    SPDLOG_ERROR("Unsupported operation for MemPartitionBufferStorage");
    throw std::runtime_error("");
}

void MemPartitionBufferStorage::shuffle() {
    SPDLOG_ERROR("Unsupported operation for MemPartitionBufferStorage");
    throw std::runtime_error("");
};

void MemPartitionBufferStorage::sort(bool src) {
    SPDLOG_ERROR("Sort not supported for MemPartitionBufferStorage");
    throw std::runtime_error("");
};

FlatFile::FlatFile(string filename, int64_t dim0_size, int64_t dim1_size, torch::Dtype dtype, bool alloc) {
    filename_ = filename;
    dim0_size_ = dim0_size;
    dim1_size_ = dim1_size;
    dtype_ = dtype;
    initialized_ = true;
    loaded_ = false;
    device_ = torch::kCPU;

    if (alloc) {
        int64_t dtype_size = 0;

        if (dtype_ == torch::kFloat64) {
            dtype_size = 8;
        } else if (dtype_ == torch::kFloat32) {
            dtype_size = 4;
        } else if (dtype_ == torch::kFloat16) {
            dtype_size = 2;
        } else if (dtype_ == torch::kInt64) {
            dtype_size = 8;
        } else if (dtype_ == torch::kInt32) {
            dtype_size = 4;
        }

        std::ofstream ofs(filename_, std::ios::binary | std::ios::out);
        ofs.seekp(dim0_size_ * dim1_size_ * dtype_size - 1);
        ofs.write("", 1);
        ofs.close();
    }
}

FlatFile::FlatFile(string filename, torch::Tensor data) {
    filename_ = filename;
    dim0_size_ = 0;
    dim1_size_ = data.size(1);
    dtype_ = data.scalar_type();
    loaded_ = false;
    append(data);
    initialized_ = true;
    device_ = torch::kCPU;
}

FlatFile::FlatFile(string filename, torch::Dtype dtype) {
    filename_ = filename;
    dim0_size_ = 0;
    initialized_ = false;
    loaded_ = false;
    dtype_ = dtype;
    device_ = torch::kCPU;
}

void FlatFile::rangePut(int64_t offset, torch::Tensor values) {
    if (!values.defined() || (dim0_size_ != 0 && (values.size(0) + offset > dim0_size_ || values.size(1) != dim1_size_))) {
        // TODO: throw invalid inputs for function error
        throw std::runtime_error("");
    }

    int64_t dtype_size = get_dtype_size_wrapper(dtype_);

    int64_t ptr_offset = offset * dim1_size_ * dtype_size;

    if (pwrite_wrapper(fd_, values.data_ptr(), values.size(0) * dim1_size_ * dtype_size, ptr_offset) == -1) {
        SPDLOG_ERROR("Unable to write {}\nError: {}", filename_, errno);
        throw std::runtime_error("");
    }
}

void FlatFile::append(torch::Tensor values) {
    ios::openmode flags = dim0_size_ == 0 ? ios::trunc | ios::binary : ios::binary | ios_base::app;

    dim0_size_ += values.size(0);
    dim1_size_ = values.size(1);
    dtype_ = values.scalar_type();

    std::ofstream outfile(filename_, flags);

    int64_t dtype_size = get_dtype_size_wrapper(dtype_);

    outfile.write((char *)values.data_ptr(), values.size(0) * values.size(1) * dtype_size);
    outfile.close();
}

void FlatFile::load() {
    if (!loaded_ && initialized_) {
        fd_ = open(filename_.c_str(), O_RDWR | IO_FLAGS);
        if (fd_ == -1) {
            SPDLOG_DEBUG("Unable to open {}\nError: {}", filename_, errno);
            return;
        }
        loaded_ = true;
    }
}

void FlatFile::write() { return; }

void FlatFile::unload(bool perform_write) {
    (void)perform_write;
    if (loaded_) {
        close(fd_);
        loaded_ = false;
    }
}

torch::Tensor FlatFile::indexRead(Indices indices) {
    SPDLOG_ERROR("Unsupported operation for FlatFile, only sequential access is supported");
    throw std::runtime_error("");
}

void FlatFile::indexAdd(Indices indices, torch::Tensor values) {
    SPDLOG_ERROR("Unsupported operation for FlatFile, only sequential access is supported");
    throw std::runtime_error("");
}

void FlatFile::indexPut(Indices indices, torch::Tensor values) {
    SPDLOG_ERROR("Unsupported operation for FlatFile, only sequential access is supported");
    throw std::runtime_error("");
}

void FlatFile::move(string new_filename) {
    unload(false);

    renameFile(filename_, new_filename);

    load();
}

void FlatFile::copy(string new_filename, bool rename) {
    unload(false);

    copyFile(filename_, new_filename);

    if (rename) {
        filename_ = new_filename;
    }
    load();
}

torch::Tensor FlatFile::range(int64_t offset, int64_t n) {
    if (n + offset > dim0_size_) {
        // TODO: throw invalid inputs for function error
        throw std::runtime_error("");
    }
    int dtype_size = get_dtype_size_wrapper(dtype_);

    int64_t ptr_offset = offset * dim1_size_ * dtype_size;

    torch::Tensor output_tensor = torch::empty({n, dim1_size_}, dtype_);
    if (pread_wrapper(fd_, output_tensor.data_ptr(), n * dim1_size_ * dtype_size, ptr_offset) == -1) {
        SPDLOG_ERROR("Unable to read {}\nError: {}", filename_, errno);
        throw std::runtime_error("");
    }
    return output_tensor;
}

void FlatFile::rangePut(int64_t offset, int64_t n, torch::Tensor values) {
    int dtype_size = get_dtype_size_wrapper(dtype_);

    int64_t ptr_offset = offset * dim1_size_ * dtype_size;

    if (pwrite_wrapper(fd_, values.data_ptr(), n * dim1_size_ * dtype_size, ptr_offset) == -1) {
        SPDLOG_ERROR("Unable to write {}\nError: {}", filename_, errno);
        throw std::runtime_error("");
    }
}

void FlatFile::shuffle() {
    bool loaded = loaded_;
    if (!loaded) {
        load();
    }
    if (edge_bucket_sizes_.empty()) {
        int64_t offset = 0;
        int64_t curr_size = 0;
        while (offset < dim0_size_) {
            if (dim0_size_ - offset < MAX_SHUFFLE_SIZE) {
                curr_size = dim0_size_ - offset;
            } else {
                curr_size = MAX_SHUFFLE_SIZE;
            }
            torch::Tensor chunk = range(offset, curr_size);
            auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
            chunk.copy_(chunk.index_select(0, torch::randperm(chunk.size(0), opts)));
            rangePut(offset, chunk);
            offset += curr_size;
        }
    } else {
        int64_t offset = 0;
        auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
        for (auto itr = edge_bucket_sizes_.begin(); itr != edge_bucket_sizes_.end(); itr++) {
            torch::Tensor edge_bucket = range(offset, *itr);
            edge_bucket.copy_(edge_bucket.index_select(0, torch::randperm(edge_bucket.size(0), opts)));
            rangePut(offset, edge_bucket);
            offset += *itr;
        }
    }
    if (!loaded) {
        unload(true);
    }
}

void FlatFile::sort(bool src) {
    // function for sorting flat file storing edges
    int sort_dim = 0;
    if (!src) {
        sort_dim = -1;
    }

    bool loaded = loaded_;
    if (!loaded) {
        load();
    }
    if (edge_bucket_sizes_.empty()) {
        int64_t offset = 0;
        int64_t curr_size = 0;
        while (offset < dim0_size_) {
            if (dim0_size_ - offset < MAX_SORT_SIZE) {
                curr_size = dim0_size_ - offset;
            } else {
                curr_size = MAX_SORT_SIZE;
            }

            torch::Tensor chunk = range(offset, curr_size);
            // auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
            chunk.copy_(chunk.index_select(0, torch::argsort(chunk.select(1, sort_dim))));
            rangePut(offset, chunk);
            offset += curr_size;
        }
    } else {
        int64_t offset = 0;
        // auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
        for (auto itr = edge_bucket_sizes_.begin(); itr != edge_bucket_sizes_.end(); itr++) {
            torch::Tensor edge_bucket = range(offset, *itr);
            edge_bucket.copy_(edge_bucket.index_select(0, torch::argsort(edge_bucket.select(1, sort_dim))));
            rangePut(offset, edge_bucket);
            offset += *itr;
        }
    }
    if (!loaded) {
        unload(true);
    }
}

void FlatFile::mem_load() {
    if (!loaded_) {
        fd_ = open((filename_).c_str(), O_RDWR);
        if (fd_ == -1) {
            SPDLOG_ERROR("Unable to open {}\nError: {}", filename_, errno);
            throw std::runtime_error("");
        }

        int64_t dtype_size = get_dtype_size_wrapper(dtype_);

        data_ = torch::empty({dim0_size_, dim1_size_}, dtype_);
        SPDLOG_DEBUG("Initialized memory edges");
        process_mem_usage();

        int64_t offset = 0;
        int64_t read_size = dim0_size_ * dim1_size_ * dtype_size;

        if (pread_wrapper(fd_, data_.data_ptr(), read_size, offset) == -1) {
            SPDLOG_ERROR("Unable to read {}\nError: {}", filename_, errno);
            throw std::runtime_error("");
        }

        SPDLOG_DEBUG("Read edges from disk");
        process_mem_usage();

        loaded_ = true;
    }
}

void FlatFile::mem_unload(bool write) {
    if (loaded_) {
        int64_t dtype_size = get_dtype_size_wrapper(dtype_);

        int64_t offset = 0;
        int64_t read_size = dim0_size_ * dim1_size_ * dtype_size;

        if (write) {
            if (pwrite_wrapper(fd_, data_.data_ptr(), read_size, offset) == -1) {
                SPDLOG_ERROR("Unable to write {}\nError: {}", filename_, errno);
                throw std::runtime_error("");
            }
        }

        close(fd_);

        SPDLOG_DEBUG("Edges written");
        process_mem_usage();
        loaded_ = false;
        process_mem_usage();
        data_ = torch::Tensor();
        SPDLOG_DEBUG("Nulled tensor and pointer");
        process_mem_usage();
    }
}

InMemory::InMemory(string filename, int64_t dim0_size, int64_t dim1_size, torch::Dtype dtype, torch::Device device) {
    filename_ = filename;
    dim0_size_ = dim0_size;
    dim1_size_ = dim1_size;
    dtype_ = dtype;
    initialized_ = true;
    loaded_ = false;
    device_ = device;
}

InMemory::InMemory(string filename, torch::Tensor data, torch::Device device) {
    filename_ = filename;
    dim0_size_ = data.size(0);
    dim1_size_ = data.size(1);
    dtype_ = data.scalar_type();
    device_ = device;
    loaded_ = false;

    torch::Tensor temp = data.to(torch::kCPU);

    std::ofstream outfile(filename_, ios::out | ios::binary);

    int64_t dtype_size = get_dtype_size_wrapper(dtype_);

    outfile.write((char *)temp.data_ptr(), data.size(0) * data.size(1) * dtype_size);

    outfile.close();
}

InMemory::InMemory(string filename, torch::Dtype dtype) {
    filename_ = filename;
    dim0_size_ = 0;
    dim1_size_ = 0;
    initialized_ = false;
    dtype_ = dtype;
    device_ = torch::kCPU;
    loaded_ = false;
}

InMemory::InMemory(torch::Tensor data) {
    if (data.sizes().size() == 2) {
        dim0_size_ = data.size(0);
        dim1_size_ = data.size(1); 
    } else if (data.sizes().size() == 1) {
        dim0_size_ = data.size(0);
        dim1_size_ = 1;
    } else {
        throw GegeRuntimeException("Tensor must have 1 or two dimensions");
    }

    filename_ = "";
    data_ = data.reshape({dim0_size_, dim1_size_});

    initialized_ = true;
    dtype_ = data.scalar_type();
    device_ = data.device();
    loaded_ = true;
}

void InMemory::load() {
    if (!loaded_ && !filename_.empty()) {
        fd_ = open((filename_).c_str(), O_RDWR);
        if (fd_ == -1) {
            SPDLOG_DEBUG("Unable to open {}\nError: {}", filename_, errno);
            return;
        }
        int64_t dtype_size = get_dtype_size_wrapper(dtype_);

        data_ = torch::empty({dim0_size_, dim1_size_}, dtype_);

        int64_t offset = 0;
        int64_t read_size = dim0_size_ * dim1_size_ * dtype_size;

        if (pread_wrapper(fd_, data_.data_ptr(), read_size, offset) == -1) {
            SPDLOG_ERROR("Unable to read {}\nError: {}", filename_, errno);
            throw std::runtime_error("");
        }

        if (device_ == torch::kCUDA) {
            data_ = data_.to(device_);
        }

        loaded_ = true;
    }
}

void InMemory::write() {
    if (loaded_ && !filename_.empty()) {
        int64_t dtype_size = get_dtype_size_wrapper(dtype_);

        torch::Tensor data = data_;
        if (device_ == torch::kCUDA) {
            data = data_.to(torch::kCPU);
        }

        int64_t offset = 0;
        int64_t read_size = dim0_size_ * dim1_size_ * dtype_size;

        if (pwrite_wrapper(fd_, data.data_ptr(), read_size, offset) == -1) {
            SPDLOG_ERROR("Unable to read {}\nError: {}", filename_, errno);
            throw std::runtime_error("");
        }
    }
}

void InMemory::unload(bool perform_write) {
    if (loaded_ && !filename_.empty()) {
        if (perform_write) {
            write();
        }

        // close(fd_);
        // loaded_ = false;
        // data_ = torch::Tensor();
    }
}

torch::Tensor InMemory::indexRead(Indices indices) {
    if (indices.sizes().size() != 1) {
        // TODO: throw invalid input to func exception
        throw std::runtime_error("");
    }

    if (data_.defined()) {
        return data_.index_select(0, indices.to(device_));
    } else {
        return torch::Tensor();
    }
}

void InMemory::indexAdd(Indices indices, torch::Tensor values) {
    if (!values.defined() || indices.sizes().size() != 1 || indices.size(0) != values.size(0) || data_.size(1) != values.size(1)) {
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
            ScopedNvtxRange nvtx_scope("storage.InMemory.indexAdd.csr");
            static bool logged = false;
            if (!logged) {
                SPDLOG_INFO("InMemory::indexAdd using direct CSR update path");
                logged = true;
            }
            torch::Tensor update_indices = indices;
            torch::Tensor update_values = values;
            if (csr_update_reduce_enabled()) {
                std::tie(update_indices, update_values) = reduce_updates_with_csr(indices, values);
                if (run_stage_debug) {
                    auto now = std::chrono::high_resolution_clock::now();
                    SPDLOG_INFO("[stage-debug][storage.indexAdd][update {}][step 1] csr_reduce ms={:.3f} in_rows={} out_rows={}",
                                debug_update_id, elapsed_ms(step_start, now), indices.numel(), update_indices.numel());
                    step_start = now;
                }
            }
            data_.index_add_(0, update_indices, update_values);
            if (run_stage_debug) {
                auto now = std::chrono::high_resolution_clock::now();
                SPDLOG_INFO("[stage-debug][storage.indexAdd][update {}][step 2] index_add_cuda ms={:.3f} rows={} dim={}",
                            debug_update_id, elapsed_ms(step_start, now), update_indices.numel(), update_values.size(1));
            }
        } else {
            data_.index_add_(0, indices, values);
            if (run_stage_debug) {
                auto now = std::chrono::high_resolution_clock::now();
                SPDLOG_INFO("[stage-debug][storage.indexAdd][update {}][step 1] index_add_cuda ms={:.3f} rows={} dim={} csr_update={}",
                            debug_update_id, elapsed_ms(step_start, now), indices.numel(), values.size(1), false);
            }
        }
#else
        data_.index_add_(0, indices, values);
#endif
    } else {
        // assumes this operation is only used on float valued data.
        auto data_accessor = data_.accessor<float, 2>();
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
            SPDLOG_INFO("[stage-debug][storage.indexAdd][update {}][step 1] index_add_cpu ms={:.3f} rows={} dim={}",
                        debug_update_id, elapsed_ms(step_start, now), size, d);
        }
    }

    if (run_stage_debug) {
        auto now = std::chrono::high_resolution_clock::now();
        SPDLOG_INFO("[stage-debug][storage.indexAdd][update {}][step 9] total_ms={:.3f} device={}",
                    debug_update_id, elapsed_ms(index_add_start, now), values.device().str());
    }
}

void InMemory::indexPut(Indices indices, torch::Tensor values) {
    if (!values.defined() || indices.sizes().size() != 1 || indices.size(0) != values.size(0) || data_.size(1) != values.size(1)) {
        // TODO: throw invalid input to func exception
        throw std::runtime_error("");
    }
    if (values.device().is_cuda()) {
        data_[indices] = values;
    } else {
        // assumes this operation is only used on float valued data.
        auto data_accessor = data_.accessor<float, 2>();
        auto ids_accessor = indices.accessor<int64_t, 1>();
        auto values_accessor = values.accessor<float, 2>();

        int d = values.size(1);
        int64_t size = indices.size(0);
#pragma omp parallel for
        for (int64_t i = 0; i < size; i++) {
            for (int j = 0; j < d; j++) {
                data_accessor[ids_accessor[i]][j] = values_accessor[i][j];
            }
        }
    }
}

torch::Tensor InMemory::range(int64_t offset, int64_t n) {
    if (n + offset > dim0_size_) {
        // TODO: throw invalid inputs for function error
        throw std::runtime_error("");
    }
    return data_.narrow(0, offset, n);
}

void InMemory::rangePut(int64_t offset, int64_t n, torch::Tensor values) { data_.narrow(0, offset, n).copy_(values); }

void InMemory::shuffle() {
    bool loaded = loaded_;
    if (!loaded) {
        load();

        // may cause silent failures
        if (!loaded_) {
            return;
        }
    }

    // full shuffle
    if (edge_bucket_sizes_.empty()) {
        auto opts = torch::TensorOptions().dtype(torch::kInt64).device(data_.device());
        data_.copy_(data_.index_select(0, torch::randperm(dim0_size_, opts)));
    }
    // shuffle within edge buckets
    else {
        int64_t start = 0;
        auto opts = torch::TensorOptions().dtype(torch::kInt64).device(data_.device());
        for (auto itr = edge_bucket_sizes_.begin(); itr != edge_bucket_sizes_.end(); itr++) {
            torch::Tensor edge_bucket = data_.narrow(0, start, *itr);
            data_.narrow(0, start, *itr) = (edge_bucket.index_select(0, torch::randperm(edge_bucket.size(0), opts)));
            start += *itr;
        }
    }
    // if (!loaded) {
    //     unload(true);
    // }
}

// void InMemory::shuffle() {
//     auto opts = torch::TensorOptions().dtype(torch::kInt64).device(data_.device());
//     torch::Tenosr perm = torch::randperm(dim0_size_, opts);


// }

void InMemory::sort(bool src) {
    // function for sorting in memory edges
    int sort_dim = 0;
    if (!src) {
        sort_dim = -1;
    }

    bool loaded = loaded_;
    if (!loaded) {
        load();

        // may cause silent failures
        if (!loaded_) {
            return;
        }
    }

    // full sort
    if (edge_bucket_sizes_.empty()) {
        // auto opts = torch::TensorOptions().dtype(torch::kInt64).device(data_.device());
        data_.copy_(data_.index_select(0, torch::argsort(data_.select(1, sort_dim))));
    }
    // sort within edge buckets
    else {
        int64_t start = 0;
        // auto opts = torch::TensorOptions().dtype(torch::kInt64).device(data_.device());
        for (auto itr = edge_bucket_sizes_.begin(); itr != edge_bucket_sizes_.end(); itr++) {
            torch::Tensor edge_bucket = data_.narrow(0, start, *itr);
            data_.narrow(0, start, *itr) = (edge_bucket.index_select(0, torch::argsort(edge_bucket.select(1, sort_dim))));
            start += *itr;
        }
    }
    if (!loaded) {
        unload(true);
    }
}
