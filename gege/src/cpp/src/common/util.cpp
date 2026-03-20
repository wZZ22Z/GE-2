#include "common/util.h"

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>

#include "reporting/logger.h"
#ifdef GEGE_CUDA
#include "common/unique_map_cuda.h"
#endif

namespace {

struct AllIdsBufferCache {
    torch::Tensor storage;
    int64_t capacity = 0;
};

thread_local AllIdsBufferCache all_ids_buffer_cache;

std::string parse_env_string(const char *name, const std::string &default_value = "") {
    const char *raw = std::getenv(name);
    return raw == nullptr ? default_value : std::string(raw);
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

const std::string &unique_capture_dir() {
    static std::string dir = parse_env_string("GEGE_UNIQUE_CAPTURE_DIR");
    return dir;
}

int64_t unique_capture_max_batches() {
    static int64_t max_batches = std::max<int64_t>(parse_env_int("GEGE_UNIQUE_CAPTURE_MAX_BATCHES", 0), 0);
    return max_batches;
}

bool unique_capture_enabled() {
    return !unique_capture_dir().empty() && unique_capture_max_batches() > 0;
}

std::atomic<int64_t> &unique_capture_counter() {
    static std::atomic<int64_t> counter{0};
    return counter;
}

std::once_flag &unique_capture_dir_once_flag() {
    static std::once_flag flag;
    return flag;
}

std::string capture_unique_input_batch(const torch::Tensor &all_ids) {
    if (!unique_capture_enabled()) {
        return "";
    }

    int64_t capture_idx = unique_capture_counter().fetch_add(1);
    if (capture_idx >= unique_capture_max_batches()) {
        return "";
    }

    std::call_once(unique_capture_dir_once_flag(), []() { std::filesystem::create_directories(unique_capture_dir()); });

    auto cpu_ids = all_ids.detach().to(torch::kCPU).contiguous();
    std::ostringstream path_builder;
    path_builder << unique_capture_dir() << "/batch_" << std::setw(6) << std::setfill('0') << capture_idx << "_n" << cpu_ids.numel() << ".pt";
    std::string capture_path = path_builder.str();
    torch::save(cpu_ids, capture_path);
    return capture_path;
}

void maybe_log_unique_backend_banner(const UniqueMapCudaDebugInfo &debug_info) {
    static std::once_flag flag;
    std::call_once(flag, [&debug_info]() {
        SPDLOG_INFO("[unique-map] requested_backend={} initial_backend={} cuco_compiled={} capture_dir={} capture_max_batches={}",
                    debug_info.requested_backend, debug_info.executed_backend, debug_info.cuco_compiled, unique_capture_dir(),
                    unique_capture_max_batches());
    });
}

std::tuple<torch::Tensor, torch::Tensor> unique_with_inverse_compat(torch::Tensor ids) {
    torch::Tensor ids64 = ids.to(torch::kInt64);
    auto sort_tup = torch::sort(ids64, 0, false);
    torch::Tensor sorted_ids = std::get<0>(sort_tup);
    torch::Tensor perm = std::get<1>(sort_tup).to(torch::kInt64);
    auto unique_tup = torch::unique_consecutive(sorted_ids, false, true);
    torch::Tensor unique_ids = std::get<0>(unique_tup);
    torch::Tensor inverse_sorted = std::get<1>(unique_tup).to(torch::kInt64);
    torch::Tensor inverse = torch::empty_like(inverse_sorted);
    inverse.scatter_(0, perm, inverse_sorted);
    return std::forward_as_tuple(unique_ids, inverse);
}

torch::Tensor pack_ids_into_cached_buffer(const std::vector<torch::Tensor> &unmapped_tensors) {
    if (unmapped_tensors.empty()) {
        throw GegeRuntimeException("Input tensors must not be empty");
    }

    auto options = unmapped_tensors.front().options();
    auto dtype = unmapped_tensors.front().scalar_type();
    auto device = unmapped_tensors.front().device();

    int64_t total_numel = 0;
    for (const auto &tensor : unmapped_tensors) {
        if (tensor.scalar_type() != dtype || tensor.device() != device) {
            throw GegeRuntimeException("Input tensors must share dtype and device");
        }
        total_numel += tensor.numel();
    }

    if (total_numel == 0) {
        return torch::empty({0}, options);
    }

    bool need_new_storage = !all_ids_buffer_cache.storage.defined() || all_ids_buffer_cache.storage.scalar_type() != dtype ||
                            all_ids_buffer_cache.storage.device() != device || all_ids_buffer_cache.capacity < total_numel;
    if (need_new_storage) {
        int64_t next_capacity = total_numel;
        if (all_ids_buffer_cache.capacity > 0 && all_ids_buffer_cache.storage.defined() &&
            all_ids_buffer_cache.storage.scalar_type() == dtype && all_ids_buffer_cache.storage.device() == device) {
            next_capacity = std::max(total_numel, all_ids_buffer_cache.capacity * 2);
        }

        all_ids_buffer_cache.storage = torch::empty({next_capacity}, options);
        all_ids_buffer_cache.capacity = next_capacity;
    }

    torch::Tensor all_ids = all_ids_buffer_cache.storage.narrow(0, 0, total_numel);
    int64_t offset = 0;
    for (const auto &tensor : unmapped_tensors) {
        TORCH_CHECK(tensor.dim() == 1, "Input tensors must be 1D");
        int64_t tensor_numel = tensor.numel();
        auto t1d = tensor.is_contiguous() ? tensor : tensor.contiguous();
        all_ids.narrow(0, offset, tensor_numel).copy_(t1d,true);
        //all_ids.narrow(0, offset, tensor_numel).copy_(tensor.reshape({tensor_numel}));
        offset += tensor_numel;
    }

    return all_ids;
}

}  // namespace

void assert_no_nans(torch::Tensor values) {
    if (torch::isnan(values).any().item<bool>()) {
        throw GegeRuntimeException("Tensor contains Nans");
    }
}

void assert_no_neg(torch::Tensor values) {
    if ((values.le(-1)).any().item<bool>()) {
        throw GegeRuntimeException("Tensor contains negative values");
    }
}

void assert_in_range(torch::Tensor values, int64_t start, int64_t end) {
    if ((values.lt(start) | values.gt(end)).any().item<bool>()) {
        throw GegeRuntimeException("Tensor contains value out of range: ...");
    }
}

void process_mem_usage() {
    double vm_usage = 0.0;
    double resident_set = 0.0;

    // the two fields we want
    unsigned long vsize;
    long rss;
    {
        std::string ignore;
        std::ifstream ifs("/proc/self/stat", std::ios_base::in);
        ifs >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >>
            ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> vsize >> rss;
    }

    long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024;  // in case x86-64 is configured to use 2MB pages
    vm_usage = vsize / 1024.0;
    resident_set = rss * page_size_kb;

    SPDLOG_DEBUG("VM Usage: {}GB. RSS: {}GB", vm_usage / pow(2, 20), resident_set / pow(2, 20));
}

void *memset_wrapper(void *ptr, int value, int64_t num) {
    int64_t curr_bytes = 0;
    int64_t local_offset = 0;

    while (local_offset < num) {
        curr_bytes = num - local_offset;
        if (curr_bytes > 1e9) {
            curr_bytes = 1e9;
        }

        memset((char *)ptr + local_offset, value, curr_bytes);

        local_offset += curr_bytes;
    }

    return ptr;
}

void *memcpy_wrapper(void *dest, const void *src, int64_t count) {
    int64_t curr_bytes = 0;
    int64_t local_offset = 0;

    while (local_offset < count) {
        curr_bytes = count - local_offset;
        if (curr_bytes > 1e9) {
            curr_bytes = 1e9;
        }

        memcpy((char *)dest + local_offset, (char *)src + local_offset, curr_bytes);

        local_offset += curr_bytes;
    }

    return dest;
}

int64_t pread_wrapper(int fd, void *buf, int64_t count, int64_t offset) {
    int64_t curr_bytes = 0;
    int64_t local_offset = 0;

    while (local_offset < count) {
        curr_bytes = count - local_offset;
        if (curr_bytes > 1e9) {
            curr_bytes = 1e9;
        }

        if (pread(fd, (char *)buf + local_offset, curr_bytes, offset + local_offset) == -1) {
            return -1;
        }

        local_offset += curr_bytes;
    }

    return count;
}

int64_t pwrite_wrapper(int fd, const void *buf, int64_t count, int64_t offset) {
    int64_t curr_bytes = 0;
    int64_t local_offset = 0;

    while (local_offset < count) {
        curr_bytes = count - local_offset;
        if (curr_bytes > 1e9) {
            curr_bytes = 1e9;
        }

        if (pwrite(fd, (char *)buf + local_offset, curr_bytes, offset + local_offset) == -1) {
            return -1;
        }

        local_offset += curr_bytes;
    }

    return count;
}

int64_t get_dtype_size_wrapper(torch::Dtype dtype_) {
    if (dtype_ == torch::kFloat64) {
        return 8;
    }
    if (dtype_ == torch::kFloat32) {
        return 4;
    }
    if (dtype_ == torch::kFloat16) {
        return 2;
    }
    if (dtype_ == torch::kInt64) {
        return 8;
    }
    if (dtype_ == torch::kInt32) {
        return 4;
    }

    SPDLOG_ERROR("Unable to determine dtype_size_ for given dtype_ {}", dtype_);
    throw std::runtime_error("");
}

std::string get_directory(std::string filename) {
    assert(!filename.empty());

    string directory;
    const size_t last_slash_idx = filename.rfind('/');
    if (std::string::npos != last_slash_idx) {
        directory = filename.substr(0, last_slash_idx);
    }

    return directory;
}

std::tuple<torch::Tensor, std::vector<torch::Tensor>> map_tensors(std::vector<torch::Tensor> unmapped_tensors, bool sorted, MapTensorTiming *timing,
                                                                  int64_t value_domain_size) {
    auto total_start = std::chrono::high_resolution_clock::now();
    auto step_start = total_start;

    for (const auto &tensor : unmapped_tensors) {
        if (tensor.sizes().size() > 1) {
            throw GegeRuntimeException("Input tensors must be 1D");
        }
    }
    if (timing != nullptr) {
        auto now = std::chrono::high_resolution_clock::now();
        timing->validate_ms = std::chrono::duration<double, std::milli>(now - step_start).count();
        step_start = now;
    }

    torch::Tensor all_ids = pack_ids_into_cached_buffer(unmapped_tensors);
    std::string capture_path = capture_unique_input_batch(all_ids);
    if (timing != nullptr) {
        auto now = std::chrono::high_resolution_clock::now();
        timing->cat_ms = std::chrono::duration<double, std::milli>(now - step_start).count();
        step_start = now;
    }

    torch::Tensor map;
    torch::Tensor mapped_all_ids;
    UniqueMapCudaDebugInfo unique_debug_info;
    unique_debug_info.measure_device_timing = timing != nullptr;
#ifdef GEGE_CUDA
    if (all_ids.is_cuda() && all_ids.scalar_type() == torch::kInt64) {
        auto unique_tup = map_tensors_unique_inverse_cuda(all_ids, sorted, &unique_debug_info, value_domain_size);
        map = std::get<0>(unique_tup);
        mapped_all_ids = std::get<1>(unique_tup);
        maybe_log_unique_backend_banner(unique_debug_info);
    } else {
        auto unique_tup = unique_with_inverse_compat(all_ids);
        map = std::get<0>(unique_tup);
        mapped_all_ids = std::get<1>(unique_tup);
    }
#else
    auto unique_tup = unique_with_inverse_compat(all_ids);
    map = std::get<0>(unique_tup);
    mapped_all_ids = std::get<1>(unique_tup);
#endif
    if (timing != nullptr) {
        auto now = std::chrono::high_resolution_clock::now();
        timing->unique_wall_ms = std::chrono::duration<double, std::milli>(now - step_start).count();
        timing->unique_ms = unique_debug_info.device_unique_timing_valid ? unique_debug_info.device_unique_ms : timing->unique_wall_ms;
        timing->unique_requested_backend = unique_debug_info.requested_backend;
        timing->unique_executed_backend = unique_debug_info.executed_backend;
        timing->unique_fallback_backend = unique_debug_info.fallback_backend;
        timing->unique_fallback_reason = unique_debug_info.fallback_reason;
        timing->capture_path = capture_path;
        timing->unique_total_calls = unique_debug_info.total_calls;
        timing->unique_total_fallbacks = unique_debug_info.total_fallbacks;
        timing->unique_used_fallback = unique_debug_info.used_fallback;
        timing->unique_cuco_compiled = unique_debug_info.cuco_compiled;
        step_start = now;
    }

    std::vector<torch::Tensor> mapped_tensors;

    int64_t offset = 0;
    int64_t size;
    for (const auto &tensor : unmapped_tensors) {
        size = tensor.size(0);
        mapped_tensors.emplace_back(mapped_all_ids.narrow(0, offset, size));
        offset += size;
    }
    if (timing != nullptr) {
        auto now = std::chrono::high_resolution_clock::now();
        timing->split_ms = std::chrono::duration<double, std::milli>(now - step_start).count();
        timing->total_ms = std::chrono::duration<double, std::milli>(now - total_start).count();
    }

    return std::forward_as_tuple(map, mapped_tensors);
}

// TODO this function uses a searchsorted to find the approriate value in the map tensor
// this can be made faster on the cpu by using an std::map to perform lookups
std::vector<torch::Tensor> apply_tensor_map(torch::Tensor map, std::vector<torch::Tensor> unmapped_tensors) {
    for (auto tensor : unmapped_tensors) {
        if (tensor.sizes().size() > 1) {
            throw GegeRuntimeException("Input tensors must be 1D");
        }
    }

    std::vector<torch::Tensor> mapped_tensors;

    for (auto tensor : unmapped_tensors) {
        mapped_tensors.emplace_back(torch::searchsorted(map, tensor));
    }

    return mapped_tensors;
}
