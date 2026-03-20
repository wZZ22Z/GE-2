#pragma once

#include "common/datatypes.h"

struct UniqueMapCudaDebugInfo {
    std::string requested_backend;
    std::string executed_backend;
    std::string fallback_backend;
    std::string fallback_reason;
    int64_t total_calls = 0;
    int64_t total_fallbacks = 0;
    double device_unique_ms = 0.0;
    bool used_fallback = false;
    bool cuco_compiled = false;
    bool device_unique_timing_valid = false;
    bool measure_device_timing = false;
    bool sorted_request = false;
};

// Returns {unique_ids, inverse_indices} for a 1D CUDA int64 tensor.
// Ordering of unique_ids depends on the selected backend.
std::tuple<torch::Tensor, torch::Tensor> map_tensors_unique_inverse_cuda(torch::Tensor all_ids, bool sorted,
                                                                         UniqueMapCudaDebugInfo *debug_info = nullptr,
                                                                         int64_t value_domain_size = -1);
