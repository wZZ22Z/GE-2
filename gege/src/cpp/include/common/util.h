#pragma once

#include "datatypes.h"

class Timer {
   public:
    bool gpu_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
    std::chrono::time_point<std::chrono::high_resolution_clock> stop_time_;
    CudaEvent *start_event_;
    CudaEvent *end_event_;

    Timer(bool gpu) {
        start_event_ = new CudaEvent(0);
        end_event_ = new CudaEvent(0);
        gpu_ = gpu;
    }

    ~Timer() {
        delete start_event_;
        delete end_event_;
    }

    void start() {
        start_time_ = std::chrono::high_resolution_clock::now();
        if (gpu_) {
            start_event_->record();
        }
    }

    void stop() {
        stop_time_ = std::chrono::high_resolution_clock::now();
        if (gpu_) {
            end_event_->record();
        }
    }

    int64_t getDuration(bool ms = true) {
        int64_t duration;
        if (ms) {
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time_ - start_time_).count();
        } else {
            duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_time_ - start_time_).count();
        }

        if (gpu_) {
            start_event_->synchronize();
            end_event_->synchronize();
            duration = start_event_->elapsed_time(*end_event_);
        }
        return duration;
    }
};

bool has_nans(torch::Tensor values);

void assert_no_nans(torch::Tensor values);

void assert_no_neg(torch::Tensor values);

void assert_in_range(torch::Tensor values, int64_t start, int64_t end);

void process_mem_usage();

void *memset_wrapper(void *ptr, int value, int64_t num);

void *memcpy_wrapper(void *dest, const void *src, int64_t count);

int64_t pread_wrapper(int fd, void *buf, int64_t count, int64_t offset);

int64_t pwrite_wrapper(int fd, const void *buf, int64_t count, int64_t offset);

int64_t get_dtype_size_wrapper(torch::Dtype dtype_);

std::string get_directory(std::string path);

template <typename T1, typename T2>
bool instance_of(std::shared_ptr<T1> instance) {
    return (std::dynamic_pointer_cast<T2>(instance) != nullptr);
}

struct MapTensorTiming {
    double validate_ms = 0.0;
    double cat_ms = 0.0;
    double unique_ms = 0.0;
    double unique_wall_ms = 0.0;
    double split_ms = 0.0;
    double total_ms = 0.0;
    std::string unique_requested_backend;
    std::string unique_executed_backend;
    std::string unique_fallback_backend;
    std::string unique_fallback_reason;
    std::string capture_path;
    int64_t unique_total_calls = 0;
    int64_t unique_total_fallbacks = 0;
    bool unique_used_fallback = false;
    bool unique_cuco_compiled = false;
};

std::tuple<torch::Tensor, std::vector<torch::Tensor>> map_tensors(std::vector<torch::Tensor> unmapped_tensors, bool sorted = true,
                                                                   MapTensorTiming *timing = nullptr, int64_t value_domain_size = -1);

std::vector<torch::Tensor> apply_tensor_map(torch::Tensor map, std::vector<torch::Tensor> unmapped_tensors);
