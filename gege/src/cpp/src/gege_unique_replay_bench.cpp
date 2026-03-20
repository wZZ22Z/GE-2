#include "common/unique_map_cuda.h"

#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct ReplayBenchConfig {
    fs::path capture_dir;
    int64_t warmup = 5;
    int64_t iters = 20;
    int64_t limit = 0;
    bool sorted = false;
};

void print_usage(const char *argv0) {
    std::cerr << "Usage: " << argv0 << " <capture_dir> [--warmup N] [--iters N] [--limit N] [--sorted 0|1]\n";
}

int64_t parse_int_arg(const char *name, const char *value) {
    try {
        return std::stoll(std::string(value));
    } catch (...) {
        throw std::runtime_error(std::string("Invalid value for ") + name + ": " + value);
    }
}

ReplayBenchConfig parse_args(int argc, char *argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        throw std::runtime_error("missing capture_dir");
    }

    ReplayBenchConfig config;
    config.capture_dir = fs::path(argv[1]);

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        auto require_value = [&](const char *name) -> const char * {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing value for ") + name);
            }
            return argv[++i];
        };

        if (arg == "--warmup") {
            config.warmup = std::max<int64_t>(parse_int_arg("--warmup", require_value("--warmup")), 0);
        } else if (arg == "--iters") {
            config.iters = std::max<int64_t>(parse_int_arg("--iters", require_value("--iters")), 1);
        } else if (arg == "--limit") {
            config.limit = std::max<int64_t>(parse_int_arg("--limit", require_value("--limit")), 0);
        } else if (arg == "--sorted") {
            config.sorted = parse_int_arg("--sorted", require_value("--sorted")) != 0;
        } else {
            print_usage(argv[0]);
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    return config;
}

double percentile(std::vector<double> values, double p) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    double pos = (values.size() - 1) * p;
    auto lo = static_cast<size_t>(pos);
    auto hi = std::min(lo + 1, values.size() - 1);
    double frac = pos - static_cast<double>(lo);
    return values[lo] + (values[hi] - values[lo]) * frac;
}

std::vector<fs::path> collect_capture_files(const ReplayBenchConfig &config) {
    TORCH_CHECK(fs::exists(config.capture_dir), "Capture directory does not exist: ", config.capture_dir.string());
    TORCH_CHECK(fs::is_directory(config.capture_dir), "Capture path is not a directory: ", config.capture_dir.string());

    std::vector<fs::path> files;
    for (const auto &entry : fs::directory_iterator(config.capture_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        if (entry.path().extension() == ".pt") {
            files.emplace_back(entry.path());
        }
    }

    std::sort(files.begin(), files.end());
    if (config.limit > 0 && static_cast<int64_t>(files.size()) > config.limit) {
        files.resize(config.limit);
    }
    TORCH_CHECK(!files.empty(), "No .pt capture files found in ", config.capture_dir.string());
    return files;
}

torch::Tensor load_capture_tensor(const fs::path &path, const torch::Device &device) {
    torch::Tensor tensor;
    torch::load(tensor, path.string());
    TORCH_CHECK(tensor.dim() == 1, "Captured tensor must be 1D: ", path.string());
    TORCH_CHECK(tensor.scalar_type() == torch::kInt64, "Captured tensor must be int64: ", path.string());
    return tensor.to(device).contiguous();
}

}  // namespace

int main(int argc, char *argv[]) {
    ReplayBenchConfig config = parse_args(argc, argv);

    TORCH_CHECK(torch::cuda::is_available(), "gege_unique_replay_bench requires CUDA");
    torch::NoGradGuard no_grad;
    torch::Device device(torch::kCUDA, 0);
    c10::cuda::CUDAGuard guard(device);

    auto files = collect_capture_files(config);

    std::vector<torch::Tensor> batches;
    std::vector<int64_t> input_sizes;
    batches.reserve(files.size());
    input_sizes.reserve(files.size());
    for (const auto &path : files) {
        auto tensor = load_capture_tensor(path, device);
        input_sizes.emplace_back(tensor.numel());
        batches.emplace_back(std::move(tensor));
    }

    int64_t validation_count_mismatches = 0;
    int64_t validation_reconstruction_failures = 0;
    int64_t validation_unique_set_mismatches = 0;
    std::string first_validation_mismatch;

    for (size_t i = 0; i < batches.size(); ++i) {
        auto reference = map_tensors_unique_inverse_cuda(batches[i], true);
        UniqueMapCudaDebugInfo validation_debug_info;
        auto candidate = map_tensors_unique_inverse_cuda(batches[i], config.sorted, &validation_debug_info);

        auto reference_unique = std::get<0>(reference);
        auto candidate_unique = std::get<0>(candidate);
        auto candidate_inverse = std::get<1>(candidate);
        auto reconstructed = candidate_unique.index_select(0, candidate_inverse);

        if (reference_unique.numel() != candidate_unique.numel()) {
            validation_count_mismatches++;
            if (first_validation_mismatch.empty()) {
                first_validation_mismatch = files[i].string();
            }
        }
        if (!torch::equal(reconstructed, batches[i])) {
            validation_reconstruction_failures++;
            if (first_validation_mismatch.empty()) {
                first_validation_mismatch = files[i].string();
            }
        }

        auto candidate_sorted = std::get<0>(torch::sort(candidate_unique));
        if (!torch::equal(reference_unique, candidate_sorted)) {
            validation_unique_set_mismatches++;
            if (first_validation_mismatch.empty()) {
                first_validation_mismatch = files[i].string();
            }
        }
    }

    auto run_pass = [&](bool collect_metrics, std::vector<double> *latencies_ms, int64_t *unique_sum, int64_t *fallback_calls,
                        UniqueMapCudaDebugInfo *last_debug_info, std::string *last_fallback_reason) {
        for (const auto &batch : batches) {
            auto start = std::chrono::high_resolution_clock::now();
            UniqueMapCudaDebugInfo debug_info;
            auto result = map_tensors_unique_inverse_cuda(batch, config.sorted, &debug_info);
            auto end = std::chrono::high_resolution_clock::now();

            if (collect_metrics) {
                latencies_ms->emplace_back(std::chrono::duration<double, std::milli>(end - start).count());
                *unique_sum += std::get<0>(result).numel();
                if (debug_info.used_fallback) {
                    (*fallback_calls)++;
                    if (!debug_info.fallback_reason.empty()) {
                        *last_fallback_reason = debug_info.fallback_reason;
                    }
                }
                *last_debug_info = debug_info;
            }
        }
    };

    for (int64_t i = 0; i < config.warmup; ++i) {
        run_pass(false, nullptr, nullptr, nullptr, nullptr, nullptr);
    }

    AT_CUDA_CHECK(cudaDeviceSynchronize());
    auto bench_start = std::chrono::high_resolution_clock::now();

    std::vector<double> latencies_ms;
    latencies_ms.reserve(static_cast<size_t>(config.iters * static_cast<int64_t>(batches.size())));
    int64_t unique_sum = 0;
    int64_t fallback_calls = 0;
    UniqueMapCudaDebugInfo last_debug_info;
    std::string last_fallback_reason;

    for (int64_t iter = 0; iter < config.iters; ++iter) {
        run_pass(true, &latencies_ms, &unique_sum, &fallback_calls, &last_debug_info, &last_fallback_reason);
    }

    AT_CUDA_CHECK(cudaDeviceSynchronize());
    auto bench_end = std::chrono::high_resolution_clock::now();

    int64_t total_calls = static_cast<int64_t>(latencies_ms.size());
    int64_t total_inputs = std::accumulate(input_sizes.begin(), input_sizes.end(), int64_t{0}) * config.iters;
    double total_ms = std::chrono::duration<double, std::milli>(bench_end - bench_start).count();
    double mean_ms = total_calls > 0 ? std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0) / static_cast<double>(total_calls) : 0.0;
    double mean_unique = total_calls > 0 ? static_cast<double>(unique_sum) / static_cast<double>(total_calls) : 0.0;

    std::cout << "capture_dir=" << config.capture_dir.string() << "\n";
    std::cout << "requested_backend=" << last_debug_info.requested_backend << "\n";
    std::cout << "executed_backend=" << last_debug_info.executed_backend << "\n";
    std::cout << "sorted=" << config.sorted << "\n";
    std::cout << "files=" << files.size() << "\n";
    std::cout << "warmup=" << config.warmup << "\n";
    std::cout << "iters=" << config.iters << "\n";
    std::cout << "total_calls=" << total_calls << "\n";
    std::cout << "total_input_ids=" << total_inputs << "\n";
    std::cout << "mean_unique_ids=" << mean_unique << "\n";
    std::cout << "mean_ms=" << mean_ms << "\n";
    std::cout << "p50_ms=" << percentile(latencies_ms, 0.50) << "\n";
    std::cout << "p90_ms=" << percentile(latencies_ms, 0.90) << "\n";
    std::cout << "max_ms=" << (latencies_ms.empty() ? 0.0 : *std::max_element(latencies_ms.begin(), latencies_ms.end())) << "\n";
    std::cout << "total_ms=" << total_ms << "\n";
    std::cout << "calls_with_fallback=" << fallback_calls << "\n";
    std::cout << "reported_total_fallbacks=" << last_debug_info.total_fallbacks << "\n";
    std::cout << "last_fallback_reason=" << last_fallback_reason << "\n";
    std::cout << "validation_count_mismatches=" << validation_count_mismatches << "\n";
    std::cout << "validation_reconstruction_failures=" << validation_reconstruction_failures << "\n";
    std::cout << "validation_unique_set_mismatches=" << validation_unique_set_mismatches << "\n";
    std::cout << "first_validation_mismatch=" << first_validation_mismatch << "\n";

    return 0;
}
