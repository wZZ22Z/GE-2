#include "data/batch.h"

#include "configuration/constants.h"
#include "reporting/logger.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <string>
#ifdef GEGE_CUDA
#include "pytorch_scatter/segment_sum.h"
#endif

using std::get;

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

bool verify_unique_batch_indices_enabled() {
    static bool enabled = parse_env_flag("GEGE_VERIFY_UNIQUE_BATCH_INDICES", false);
    return enabled;
}

bool csr_grad_reduce_enabled() {
    static bool enabled = parse_env_flag("GEGE_CSR_GRAD_REDUCE", false);
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

int64_t stage_debug_max_batches() {
    static int64_t max_batches = parse_env_int("GEGE_STAGE_DEBUG_MAX_BATCHES", 20);
    return std::max<int64_t>(max_batches, 0);
}

std::atomic<int64_t> &stage_debug_counter() {
    static std::atomic<int64_t> counter{0};
    return counter;
}

bool should_run_stage_debug(int64_t &debug_batch_id) {
    if (!stage_debug_enabled()) {
        return false;
    }
    debug_batch_id = stage_debug_counter().fetch_add(1);
    return debug_batch_id < stage_debug_max_batches();
}

double elapsed_ms(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

#ifdef GEGE_CUDA
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> reduce_local_gradients_with_csr(torch::Tensor node_ids, torch::Tensor gradients,
                                                                                         torch::Tensor optimizer_state) {
    if (!node_ids.defined() || !gradients.defined() || node_ids.numel() == 0 || gradients.numel() == 0 || node_ids.numel() != gradients.size(0)) {
        return std::forward_as_tuple(node_ids, gradients, optimizer_state);
    }

    torch::Tensor node_ids64 = node_ids.to(torch::kInt64);
    torch::Tensor permutation = torch::argsort(node_ids64);
    torch::Tensor sorted_ids = node_ids64.index_select(0, permutation);
    torch::Tensor sorted_gradients = gradients.index_select(0, permutation);

    auto unique_tup = torch::unique_consecutive(sorted_ids, false, true);
    torch::Tensor unique_ids = std::get<0>(unique_tup);
    torch::Tensor counts = std::get<2>(unique_tup).to(torch::kInt64);

    if (unique_ids.numel() == sorted_ids.numel()) {
        return std::forward_as_tuple(node_ids64, gradients, optimizer_state);
    }

    auto indptr_opts = torch::TensorOptions().dtype(torch::kInt64).device(node_ids.device());
    torch::Tensor indptr = torch::zeros({unique_ids.numel() + 1}, indptr_opts);
    if (counts.numel() > 0) {
        indptr.narrow(0, 1, counts.numel()).copy_(counts.cumsum(0));
    }

    torch::Tensor reduced_gradients = segment_sum_csr(sorted_gradients, indptr, torch::nullopt);

    torch::Tensor reduced_state = optimizer_state;
    if (optimizer_state.defined() && optimizer_state.size(0) == node_ids.size(0)) {
        torch::Tensor sorted_state = optimizer_state.index_select(0, permutation);
        torch::Tensor segment_starts = indptr.narrow(0, 0, unique_ids.numel());
        reduced_state = sorted_state.index_select(0, segment_starts);
    }

    return std::forward_as_tuple(unique_ids, reduced_gradients, reduced_state);
}
#endif

void verify_unique_batch_indices(int batch_id, torch::Tensor unique_node_indices) {
    if (!verify_unique_batch_indices_enabled() || !unique_node_indices.defined() || unique_node_indices.numel() == 0) {
        return;
    }

    torch::Tensor sorted_ids = std::get<0>(torch::sort(unique_node_indices));
    auto unique_tup = torch::unique_consecutive(sorted_ids, false, true);
    int64_t unique_count = std::get<0>(unique_tup).numel();
    int64_t total_count = unique_node_indices.numel();
    if (unique_count != total_count) {
        int64_t duplicate_count = total_count - unique_count;
        SPDLOG_ERROR("Batch {} has duplicate node ids in unique_node_indices_: total={}, unique={}, duplicates={}",
                     batch_id, total_count, unique_count, duplicate_count);
        throw GegeRuntimeException("Duplicate ids found in unique_node_indices_");
    }
}

}  // namespace

Batch::Batch(bool train) : device_transfer_(0), host_transfer_(0), timer_(false) {
    train_ = train;
    device_id_ = -1;
    clear();
}

Batch::~Batch() { clear(); }

void Batch::to(torch::Device device) {
    device_id_ = device.index();

    if (device.is_cuda()) {
        device_transfer_ = CudaEvent(device_id_);
        host_transfer_ = CudaEvent(device_id_);
    }

    if (edges_.defined()) {
        edges_ = edges_.to(device);
    }

    if (neg_edges_.defined()) {
        neg_edges_ = neg_edges_.to(device);
    }

    if (root_node_indices_.defined()) {
        root_node_indices_ = root_node_indices_.to(device);
    }

    if (unique_node_indices_.defined()) {
        unique_node_indices_ = unique_node_indices_.to(device);
    }

    if (node_labels_.defined()) {
        node_labels_ = node_labels_.to(device);
    }

    if (src_neg_indices_mapping_.defined()) {
        src_neg_indices_mapping_ = src_neg_indices_mapping_.to(device);
    }

    if (dst_neg_indices_mapping_.defined()) {
        dst_neg_indices_mapping_ = dst_neg_indices_mapping_.to(device);
    }

    if (src_neg_filter_.defined()) {
        src_neg_filter_ = src_neg_filter_.to(device);
    }

    if (dst_neg_filter_.defined()) {
        dst_neg_filter_ = dst_neg_filter_.to(device);
    }

    if (node_embeddings_.defined()) {
        node_embeddings_ = node_embeddings_.to(device);
    }

    if (node_embeddings_state_.defined()) {
        node_embeddings_state_ = node_embeddings_state_.to(device);
    }

    if (node_embeddings_g_.defined()) {
        node_embeddings_g_ = node_embeddings_g_.to(device);
    }

    if (node_embeddings_state_g_.defined()) {
        node_embeddings_state_g_ = node_embeddings_state_g_.to(device);
    }

    if (node_features_.defined()) {
        node_features_ = node_features_.to(device);
    }

    if (encoded_uniques_.defined()) {
        encoded_uniques_ = encoded_uniques_.to(device);
    }

    if (dense_graph_.node_ids_.defined()) {
        dense_graph_.to(device);
    }

    if (device.is_cuda()) {
        device_transfer_.record();
    }
}

void Batch::accumulateGradients(float learning_rate) {
    int64_t debug_batch_id = -1;
    bool run_stage_debug = should_run_stage_debug(debug_batch_id);
    auto accumulate_start = std::chrono::high_resolution_clock::now();
    auto step_start = accumulate_start;

    verify_unique_batch_indices(batch_id_, unique_node_indices_);

    if (node_embeddings_.defined()) {
        node_gradients_ = node_embeddings_.grad();
        SPDLOG_TRACE("Batch: {} accumulated node gradients", batch_id_);
        if (run_stage_debug) {
            auto now = std::chrono::high_resolution_clock::now();
            int64_t grad_rows = node_gradients_.defined() && node_gradients_.dim() > 0 ? node_gradients_.size(0) : 0;
            int64_t unique_rows = unique_node_indices_.defined() ? unique_node_indices_.numel() : 0;
            int64_t duplicates = std::max<int64_t>(unique_rows - grad_rows, 0);
            SPDLOG_INFO("[stage-debug][accumulateGradients][batch {}][step 1] grad_capture ms={:.3f} grad_rows={} unique_rows={} possible_duplicates={}",
                        debug_batch_id, elapsed_ms(step_start, now), grad_rows, unique_rows, duplicates);
            step_start = now;
        }

#ifdef GEGE_CUDA
        if (csr_grad_reduce_enabled() && node_gradients_.device().is_cuda()) {
            std::tie(unique_node_indices_, node_gradients_, node_embeddings_state_) =
                reduce_local_gradients_with_csr(unique_node_indices_, node_gradients_, node_embeddings_state_);
            if (run_stage_debug) {
                auto now = std::chrono::high_resolution_clock::now();
                int64_t grad_rows = node_gradients_.defined() && node_gradients_.dim() > 0 ? node_gradients_.size(0) : 0;
                SPDLOG_INFO("[stage-debug][accumulateGradients][batch {}][step 2] csr_reduce ms={:.3f} reduced_rows={}",
                            debug_batch_id, elapsed_ms(step_start, now), grad_rows);
                step_start = now;
            }
        }
#endif

        node_state_update_ = node_gradients_.pow(2);
        node_embeddings_state_.add_(node_state_update_);
        node_gradients_ = -learning_rate * (node_gradients_ / (node_embeddings_state_.sqrt().add_(1e-10)));
        if (run_stage_debug) {
            auto now = std::chrono::high_resolution_clock::now();
            SPDLOG_INFO("[stage-debug][accumulateGradients][batch {}][step 3] optimizer_update ms={:.3f}",
                        debug_batch_id, elapsed_ms(step_start, now));
            step_start = now;
        }

        SPDLOG_TRACE("Batch: {} adjusted gradients", batch_id_);
    }

    node_embeddings_state_ = torch::Tensor();

    if (run_stage_debug) {
        auto now = std::chrono::high_resolution_clock::now();
        SPDLOG_INFO("[stage-debug][accumulateGradients][batch {}][step 4] finalize ms={:.3f} total_ms={:.3f}",
                    debug_batch_id, elapsed_ms(step_start, now), elapsed_ms(accumulate_start, now));
    }

    SPDLOG_TRACE("Batch: {} cleared gpu embeddings and gradients", batch_id_);
}

void Batch::accumulateGradientsG(float learning_rate) {
    if (node_embeddings_g_.defined()) {
        node_gradients_g_ = node_embeddings_g_.grad();
        SPDLOG_TRACE("Batch: {} accumulated node gradients g", batch_id_);

        node_state_update_g_ = node_gradients_g_.pow(2);
        node_embeddings_state_g_.add_(node_state_update_g_);
        node_gradients_g_ = -learning_rate * (node_gradients_g_ / (node_embeddings_state_g_.sqrt().add_(1e-10)));

        SPDLOG_TRACE("Batch: {} adjusted gradients g", batch_id_);
    }

    node_embeddings_state_g_ = torch::Tensor();

    SPDLOG_TRACE("Batch: {} cleared gpu embeddings and gradients g", batch_id_);
}

void Batch::embeddingsToHost() {
    if (node_gradients_.defined() && node_gradients_.device().is_cuda()) {
        auto grad_opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(true);
        Gradients temp_grads = torch::empty(node_gradients_.sizes(), grad_opts);
        temp_grads.copy_(node_gradients_, true);
        Gradients temp_grads2 = torch::empty(node_state_update_.sizes(), grad_opts);
        temp_grads2.copy_(node_state_update_, true);
        node_gradients_ = temp_grads;
        node_state_update_ = temp_grads2;
    }

    if (unique_node_indices_.defined()) {
        unique_node_indices_ = unique_node_indices_.to(torch::kCPU);
    }

    if (encoded_uniques_.defined()) {
        encoded_uniques_ = encoded_uniques_.to(torch::kCPU);
    }

    host_transfer_.record();
    host_transfer_.synchronize();
}

void Batch::embeddingsToHostG() {
    if (node_gradients_g_.defined() && node_gradients_g_.device().is_cuda()) {
        auto grad_opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(true);
        Gradients temp_grads = torch::empty(node_gradients_g_.sizes(), grad_opts);
        temp_grads.copy_(node_gradients_g_, true);
        Gradients temp_grads2 = torch::empty(node_state_update_g_.sizes(), grad_opts);
        temp_grads2.copy_(node_state_update_g_, true);
        node_gradients_g_ = temp_grads;
        node_state_update_g_ = temp_grads2;
    }

    host_transfer_.record();
    host_transfer_.synchronize();
}

void Batch::clear() {
    root_node_indices_ = torch::Tensor();
    unique_node_indices_ = torch::Tensor();
    node_embeddings_ = torch::Tensor();
    node_embeddings_g_ = torch::Tensor();
    node_gradients_ = torch::Tensor();
    node_gradients_g_ = torch::Tensor();
    node_state_update_ = torch::Tensor();
    node_state_update_g_ = torch::Tensor();
    node_embeddings_state_ = torch::Tensor();
    node_embeddings_state_g_ = torch::Tensor();

    node_features_ = torch::Tensor();
    node_labels_ = torch::Tensor();

    src_neg_indices_mapping_ = torch::Tensor();
    dst_neg_indices_mapping_ = torch::Tensor();

    edges_ = torch::Tensor();
    neg_edges_ = torch::Tensor();
    src_neg_indices_ = torch::Tensor();
    dst_neg_indices_ = torch::Tensor();

    dense_graph_.clear();
    encoded_uniques_ = torch::Tensor();

    src_neg_filter_ = torch::Tensor();
    dst_neg_filter_ = torch::Tensor();
}
