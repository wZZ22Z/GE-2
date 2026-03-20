#include "data/dataloader.h"

#include "common/util.h"
#include "data/ordering.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <sstream>
#include <string>
#ifdef GEGE_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
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

bool fast_map_tensors_enabled() {
    static bool enabled = parse_env_flag("GEGE_FAST_MAP_TENSORS", true);
    return enabled;
}

bool verify_node_mapping_enabled() {
    static bool enabled = parse_env_flag("GEGE_VERIFY_NODE_MAPPING", false);
    return enabled;
}

bool partition_buffer_lp_fast_path_env_enabled(bool default_value) {
    return parse_env_flag("GEGE_PARTITION_BUFFER_LP_FAST_PATH", default_value);
}

bool partition_buffer_pipeline_timing_enabled() {
    static bool enabled = parse_env_flag("GEGE_PARTITION_BUFFER_PIPELINE_TIMING", false);
    return enabled;
}

int64_t partition_buffer_pipeline_timing_max() {
    static int64_t max_timings = std::max<int64_t>(parse_env_int("GEGE_PARTITION_BUFFER_PIPELINE_TIMING_MAX", 8), 0);
    return max_timings;
}

std::atomic<int64_t> &partition_buffer_pipeline_timing_counter() {
    static std::atomic<int64_t> counter{0};
    return counter;
}

bool should_log_partition_buffer_pipeline_timing(int64_t &timing_id) {
    if (!partition_buffer_pipeline_timing_enabled()) {
        return false;
    }

    timing_id = partition_buffer_pipeline_timing_counter().fetch_add(1);
    return timing_id < partition_buffer_pipeline_timing_max();
}

bool negative_sampler_filtered(const shared_ptr<NegativeSampler> &negative_sampler) {
    if (negative_sampler == nullptr) {
        return false;
    }

    if (instance_of<NegativeSampler, NegativeSamplingBase>(negative_sampler)) {
        return std::dynamic_pointer_cast<NegativeSamplingBase>(negative_sampler)->filtered_;
    }

    if (instance_of<NegativeSampler, CorruptNodeNegativeSampler>(negative_sampler)) {
        return std::dynamic_pointer_cast<CorruptNodeNegativeSampler>(negative_sampler)->filtered_;
    }

    return true;
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

int64_t elapsed_ns(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point end) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

void initialize_perf_vector(std::vector<int64_t> &values, std::size_t size) {
    values.assign(size, 0);
}

void initialize_perf_samples(std::vector<std::vector<int64_t>> &values, std::size_t size) {
    values.assign(size, {});
}

void add_perf_stat(std::atomic<int64_t> &aggregate, std::vector<int64_t> &per_device, int32_t device_idx, int64_t elapsed_ns) {
    aggregate.fetch_add(elapsed_ns);
    if (device_idx >= 0 && static_cast<std::size_t>(device_idx) < per_device.size()) {
        per_device[device_idx] += elapsed_ns;
    }
}

void add_perf_sample(std::vector<std::vector<int64_t>> &per_device, int32_t device_idx, int64_t value) {
    if (device_idx >= 0 && static_cast<std::size_t>(device_idx) < per_device.size()) {
        per_device[device_idx].push_back(value);
    }
}

std::string format_tensor_values(torch::Tensor values) {
    if (!values.defined()) {
        return "-";
    }

    values = values.reshape({-1}).to(torch::kCPU).to(torch::kInt64).contiguous();
    if (values.numel() == 0) {
        return "-";
    }

    auto accessor = values.accessor<int64_t, 1>();
    std::ostringstream oss;
    for (int64_t i = 0; i < values.size(0); i++) {
        if (i > 0) {
            oss << ",";
        }
        oss << accessor[i];
    }
    return oss.str();
}

}  // namespace

DataLoader::DataLoader(shared_ptr<GraphModelStorage> graph_storage, LearningTask learning_task, shared_ptr<TrainingConfig> training_config,
                       shared_ptr<EvaluationConfig> evaluation_config, shared_ptr<EncoderConfig> encoder_config, vector<torch::Device> devices,
                       NegativeSamplingMethod nsm, bool use_inverse_relations) {
    current_edge_ = 0;
    train_ = true;
    epochs_processed_ = 0;
    batches_processed_ = 0;
    loaded_subgraphs = 0;
    false_negative_edges = 0;
    swap_tasks_completed = 0;
    sampler_lock_ = new std::mutex();
    batch_lock_ = new std::mutex;
    batch_cv_ = new std::condition_variable;
    waiting_for_batches_ = false;

    single_dataset_ = false;
    async_barrier = 0;
    graph_storage_ = graph_storage;
    learning_task_ = learning_task;
    training_config_ = training_config;
    evaluation_config_ = evaluation_config;
    only_root_features_ = false;
    use_inverse_relations_ = use_inverse_relations;

    edge_sampler_ = std::make_shared<RandomEdgeSampler>(graph_storage_);

    devices_ = devices;
    activate_devices_ = 0;
    initialize_perf_vector(device_swap_barrier_wait_ns_, devices_.size());
    initialize_perf_vector(device_swap_update_ns_, devices_.size());
    initialize_perf_vector(device_swap_rebuild_ns_, devices_.size());
    initialize_perf_vector(device_swap_sync_wait_ns_, devices_.size());
    initialize_perf_vector(device_swap_count_, devices_.size());
    initialize_perf_vector(device_get_next_batch_ns_, devices_.size());
    initialize_perf_vector(device_edge_sample_ns_, devices_.size());
    initialize_perf_vector(device_edge_get_edges_ns_, devices_.size());
    initialize_perf_vector(device_edge_negative_sample_ns_, devices_.size());
    initialize_perf_vector(device_edge_map_collect_ids_ns_, devices_.size());
    initialize_perf_vector(device_edge_map_lookup_ns_, devices_.size());
    initialize_perf_vector(device_edge_map_verify_ns_, devices_.size());
    initialize_perf_vector(device_edge_remap_assign_ns_, devices_.size());
    initialize_perf_vector(device_edge_finalize_ns_, devices_.size());
    initialize_perf_vector(device_node_sample_ns_, devices_.size());
    initialize_perf_vector(device_load_cpu_parameters_ns_, devices_.size());
    initialize_perf_vector(device_get_batch_device_prepare_ns_, devices_.size());
    initialize_perf_vector(device_get_batch_perform_map_ns_, devices_.size());
    initialize_perf_vector(device_get_batch_overhead_ns_, devices_.size());
    initialize_perf_samples(device_swap_active_bucket_samples_, devices_.size());
    initialize_perf_samples(device_swap_active_edge_samples_, devices_.size());
    initialize_perf_samples(device_swap_batch_count_samples_, devices_.size());
    initialize_perf_samples(device_swap_rebuild_samples_ns_, devices_.size());
    initialize_perf_vector(device_current_state_index_, devices_.size());
    std::fill(device_current_state_index_.begin(), device_current_state_index_.end(), -1);
    initialize_perf_vector(device_current_active_bucket_count_, devices_.size());
    initialize_perf_vector(device_current_active_edge_count_, devices_.size());
    device_current_state_partitions_.assign(devices_.size(), std::string("-"));
    initialize_perf_vector(device_state_build_sequence_, devices_.size());

    negative_sampling_method_ = nsm;

    if (learning_task_ == LearningTask::LINK_PREDICTION) {
        if (negative_sampling_method_ == NegativeSamplingMethod::RNS) {
            training_negative_sampler_ = std::make_shared<RNS>(
                training_config_->negative_sampling->num_chunks, training_config_->negative_sampling->negatives_per_positive,
                training_config_->negative_sampling->degree_fraction, training_config_->negative_sampling->filtered,
                training_config_->negative_sampling->superbatch_negative_plan_batches, training_config_->negative_sampling->local_filter_mode,
                training_config_->negative_sampling->tournament_selection,
                training_config_->negative_sampling->tiled_tournament_scores,
                training_config_->negative_sampling->tiled_tournament_groups_per_tile);
        } else if (negative_sampling_method_ == NegativeSamplingMethod::DNS) {
            training_negative_sampler_ = std::make_shared<DNS>(
                training_config_->negative_sampling->num_chunks, training_config_->negative_sampling->negatives_per_positive,
                training_config_->negative_sampling->degree_fraction, training_config_->negative_sampling->filtered,
                training_config_->negative_sampling->superbatch_negative_plan_batches, training_config_->negative_sampling->local_filter_mode,
                training_config_->negative_sampling->tournament_selection,
                training_config_->negative_sampling->tiled_tournament_scores,
                training_config_->negative_sampling->tiled_tournament_groups_per_tile);
        } else if (negative_sampling_method_ == NegativeSamplingMethod::GAN) {
            training_negative_sampler_ = std::make_shared<KBGAN>(
                training_config_->negative_sampling->num_chunks, training_config_->negative_sampling->negatives_per_positive,
                training_config_->negative_sampling->degree_fraction, training_config_->negative_sampling->filtered,
                training_config_->negative_sampling->superbatch_negative_plan_batches, training_config_->negative_sampling->local_filter_mode,
                training_config_->negative_sampling->tournament_selection,
                training_config_->negative_sampling->tiled_tournament_scores,
                training_config_->negative_sampling->tiled_tournament_groups_per_tile);
        } else {
            SPDLOG_INFO("NegativeSampling: not supported");
        }
        evaluation_negative_sampler_ = std::make_shared<CorruptNodeNegativeSampler>(
            evaluation_config_->negative_sampling->num_chunks, evaluation_config_->negative_sampling->negatives_per_positive,
            evaluation_config_->negative_sampling->degree_fraction, evaluation_config_->negative_sampling->filtered,
            evaluation_config_->negative_sampling->local_filter_mode);
    } else {
        training_negative_sampler_ = nullptr;
        evaluation_negative_sampler_ = nullptr;
    }

    if (encoder_config != nullptr) {
        if (!encoder_config->train_neighbor_sampling.empty()) {
            training_neighbor_sampler_ = std::make_shared<LayeredNeighborSampler>(graph_storage_, encoder_config->train_neighbor_sampling);

            if (!encoder_config->eval_neighbor_sampling.empty()) {
                evaluation_neighbor_sampler_ = std::make_shared<LayeredNeighborSampler>(graph_storage_, encoder_config->eval_neighbor_sampling);
            } else {
                evaluation_neighbor_sampler_ = training_neighbor_sampler_;
            }

        } else {
            training_neighbor_sampler_ = nullptr;
            evaluation_neighbor_sampler_ = nullptr;
        }
    } else {
        training_neighbor_sampler_ = nullptr;
        evaluation_neighbor_sampler_ = nullptr;
    }

    negative_sampler_ = training_negative_sampler_;
    neighbor_sampler_ = training_neighbor_sampler_;
    auto initialize_negative_sampler_perf = [&](const shared_ptr<NegativeSampler> &sampler) {
        if (instance_of<NegativeSampler, NegativeSamplingBase>(sampler)) {
            std::dynamic_pointer_cast<NegativeSamplingBase>(sampler)->initializePerfStats(devices_.size());
        }
    };
    initialize_negative_sampler_perf(training_negative_sampler_);
    initialize_negative_sampler_perf(evaluation_negative_sampler_);
    refreshGraphStorageMode();
}

DataLoader::DataLoader(shared_ptr<GraphModelStorage> graph_storage, LearningTask learning_task, int batch_size, shared_ptr<NegativeSampler> negative_sampler,
                       shared_ptr<NeighborSampler> neighbor_sampler, bool train) {
    current_edge_ = 0;
    train_ = train;
    epochs_processed_ = 0;
    loaded_subgraphs = 0;
    false_negative_edges = 0;
    batches_processed_ = 0;
    sampler_lock_ = new std::mutex();
    batch_lock_ = new std::mutex;
    batch_cv_ = new std::condition_variable;
    async_barrier = 0;
    waiting_for_batches_ = false;

    batch_size_ = batch_size;
    single_dataset_ = true;

    graph_storage_ = graph_storage;
    learning_task_ = learning_task;
    use_inverse_relations_ = true;
    only_root_features_ = false;

    edge_sampler_ = std::make_shared<RandomEdgeSampler>(graph_storage_);
    if (graph_storage_ != nullptr) {
        devices_ = graph_storage_->devices_;
    }
    activate_devices_ = 0;
    initialize_perf_vector(device_swap_barrier_wait_ns_, devices_.size());
    initialize_perf_vector(device_swap_update_ns_, devices_.size());
    initialize_perf_vector(device_swap_rebuild_ns_, devices_.size());
    initialize_perf_vector(device_swap_sync_wait_ns_, devices_.size());
    initialize_perf_vector(device_swap_count_, devices_.size());
    initialize_perf_vector(device_get_next_batch_ns_, devices_.size());
    initialize_perf_vector(device_edge_sample_ns_, devices_.size());
    initialize_perf_vector(device_edge_get_edges_ns_, devices_.size());
    initialize_perf_vector(device_edge_negative_sample_ns_, devices_.size());
    initialize_perf_vector(device_edge_map_collect_ids_ns_, devices_.size());
    initialize_perf_vector(device_edge_map_lookup_ns_, devices_.size());
    initialize_perf_vector(device_edge_map_verify_ns_, devices_.size());
    initialize_perf_vector(device_edge_remap_assign_ns_, devices_.size());
    initialize_perf_vector(device_edge_finalize_ns_, devices_.size());
    initialize_perf_vector(device_node_sample_ns_, devices_.size());
    initialize_perf_vector(device_load_cpu_parameters_ns_, devices_.size());
    initialize_perf_vector(device_get_batch_device_prepare_ns_, devices_.size());
    initialize_perf_vector(device_get_batch_perform_map_ns_, devices_.size());
    initialize_perf_vector(device_get_batch_overhead_ns_, devices_.size());
    initialize_perf_samples(device_swap_active_bucket_samples_, devices_.size());
    initialize_perf_samples(device_swap_active_edge_samples_, devices_.size());
    initialize_perf_samples(device_swap_batch_count_samples_, devices_.size());
    initialize_perf_samples(device_swap_rebuild_samples_ns_, devices_.size());
    initialize_perf_vector(device_current_state_index_, devices_.size());
    std::fill(device_current_state_index_.begin(), device_current_state_index_.end(), -1);
    initialize_perf_vector(device_current_active_bucket_count_, devices_.size());
    initialize_perf_vector(device_current_active_edge_count_, devices_.size());
    device_current_state_partitions_.assign(devices_.size(), std::string("-"));
    initialize_perf_vector(device_state_build_sequence_, devices_.size());
    negative_sampler_ = negative_sampler;
    neighbor_sampler_ = neighbor_sampler;
    if (instance_of<NegativeSampler, NegativeSamplingBase>(negative_sampler_)) {
        std::dynamic_pointer_cast<NegativeSamplingBase>(negative_sampler_)->initializePerfStats(devices_.size());
    }

    training_negative_sampler_ = nullptr;
    evaluation_negative_sampler_ = nullptr;

    training_neighbor_sampler_ = nullptr;
    evaluation_neighbor_sampler_ = nullptr;

    refreshGraphStorageMode();

    loadStorage();
}

DataLoader::~DataLoader() {
    delete sampler_lock_;
    delete batch_lock_;
    delete batch_cv_;
}

void DataLoader::refreshGraphStorageMode() {
    if (graph_storage_ == nullptr) {
        return;
    }

    bool enable_fast_path = false;
    if (graph_storage_ != nullptr && learning_task_ == LearningTask::LINK_PREDICTION && train_ && graph_storage_->useInMemorySubGraph() &&
        neighbor_sampler_ == nullptr && !negative_sampler_filtered(negative_sampler_)) {
        enable_fast_path = partition_buffer_lp_fast_path_env_enabled(true);
    }
    graph_storage_->setPartitionBufferLPFastPathEnabled(enable_fast_path);

    if (enable_fast_path) {
        SPDLOG_INFO("Using partition-buffer LP fast path (arithmetic remap, no in-memory graph build)");
    }
}

void DataLoader::resetPerfStats() {
    swap_barrier_wait_ns_.store(0);
    swap_update_ns_.store(0);
    swap_rebuild_ns_.store(0);
    swap_sync_wait_ns_.store(0);
    swap_count_.store(0);
    get_next_batch_ns_.store(0);
    edge_sample_ns_.store(0);
    edge_get_edges_ns_.store(0);
    edge_negative_sample_ns_.store(0);
    edge_map_collect_ids_ns_.store(0);
    edge_map_lookup_ns_.store(0);
    edge_map_verify_ns_.store(0);
    edge_remap_assign_ns_.store(0);
    edge_finalize_ns_.store(0);
    node_sample_ns_.store(0);
    load_cpu_parameters_ns_.store(0);
    get_batch_device_prepare_ns_.store(0);
    get_batch_perform_map_ns_.store(0);
    get_batch_overhead_ns_.store(0);
    std::fill(device_swap_barrier_wait_ns_.begin(), device_swap_barrier_wait_ns_.end(), 0);
    std::fill(device_swap_update_ns_.begin(), device_swap_update_ns_.end(), 0);
    std::fill(device_swap_rebuild_ns_.begin(), device_swap_rebuild_ns_.end(), 0);
    std::fill(device_swap_sync_wait_ns_.begin(), device_swap_sync_wait_ns_.end(), 0);
    std::fill(device_swap_count_.begin(), device_swap_count_.end(), 0);
    std::fill(device_get_next_batch_ns_.begin(), device_get_next_batch_ns_.end(), 0);
    std::fill(device_edge_sample_ns_.begin(), device_edge_sample_ns_.end(), 0);
    std::fill(device_edge_get_edges_ns_.begin(), device_edge_get_edges_ns_.end(), 0);
    std::fill(device_edge_negative_sample_ns_.begin(), device_edge_negative_sample_ns_.end(), 0);
    std::fill(device_edge_map_collect_ids_ns_.begin(), device_edge_map_collect_ids_ns_.end(), 0);
    std::fill(device_edge_map_lookup_ns_.begin(), device_edge_map_lookup_ns_.end(), 0);
    std::fill(device_edge_map_verify_ns_.begin(), device_edge_map_verify_ns_.end(), 0);
    std::fill(device_edge_remap_assign_ns_.begin(), device_edge_remap_assign_ns_.end(), 0);
    std::fill(device_edge_finalize_ns_.begin(), device_edge_finalize_ns_.end(), 0);
    std::fill(device_node_sample_ns_.begin(), device_node_sample_ns_.end(), 0);
    std::fill(device_load_cpu_parameters_ns_.begin(), device_load_cpu_parameters_ns_.end(), 0);
    std::fill(device_get_batch_device_prepare_ns_.begin(), device_get_batch_device_prepare_ns_.end(), 0);
    std::fill(device_get_batch_perform_map_ns_.begin(), device_get_batch_perform_map_ns_.end(), 0);
    std::fill(device_get_batch_overhead_ns_.begin(), device_get_batch_overhead_ns_.end(), 0);
    for (auto &samples : device_swap_active_bucket_samples_) {
        samples.clear();
    }
    for (auto &samples : device_swap_active_edge_samples_) {
        samples.clear();
    }
    for (auto &samples : device_swap_batch_count_samples_) {
        samples.clear();
    }
    for (auto &samples : device_swap_rebuild_samples_ns_) {
        samples.clear();
    }
    auto reset_negative_sampler_perf = [&](const shared_ptr<NegativeSampler> &sampler) {
        if (instance_of<NegativeSampler, NegativeSamplingBase>(sampler)) {
            std::dynamic_pointer_cast<NegativeSamplingBase>(sampler)->resetPerfStats();
        }
    };
    reset_negative_sampler_perf(training_negative_sampler_);
    reset_negative_sampler_perf(evaluation_negative_sampler_);
    reset_negative_sampler_perf(negative_sampler_);
}

DataLoaderPerfStats DataLoader::getPerfStats() const {
    DataLoaderPerfStats stats;
    stats.swap_barrier_wait_ns = swap_barrier_wait_ns_.load();
    stats.swap_update_ns = swap_update_ns_.load();
    stats.swap_rebuild_ns = swap_rebuild_ns_.load();
    stats.swap_sync_wait_ns = swap_sync_wait_ns_.load();
    stats.swap_count = swap_count_.load();
    stats.get_next_batch_ns = get_next_batch_ns_.load();
    stats.edge_sample_ns = edge_sample_ns_.load();
    stats.edge_get_edges_ns = edge_get_edges_ns_.load();
    stats.edge_negative_sample_ns = edge_negative_sample_ns_.load();
    stats.edge_map_collect_ids_ns = edge_map_collect_ids_ns_.load();
    stats.edge_map_lookup_ns = edge_map_lookup_ns_.load();
    stats.edge_map_verify_ns = edge_map_verify_ns_.load();
    stats.edge_remap_assign_ns = edge_remap_assign_ns_.load();
    stats.edge_finalize_ns = edge_finalize_ns_.load();
    stats.node_sample_ns = node_sample_ns_.load();
    stats.load_cpu_parameters_ns = load_cpu_parameters_ns_.load();
    stats.get_batch_device_prepare_ns = get_batch_device_prepare_ns_.load();
    stats.get_batch_perform_map_ns = get_batch_perform_map_ns_.load();
    stats.get_batch_overhead_ns = get_batch_overhead_ns_.load();
    stats.device_swap_barrier_wait_ns = device_swap_barrier_wait_ns_;
    stats.device_swap_update_ns = device_swap_update_ns_;
    stats.device_swap_rebuild_ns = device_swap_rebuild_ns_;
    stats.device_swap_sync_wait_ns = device_swap_sync_wait_ns_;
    stats.device_swap_count = device_swap_count_;
    stats.device_get_next_batch_ns = device_get_next_batch_ns_;
    stats.device_edge_sample_ns = device_edge_sample_ns_;
    stats.device_edge_get_edges_ns = device_edge_get_edges_ns_;
    stats.device_edge_negative_sample_ns = device_edge_negative_sample_ns_;
    stats.device_edge_map_collect_ids_ns = device_edge_map_collect_ids_ns_;
    stats.device_edge_map_lookup_ns = device_edge_map_lookup_ns_;
    stats.device_edge_map_verify_ns = device_edge_map_verify_ns_;
    stats.device_edge_remap_assign_ns = device_edge_remap_assign_ns_;
    stats.device_edge_finalize_ns = device_edge_finalize_ns_;
    stats.device_node_sample_ns = device_node_sample_ns_;
    stats.device_load_cpu_parameters_ns = device_load_cpu_parameters_ns_;
    stats.device_get_batch_device_prepare_ns = device_get_batch_device_prepare_ns_;
    stats.device_get_batch_perform_map_ns = device_get_batch_perform_map_ns_;
    stats.device_get_batch_overhead_ns = device_get_batch_overhead_ns_;
    stats.device_swap_active_bucket_samples = device_swap_active_bucket_samples_;
    stats.device_swap_active_edge_samples = device_swap_active_edge_samples_;
    stats.device_swap_batch_count_samples = device_swap_batch_count_samples_;
    stats.device_swap_rebuild_samples_ns = device_swap_rebuild_samples_ns_;
    if (instance_of<NegativeSampler, NegativeSamplingBase>(negative_sampler_)) {
        stats.negative_sampler = std::dynamic_pointer_cast<NegativeSamplingBase>(negative_sampler_)->getPerfStats();
    }
    return stats;
}

void DataLoader::nextEpoch() {
    batch_id_offset_ = 0;
    total_batches_processed_ = 0;
    epochs_processed_++;
    std::fill(device_state_build_sequence_.begin(), device_state_build_sequence_.end(), 0);
    if (negative_sampler_ != nullptr) {
        negative_sampler_->resetPlanCache();
    }
    buffer_states_.clear();
    if (graph_storage_->useInMemorySubGraph()) {
        unloadStorage();
    }
}

void DataLoader::setActiveEdges(int32_t device_idx) {
    EdgeList active_edges;
    torch::Tensor edge_bucket_sizes;
    int64_t timing_id = -1;
    bool log_timing = should_log_partition_buffer_pipeline_timing(timing_id);
    auto total_start = std::chrono::high_resolution_clock::now();
    auto phase_start = total_start;
    double bucket_lookup_ms = 0.0;
    double gather_ms = 0.0;
    double shuffle_ms = 0.0;
    int64_t num_active_buckets = 0;
    int64_t state_idx = -1;
    std::string resident_partitions = "-";

    if (graph_storage_->useInMemorySubGraph()) {
        state_idx = std::distance(edge_buckets_per_buffer_.begin(), edge_buckets_per_buffer_iterators_[device_idx]);
        if (state_idx >= 0 && static_cast<std::size_t>(state_idx) < buffer_states_.size()) {
            resident_partitions = format_tensor_values(buffer_states_[state_idx]);
        }
        torch::Tensor edge_bucket_ids = *edge_buckets_per_buffer_iterators_[device_idx];
        for (int i = 0; i < devices_.size(); i++) {
            if (edge_buckets_per_buffer_iterators_[device_idx] != edge_buckets_per_buffer_.end()) {
                edge_buckets_per_buffer_iterators_[device_idx]++;
            }
        }

        int num_partitions = graph_storage_->getNumPartitions();
        num_active_buckets = edge_bucket_ids.size(0);
        edge_bucket_ids = edge_bucket_ids.select(1, 0) * num_partitions + edge_bucket_ids.select(1, 1);
        torch::Tensor in_memory_edge_bucket_idx = torch::empty({edge_bucket_ids.size(0)}, edge_bucket_ids.options());
        edge_bucket_sizes = torch::empty({edge_bucket_ids.size(0)}, edge_bucket_ids.options());

        auto edge_bucket_ids_accessor = edge_bucket_ids.accessor<int64_t, 1>();
        auto in_memory_edge_bucket_idx_accessor = in_memory_edge_bucket_idx.accessor<int64_t, 1>();
        auto edge_bucket_sizes_accessor = edge_bucket_sizes.accessor<int64_t, 1>();
        auto all_edge_bucket_sizes_accessor = graph_storage_->current_subgraph_states_[device_idx]->in_memory_edge_bucket_sizes_.accessor<int64_t, 1>();
        auto all_edge_bucket_starts_accessor = graph_storage_->current_subgraph_states_[device_idx]->in_memory_edge_bucket_starts_.accessor<int64_t, 1>();
        
        auto tup = torch::sort(graph_storage_->current_subgraph_states_[device_idx]->in_memory_edge_bucket_ids_);
        torch::Tensor sorted_in_memory_ids = std::get<0>(tup);
        torch::Tensor in_memory_id_indices = std::get<1>(tup);
        auto in_memory_id_indices_accessor = in_memory_id_indices.accessor<int64_t, 1>();

#pragma omp parallel for
        for (int i = 0; i < in_memory_edge_bucket_idx.size(0); i++) {
            int64_t edge_bucket_id = edge_bucket_ids_accessor[i];
            int64_t idx = torch::searchsorted(sorted_in_memory_ids, edge_bucket_id).item<int64_t>();
            idx = in_memory_id_indices_accessor[idx];
            int64_t edge_bucket_size = all_edge_bucket_sizes_accessor[idx];

            in_memory_edge_bucket_idx_accessor[i] = idx;
            edge_bucket_sizes_accessor[i] = edge_bucket_size;
        }

        if (log_timing) {
            auto now = std::chrono::high_resolution_clock::now();
            bucket_lookup_ms = elapsed_ms(phase_start, now);
            phase_start = now;
        }

        torch::Tensor local_offsets = edge_bucket_sizes.cumsum(0);
        int64_t total_size = local_offsets[-1].item<int64_t>();
        local_offsets = local_offsets - edge_bucket_sizes;

        auto local_offsets_accessor = local_offsets.accessor<int64_t, 1>();

        active_edges = torch::empty({total_size, graph_storage_->storage_ptrs_.edges->dim1_size_},
                                    graph_storage_->current_subgraph_states_[device_idx]->all_in_memory_mapped_edges_.options());

#pragma omp parallel for
        for (int i = 0; i < in_memory_edge_bucket_idx.size(0); i++) {
            int64_t idx = in_memory_edge_bucket_idx_accessor[i];
            int64_t edge_bucket_size = edge_bucket_sizes_accessor[i];
            int64_t edge_bucket_start = all_edge_bucket_starts_accessor[idx];
            int64_t local_offset = local_offsets_accessor[i];

            active_edges.narrow(0, local_offset, edge_bucket_size) =
                graph_storage_->current_subgraph_states_[device_idx]->all_in_memory_mapped_edges_.narrow(0, edge_bucket_start, edge_bucket_size);
        }

        if (log_timing) {
            auto now = std::chrono::high_resolution_clock::now();
            gather_ms = elapsed_ms(phase_start, now);
            phase_start = now;
        }

    } else {
        active_edges = graph_storage_->storage_ptrs_.edges->range(0, graph_storage_->storage_ptrs_.edges->getDim0());
    }
    auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    auto perm = torch::randperm(active_edges.size(0), opts);
    perm = perm.to(active_edges.device());
    active_edges = active_edges.index_select(0, perm);
    if (log_timing) {
        auto now = std::chrono::high_resolution_clock::now();
        shuffle_ms = elapsed_ms(phase_start, now);
        SPDLOG_INFO(
            "[partition-buffer-pipeline][setActiveEdges {}] device={} buckets={} edges={} bucket_lookup_ms={:.3f} gather_ms={:.3f} shuffle_ms={:.3f} total_ms={:.3f}",
            timing_id, device_idx, num_active_buckets, active_edges.size(0), bucket_lookup_ms, gather_ms, shuffle_ms, elapsed_ms(total_start, now));
    }
    add_perf_sample(device_swap_active_bucket_samples_, device_idx, num_active_buckets);
    add_perf_sample(device_swap_active_edge_samples_, device_idx, active_edges.size(0));
    if (device_idx >= 0 && static_cast<std::size_t>(device_idx) < device_current_state_index_.size()) {
        device_current_state_index_[device_idx] = state_idx;
        device_current_active_bucket_count_[device_idx] = num_active_buckets;
        device_current_active_edge_count_[device_idx] = active_edges.size(0);
        device_current_state_partitions_[device_idx] = resident_partitions;
    }
    graph_storage_->setActiveEdges(active_edges, device_idx);
}

void DataLoader::setActiveNodes() {
    torch::Tensor node_ids;

    if (graph_storage_->useInMemorySubGraph()) {
        node_ids = *node_ids_per_buffer_iterator_++;
    } else {
        node_ids = graph_storage_->storage_ptrs_.nodes->range(0, graph_storage_->storage_ptrs_.nodes->getDim0());
        if (node_ids.sizes().size() == 2) {
            node_ids = node_ids.flatten(0, 1);
        }
    }

    auto opts = torch::TensorOptions().dtype(torch::kInt64).device(node_ids.device());
    node_ids = (node_ids.index_select(0, torch::randperm(node_ids.size(0), opts)));
    graph_storage_->setActiveNodes(node_ids);
}

void DataLoader::initializeBatches(bool prepare_encode, int32_t device_idx) {
    int64_t batch_id = 0;
    int64_t start_idx = 0;

    int64_t num_items;
    if (prepare_encode) {
        num_items = graph_storage_->getNumNodes();
    } else {
        if (learning_task_ == LearningTask::LINK_PREDICTION) {
            setActiveEdges(device_idx);
            num_items = graph_storage_->getNumActiveEdges(device_idx);
        } else {
            setActiveNodes();
            num_items = graph_storage_->getNumActiveNodes();
        }
    }

    int64_t batch_size = batch_size_;
    vector<shared_ptr<Batch>> batches;
    while (start_idx < num_items) {
        if (num_items - (start_idx + batch_size) < 0) {
            batch_size = num_items - start_idx;
        }
        shared_ptr<Batch> curr_batch = std::make_shared<Batch>(train_);
        curr_batch->batch_id_ = batch_id + batch_id_offset_;
        curr_batch->start_idx_ = start_idx;
        curr_batch->batch_size_ = batch_size;

        if (prepare_encode) {
            curr_batch->task_ = LearningTask::ENCODE;
        } else {
            curr_batch->task_ = learning_task_;
        }

        batches.emplace_back(curr_batch);
        batch_id++;
        start_idx += batch_size;
    }
    all_batches_[device_idx] = batches;
    batches_left_[device_idx] = batches.size();
    batch_iterators_[device_idx] = all_batches_[device_idx].begin();
    add_perf_sample(device_swap_batch_count_samples_, device_idx, static_cast<int64_t>(batches.size()));
    if (!prepare_encode && learning_task_ == LearningTask::LINK_PREDICTION && graph_storage_->useInMemorySubGraph() &&
        device_idx >= 0 && static_cast<std::size_t>(device_idx) < device_current_state_index_.size() &&
        device_current_state_index_[device_idx] >= 0) {
        int64_t state_sequence = 0;
        if (static_cast<std::size_t>(device_idx) < device_state_build_sequence_.size()) {
            state_sequence = device_state_build_sequence_[device_idx]++;
        }
        SPDLOG_INFO(
            "[perf][epoch {}][gpu {}][state {}] phase={} state_idx={} resident_partitions={} active_buckets={} active_edges={} batches={}",
            epochs_processed_ + 1, device_idx, state_sequence, state_sequence == 0 ? "init" : "swap",
            device_current_state_index_[device_idx], device_current_state_partitions_[device_idx],
            device_current_active_bucket_count_[device_idx], device_current_active_edge_count_[device_idx], batches.size());
    }
}

void DataLoader::setBufferOrdering() {
    int physical_devices = std::max<int>(devices_.size(), 1);
    int requested_active_devices = physical_devices;
    if (training_config_ != nullptr && training_config_->logical_active_devices > 0) {
        requested_active_devices = training_config_->logical_active_devices;
    } else {
        const char *logical_devices_env = std::getenv("GEGE_LOGICAL_ACTIVE_DEVICES");
        if (logical_devices_env != nullptr && logical_devices_env[0] != '\0') {
            try {
                requested_active_devices = std::max(0, std::stoi(logical_devices_env));
            } catch (const std::exception &) {
                SPDLOG_WARN("Ignoring invalid GEGE_LOGICAL_ACTIVE_DEVICES={}", logical_devices_env);
            }
        }
    }

    shared_ptr<PartitionBufferOptions> options;
    if (instance_of<Storage, PartitionBufferStorage>(graph_storage_->storage_ptrs_.node_embeddings)) {
        options = std::dynamic_pointer_cast<PartitionBufferStorage>(graph_storage_->storage_ptrs_.node_embeddings)->options_;
    } else if (instance_of<Storage, MemPartitionBufferStorage>(graph_storage_->storage_ptrs_.node_embeddings)) {
        options = std::dynamic_pointer_cast<MemPartitionBufferStorage>(graph_storage_->storage_ptrs_.node_embeddings)->options_;
    } else if (instance_of<Storage, PartitionBufferStorage>(graph_storage_->storage_ptrs_.node_features)) {
        options = std::dynamic_pointer_cast<PartitionBufferStorage>(graph_storage_->storage_ptrs_.node_features)->options_;
    }

    if (graph_storage_->useInMemorySubGraph() && options == nullptr) {
        throw std::runtime_error("Partition-buffer ordering requires partition-buffer storage options");
    }

    auto reorder_buffer_ordering = [&](std::vector<torch::Tensor> &buffer_states, std::vector<torch::Tensor> &edge_buckets_per_buffer) {
        if (requested_active_devices <= 1 || buffer_states.size() <= 1 || edge_buckets_per_buffer.size() != buffer_states.size()) {
            return;
        }

        if (requested_active_devices != physical_devices) {
            SPDLOG_INFO("Using logical_active_devices={} for buffer ordering on {} physical device(s)", requested_active_devices, physical_devices);
        }

        bool access_aware_scheduler = false;
        const char *access_aware_env = std::getenv("GEGE_ACCESS_AWARE_SCHEDULER");
        if (access_aware_env != nullptr && access_aware_env[0] != '\0' && std::string(access_aware_env) != "0") {
            access_aware_scheduler = true;
        }

        auto permutation = access_aware_scheduler
            ? getAccessAwareDisjointBufferStatePermutation(buffer_states, edge_buckets_per_buffer, requested_active_devices)
            : getDisjointBufferStatePermutation(buffer_states, requested_active_devices);
        bool changed = false;
        for (std::size_t i = 0; i < permutation.size(); i++) {
            if (permutation[i] != static_cast<int64_t>(i)) {
                changed = true;
                break;
            }
        }

        if (!changed) {
            return;
        }

        std::vector<torch::Tensor> reordered_states;
        std::vector<torch::Tensor> reordered_buckets;
        reordered_states.reserve(buffer_states.size());
        reordered_buckets.reserve(edge_buckets_per_buffer.size());
        for (auto idx : permutation) {
            reordered_states.emplace_back(buffer_states[idx]);
            reordered_buckets.emplace_back(edge_buckets_per_buffer[idx]);
        }
        buffer_states = std::move(reordered_states);
        edge_buckets_per_buffer = std::move(reordered_buckets);

        if (access_aware_scheduler) {
            SPDLOG_INFO("Reordered {} buffer states into access-aware disjoint multi-GPU supersteps for {} logical device(s) on {} physical device(s)",
                        buffer_states.size(), requested_active_devices, physical_devices);
        } else {
            SPDLOG_INFO("Reordered {} buffer states into disjoint multi-GPU supersteps for {} logical device(s) on {} physical device(s)",
                        buffer_states.size(), requested_active_devices, physical_devices);
        }
    };

    if (learning_task_ == LearningTask::LINK_PREDICTION) {
        if (graph_storage_->useInMemorySubGraph()) {
            bool access_aware_state_generation = false;
            bool optimized_custom_schedule = parse_env_flag("GEGE_OPTIMIZED_CUSTOM_SCHEDULE", true);
            const char *access_aware_state_generation_env = std::getenv("GEGE_ACCESS_AWARE_STATE_GENERATION");
            if (access_aware_state_generation_env != nullptr && access_aware_state_generation_env[0] != '\0' &&
                std::string(access_aware_state_generation_env) != "0") {
                access_aware_state_generation = true;
            }

            std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> tup;
            bool used_optimized_custom_schedule = false;
            if (access_aware_state_generation && options->edge_bucket_ordering == EdgeBucketOrdering::CUSTOM) {
                tup = getAccessAwareCustomEdgeBucketOrdering(options->num_partitions, options->buffer_capacity, requested_active_devices);
                SPDLOG_INFO("Using access-aware state generation for CUSTOM ordering with {} logical device(s)", requested_active_devices);
            } else if (optimized_custom_schedule && options->edge_bucket_ordering == EdgeBucketOrdering::CUSTOM &&
                       !options->randomly_assign_edge_buckets && requested_active_devices == physical_devices &&
                       requested_active_devices == 4 && options->buffer_capacity == 4) {
                auto edge_bucket_sizes = graph_storage_->storage_ptrs_.edges->getEdgeBucketSizes();
                tup = getOptimizedCustomEdgeBucketOrdering(options->num_partitions, options->buffer_capacity, requested_active_devices,
                                                           batch_size_, edge_bucket_sizes);
                used_optimized_custom_schedule = true;
                SPDLOG_INFO("Using optimized CUSTOM ordering for {} active device(s)", requested_active_devices);
            } else {
                tup = getEdgeBucketOrdering(options->edge_bucket_ordering, options->num_partitions, options->buffer_capacity, options->fine_to_coarse_ratio,
                                            options->num_cache_partitions, options->randomly_assign_edge_buckets);
            }
            buffer_states_ = std::get<0>(tup);
            edge_buckets_per_buffer_ = std::get<1>(tup);
            if (!access_aware_state_generation && !used_optimized_custom_schedule) {
                reorder_buffer_ordering(buffer_states_, edge_buckets_per_buffer_);
            }
            SPDLOG_INFO("buffer_states_ sizes() {}", buffer_states_.size());
            auto edge_buckets_per_buffer_iterator = edge_buckets_per_buffer_.begin();
            edge_buckets_per_buffer_iterators_.resize(devices_.size());
            for (int i = 0; i < devices_.size(); i ++) {
                edge_buckets_per_buffer_iterators_[i] = edge_buckets_per_buffer_iterator;
                edge_buckets_per_buffer_iterator++;
            }
            graph_storage_->setBufferOrdering(buffer_states_);
        }
    } else {
        if (graph_storage_->useInMemorySubGraph()) {
            graph_storage_->storage_ptrs_.train_nodes->load();
            int64_t num_train_nodes = graph_storage_->storage_ptrs_.nodes->getDim0();
            auto tup = getNodePartitionOrdering(
                options->node_partition_ordering, graph_storage_->storage_ptrs_.train_nodes->range(0, num_train_nodes).flatten(0, 1),
                graph_storage_->getNumNodes(), options->num_partitions, options->buffer_capacity, options->fine_to_coarse_ratio, options->num_cache_partitions);
            buffer_states_ = std::get<0>(tup);
            node_ids_per_buffer_ = std::get<1>(tup);

            node_ids_per_buffer_iterator_ = node_ids_per_buffer_.begin();

            graph_storage_->setBufferOrdering(buffer_states_);
        }
    }
}

void DataLoader::clearBatches() { all_batches_ = std::vector<std::vector<shared_ptr<Batch>>>(); }

shared_ptr<Batch> DataLoader::getNextBatch(int32_t device_idx) {
    // std::unique_lock batch_lock(*batch_lock_);
    // // batch_cv_->wait(batch_lock, [this] { return !waiting_for_batches_; });

    shared_ptr<Batch> batch;
    if (batch_iterators_[device_idx] != all_batches_[device_idx].end()) {
        batch = *batch_iterators_[device_idx];
        batch_iterators_[device_idx]++;

        // all_reads_[device_idx] = true;

        // check if all batches have been read
        if (batch_iterators_[device_idx] == all_batches_[device_idx].end()) {
            ++ async_barrier;
            // all_reads_[device_idx] = true;
            if (graph_storage_->useInMemorySubGraph()) {
                if (!graph_storage_->hasSwap(device_idx)) {
                    all_reads_[device_idx] = true;
                }
            } else {
                all_reads_[device_idx] = true;
            }
        }
    } else {
        batch = nullptr;
        if (graph_storage_->useInMemorySubGraph()) {
            if (graph_storage_->hasSwap(device_idx)) {
                // wait for all batches to finish before swapping
                if (swap_tasks_completed == all_batches_.size()) {
                    swap_tasks_completed = 0;
                }
                auto swap_barrier_start = std::chrono::high_resolution_clock::now();
                // batch_cv_->wait(batch_lock, [this, device_idx] { 
                //     return async_barrier.load() % all_batches_.size() == 0; });
                // batch_lock.unlock();
                while(async_barrier.load() % all_batches_.size() != 0) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
                int64_t swap_barrier_elapsed = elapsed_ns(swap_barrier_start, std::chrono::high_resolution_clock::now());
                swap_barrier_wait_ns_.fetch_add(swap_barrier_elapsed);
                if (device_idx < device_swap_barrier_wait_ns_.size()) {
                    device_swap_barrier_wait_ns_[device_idx] += swap_barrier_elapsed;
                }

#ifdef GEGE_CUDA
                c10::cuda::CUDACachingAllocator::emptyCache();
#endif
                // SPDLOG_INFO("Swapping subgraph for device {}", device_idx);
                // auto t1 = std::chrono::high_resolution_clock::now();
                auto update_start = std::chrono::high_resolution_clock::now();
                graph_storage_->updateInMemorySubGraph(device_idx);
                int64_t swap_update_elapsed = elapsed_ns(update_start, std::chrono::high_resolution_clock::now());
                swap_update_ns_.fetch_add(swap_update_elapsed);
                if (device_idx < device_swap_update_ns_.size()) {
                    device_swap_update_ns_[device_idx] += swap_update_elapsed;
                }
                // SPDLOG_INFO("graph_storage_->updateInMemorySubGraph");
#ifdef GEGE_CUDA
                c10::cuda::CUDACachingAllocator::emptyCache();
#endif
                // auto t11 = std::chrono::high_resolution_clock::now();
                // SPDLOG_INFO("Time to updateInMemorySubGraph for device {}: {} ms", device_idx, std::chrono::duration_cast<std::chrono::milliseconds>(t11 - t1).count());
                auto rebuild_start = std::chrono::high_resolution_clock::now();
                initializeBatches(false, device_idx);
                int64_t swap_rebuild_elapsed = elapsed_ns(rebuild_start, std::chrono::high_resolution_clock::now());
                swap_rebuild_ns_.fetch_add(swap_rebuild_elapsed);
                if (device_idx < device_swap_rebuild_ns_.size()) {
                    device_swap_rebuild_ns_[device_idx] += swap_rebuild_elapsed;
                }
                add_perf_sample(device_swap_rebuild_samples_ns_, device_idx, swap_rebuild_elapsed);
                if (device_idx >= 0 && static_cast<std::size_t>(device_idx) < device_current_state_index_.size() &&
                    device_current_state_index_[device_idx] >= 0) {
                    int64_t state_sequence = -1;
                    if (static_cast<std::size_t>(device_idx) < device_state_build_sequence_.size() &&
                        device_state_build_sequence_[device_idx] > 0) {
                        state_sequence = device_state_build_sequence_[device_idx] - 1;
                    }
                    SPDLOG_INFO(
                        "[perf][epoch {}][gpu {}][state {}][rebuild] state_idx={} resident_partitions={} active_buckets={} active_edges={} batches={} rebuild_ms={:.3f}",
                        epochs_processed_ + 1, device_idx, state_sequence, device_current_state_index_[device_idx],
                        device_current_state_partitions_[device_idx], device_current_active_bucket_count_[device_idx],
                        device_current_active_edge_count_[device_idx], all_batches_[device_idx].size(),
                        static_cast<double>(swap_rebuild_elapsed) / 1'000'000.0);
                }
#ifdef GEGE_CUDA
                c10::cuda::CUDACachingAllocator::emptyCache();
#endif

                swap_tasks_completed ++;
                activate_devices_ ++;
                swap_count_.fetch_add(1);
                if (device_idx < device_swap_count_.size()) {
                    device_swap_count_[device_idx] += 1;
                }

                auto swap_sync_start = std::chrono::high_resolution_clock::now();
                while(swap_tasks_completed.load() != all_batches_.size()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
                int64_t swap_sync_elapsed = elapsed_ns(swap_sync_start, std::chrono::high_resolution_clock::now());
                swap_sync_wait_ns_.fetch_add(swap_sync_elapsed);
                if (device_idx < device_swap_sync_wait_ns_.size()) {
                    device_swap_sync_wait_ns_[device_idx] += swap_sync_elapsed;
                }
                // auto t2 = std::chrono::high_resolution_clock::now();
                // SPDLOG_INFO("Finished swapping subgraph for device {}", device_idx);
                // SPDLOG_INFO("Time to swap subgraph for device {}: {} ms", device_idx, std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());
                // if (device_idx == 0) {
                //     graph_storage_->updateInMemorySubGraph();
                //     for (int i = 0; i < devices_.size(); i ++) {
                //         initializeBatches(false, i);
                //     }
                //     swap_tasks_completed = true;
                // } else {
                //     batch_cv_->wait(batch_lock, [this, device_idx] {
                //         return swap_tasks_completed.load() == true;
                //     });
                // }
                batch = *batch_iterators_[device_idx];
                batch_iterators_[device_idx]++;

                // check if all batches have been read
                if (batch_iterators_[device_idx] == all_batches_[device_idx].end()) {
                    if (graph_storage_->useInMemorySubGraph()) {
                        if (!graph_storage_->hasSwap(device_idx)) {
                            all_reads_[device_idx] = true;
                        }
                    } else {
                        all_reads_[device_idx] = true;
                    }
                }
            } else {
                all_reads_[device_idx] = true;
            }
        } else {
            all_reads_[device_idx] = true;
        }
    }
    return batch;
}

bool DataLoader::hasNextBatch(int32_t device_idx) {
    batch_lock_->lock();
    bool ret = !all_reads_[device_idx];
    batch_lock_->unlock();
    return ret;
}

void DataLoader::finishedBatch(int32_t device_idx) {
    batch_lock_->lock();
    batches_left_[device_idx]--;
    if (batches_left_[device_idx] == 0) {
        activate_devices_ --;
    }
    total_batches_processed_++;
    batch_lock_->unlock();
    batch_cv_->notify_all();
}

shared_ptr<Batch> DataLoader::getBatch(at::optional<torch::Device> device, bool perform_map, int32_t device_idx) {
    auto get_batch_start = std::chrono::high_resolution_clock::now();
    auto get_next_batch_start = get_batch_start;
    shared_ptr<Batch> batch = getNextBatch(device_idx);
    int64_t get_next_batch_elapsed = elapsed_ns(get_next_batch_start, std::chrono::high_resolution_clock::now());
    add_perf_stat(get_next_batch_ns_, device_get_next_batch_ns_, device_idx, get_next_batch_elapsed);
    
    if (batch == nullptr) {
        return batch;
    }

    int64_t edge_sample_elapsed = 0;
    int64_t node_sample_elapsed = 0;
    int64_t load_cpu_parameters_elapsed = 0;
    int64_t device_prepare_elapsed = 0;
    int64_t perform_map_elapsed = 0;
 
    if (batch->task_ == LearningTask::LINK_PREDICTION) {
        auto edge_sample_start = std::chrono::high_resolution_clock::now();
        edgeSample(batch, device_idx);
        edge_sample_elapsed = elapsed_ns(edge_sample_start, std::chrono::high_resolution_clock::now());
        add_perf_stat(edge_sample_ns_, device_edge_sample_ns_, device_idx, edge_sample_elapsed);
    } else if (batch->task_ == LearningTask::NODE_CLASSIFICATION || batch->task_ == LearningTask::ENCODE) {
        auto node_sample_start = std::chrono::high_resolution_clock::now();
        nodeSample(batch, device_idx);
        node_sample_elapsed = elapsed_ns(node_sample_start, std::chrono::high_resolution_clock::now());
        add_perf_stat(node_sample_ns_, device_node_sample_ns_, device_idx, node_sample_elapsed);
    }

    auto load_cpu_parameters_start = std::chrono::high_resolution_clock::now();
    loadCPUParameters(batch);
    load_cpu_parameters_elapsed = elapsed_ns(load_cpu_parameters_start, std::chrono::high_resolution_clock::now());
    add_perf_stat(load_cpu_parameters_ns_, device_load_cpu_parameters_ns_, device_idx, load_cpu_parameters_elapsed);

    if (device.has_value()) {
        if (device.value().is_cuda()) {
            auto device_prepare_start = std::chrono::high_resolution_clock::now();
            batch->to(device.value());
            loadGPUParameters(batch);
            batch->dense_graph_.performMap();
            device_prepare_elapsed = elapsed_ns(device_prepare_start, std::chrono::high_resolution_clock::now());
            add_perf_stat(get_batch_device_prepare_ns_, device_get_batch_device_prepare_ns_, device_idx, device_prepare_elapsed);
        }
    }

    if (perform_map) {
        auto perform_map_start = std::chrono::high_resolution_clock::now();
        batch->dense_graph_.performMap();
        perform_map_elapsed = elapsed_ns(perform_map_start, std::chrono::high_resolution_clock::now());
        add_perf_stat(get_batch_perform_map_ns_, device_get_batch_perform_map_ns_, device_idx, perform_map_elapsed);
    }

    int64_t get_batch_total_elapsed = elapsed_ns(get_batch_start, std::chrono::high_resolution_clock::now());
    int64_t get_batch_overhead_elapsed =
        std::max<int64_t>(get_batch_total_elapsed - get_next_batch_elapsed - edge_sample_elapsed - node_sample_elapsed - load_cpu_parameters_elapsed -
                              device_prepare_elapsed - perform_map_elapsed,
                          0LL);
    add_perf_stat(get_batch_overhead_ns_, device_get_batch_overhead_ns_, device_idx, get_batch_overhead_elapsed);

    return batch;
}

void DataLoader::edgeSample(shared_ptr<Batch> batch, int32_t device_idx) {
    int64_t debug_batch_id = -1;
    bool run_stage_debug = should_run_stage_debug(debug_batch_id);
    auto edge_sample_start = std::chrono::high_resolution_clock::now();
    auto step_start = edge_sample_start;
    int64_t get_edges_elapsed = 0;
    int64_t negative_sample_elapsed = 0;
    int64_t map_collect_elapsed = 0;
    int64_t map_lookup_elapsed = 0;
    int64_t map_verify_elapsed = 0;
    int64_t remap_assign_elapsed = 0;
    int64_t finalize_elapsed = 0;

    if (!batch->edges_.defined()) {
        auto get_edges_start = std::chrono::high_resolution_clock::now();
        batch->edges_ = edge_sampler_->getEdges(batch, device_idx);
        get_edges_elapsed = elapsed_ns(get_edges_start, std::chrono::high_resolution_clock::now());
    }
    if (run_stage_debug) {
        auto now = std::chrono::high_resolution_clock::now();
        SPDLOG_INFO("[stage-debug][edgeSample][batch {}][step 1] getEdges ms={:.3f} edges={}x{}",
                    debug_batch_id, elapsed_ms(step_start, now), batch->edges_.size(0), batch->edges_.size(1));
        step_start = now;
    }

    if (negative_sampler_ != nullptr) {
        auto negative_sample_start = std::chrono::high_resolution_clock::now();
        negativeSample(batch, device_idx);
        negative_sample_elapsed = elapsed_ns(negative_sample_start, std::chrono::high_resolution_clock::now());
    }
    if (run_stage_debug) {
        auto now = std::chrono::high_resolution_clock::now();
        int64_t src_neg_numel = batch->src_neg_indices_.defined() ? batch->src_neg_indices_.numel() : 0;
        int64_t dst_neg_numel = batch->dst_neg_indices_.defined() ? batch->dst_neg_indices_.numel() : 0;
        SPDLOG_INFO("[stage-debug][edgeSample][batch {}][step 2] negativeSample ms={:.3f} src_neg_numel={} dst_neg_numel={}",
                    debug_batch_id, elapsed_ms(step_start, now), src_neg_numel, dst_neg_numel);
        step_start = now;
    }
    // std::cout << batch->src_neg_indices_ << std::endl;
    // std::cout << batch->edges_.sizes() << " " <<  batch->src_neg_indices_.sizes() << " " << batch->dst_neg_indices_.sizes() << std::endl;

    // int32_t false_negative_edges_src = 0;
    // int32_t false_negative_edges_dst = 0;
    // for (int32_t i = 0; i < 20; i++) {
    //     for (int32_t j = 0; j < batch->src_neg_indices_.size(1); j++) {
    //         if (batch->edges_[i][0].item<int64_t>() == batch->src_neg_indices_[i / 500][j].item<int64_t>()) {
    //             false_negative_edges_src ++;
    //             // std::cout << "[ " << i << " " << j << " " 
    //             //           << batch->edges_[i][0].item<int64_t>() << " " << batch->edges_[i][-1].item<int64_t>() << " "
    //             //           << batch->src_neg_indices_[i / 500][j].item<int64_t>() << " " << batch->dst_neg_indices_[i / 500][j].item<int64_t>() << " ]" << std::endl;

    //         }
    //         if (batch->edges_[i][-1].item<int32_t>() == batch->dst_neg_indices_[i / 500][j].item<int32_t>()) {
    //             false_negative_edges_dst++;
    //         }
    //     }
    // }
    // SPDLOG_INFO("false_negative_edges_src: {}", false_negative_edges_src);
    // SPDLOG_INFO("false_negative_edges_dst: {}", false_negative_edges_dst);
    auto map_collect_start = std::chrono::high_resolution_clock::now();
    std::vector<torch::Tensor> all_ids = {batch->edges_.select(1, 0), batch->edges_.select(1, -1)};

    if (batch->src_neg_indices_.defined()) {
        all_ids.emplace_back(batch->src_neg_indices_.flatten(0, 1));
    }

    if (batch->dst_neg_indices_.defined()) {
        all_ids.emplace_back(batch->dst_neg_indices_.flatten(0, 1));
    }
    auto map_collect_end = std::chrono::high_resolution_clock::now();
    map_collect_elapsed = elapsed_ns(map_collect_start, map_collect_end);

    torch::Tensor src_mapping;
    torch::Tensor dst_mapping;
    torch::Tensor src_neg_mapping;
    torch::Tensor dst_neg_mapping;

    std::vector<torch::Tensor> mapped_tensors;
    double map_lookup_ms = 0.0;
    double map_verify_ms = 0.0;
    double remap_assign_ms = 0.0;
    MapTensorTiming map_tensor_timing;
    bool has_map_tensor_timing = false;

    if (neighbor_sampler_ != nullptr) {
        auto map_lookup_start = std::chrono::high_resolution_clock::now();
        // get unique nodes in edges and negatives
        batch->root_node_indices_ = std::get<0>(torch::_unique(torch::cat(all_ids)));

        // sample neighbors and get unique nodes
        batch->dense_graph_ = neighbor_sampler_->getNeighbors(batch->root_node_indices_, graph_storage_->current_subgraph_state_->in_memory_subgraph_);
        batch->unique_node_indices_ = batch->dense_graph_.getNodeIDs();

        // map edges and negatives to their corresponding index in unique_node_indices_
        auto tup = torch::sort(batch->unique_node_indices_);
        torch::Tensor sorted_map = std::get<0>(tup);
        torch::Tensor map_to_unsorted = std::get<1>(tup);

        mapped_tensors = apply_tensor_map(sorted_map, all_ids);
        auto map_lookup_end = std::chrono::high_resolution_clock::now();
        map_lookup_ms = elapsed_ms(map_lookup_start, map_lookup_end);
        map_lookup_elapsed = elapsed_ns(map_lookup_start, map_lookup_end);

        int64_t num_nbrs_sampled = batch->dense_graph_.hop_offsets_[-2].item<int64_t>();

        auto remap_assign_start = std::chrono::high_resolution_clock::now();
        std::size_t mapped_tensor_idx = 0;
        src_mapping = map_to_unsorted.index_select(0, mapped_tensors[mapped_tensor_idx++]) - num_nbrs_sampled;
        dst_mapping = map_to_unsorted.index_select(0, mapped_tensors[mapped_tensor_idx++]) - num_nbrs_sampled;

        if (batch->src_neg_indices_.defined()) {
            src_neg_mapping =
                map_to_unsorted.index_select(0, mapped_tensors[mapped_tensor_idx++]).reshape(batch->src_neg_indices_.sizes()) - num_nbrs_sampled;
        }

        if (batch->dst_neg_indices_.defined()) {
            dst_neg_mapping =
                map_to_unsorted.index_select(0, mapped_tensors[mapped_tensor_idx++]).reshape(batch->dst_neg_indices_.sizes()) - num_nbrs_sampled;
        }
        auto remap_assign_end = std::chrono::high_resolution_clock::now();
        remap_assign_ms = elapsed_ms(remap_assign_start, remap_assign_end);
        remap_assign_elapsed = elapsed_ns(remap_assign_start, remap_assign_end);
    } else {
        // map edges and negatives to their corresponding index in unique_node_indices_
        bool map_sorted = !fast_map_tensors_enabled();
        auto map_lookup_start = std::chrono::high_resolution_clock::now();
        auto tup = map_tensors(all_ids, map_sorted, run_stage_debug ? &map_tensor_timing : nullptr, graph_storage_->getNumNodesInMemory(device_idx));
        auto map_lookup_end = std::chrono::high_resolution_clock::now();
        map_lookup_ms = elapsed_ms(map_lookup_start, map_lookup_end);
        map_lookup_elapsed = elapsed_ns(map_lookup_start, map_lookup_end);
        has_map_tensor_timing = run_stage_debug;
    

        batch->unique_node_indices_ = std::get<0>(tup);

        auto map_verify_start = std::chrono::high_resolution_clock::now();
        if (verify_node_mapping_enabled()) {
            // Optional expensive guardrail: avoid synchronizing .item() on every batch unless requested.
            if (torch::any(batch->unique_node_indices_ < 0).item<bool>()) {
                SPDLOG_ERROR("Node mapping is broken. Try repartition again.");
                throw std::runtime_error("");
            }
        }
        auto map_verify_end = std::chrono::high_resolution_clock::now();
        map_verify_ms = elapsed_ms(map_verify_start, map_verify_end);
        map_verify_elapsed = elapsed_ns(map_verify_start, map_verify_end);


        auto remap_assign_start = std::chrono::high_resolution_clock::now();
        mapped_tensors = std::get<1>(tup);

        std::size_t mapped_tensor_idx = 0;
        src_mapping = mapped_tensors[mapped_tensor_idx++];
        dst_mapping = mapped_tensors[mapped_tensor_idx++];

        if (batch->src_neg_indices_.defined()) {
            src_neg_mapping = mapped_tensors[mapped_tensor_idx++].reshape(batch->src_neg_indices_.sizes());
        }

        if (batch->dst_neg_indices_.defined()) {
            dst_neg_mapping = mapped_tensors[mapped_tensor_idx++].reshape(batch->dst_neg_indices_.sizes());
        }
        auto remap_assign_end = std::chrono::high_resolution_clock::now();
        remap_assign_ms = elapsed_ms(remap_assign_start, remap_assign_end);
        remap_assign_elapsed = elapsed_ns(remap_assign_start, remap_assign_end);
    }
    if (run_stage_debug) {
        auto now = std::chrono::high_resolution_clock::now();
        int64_t input_ids_numel = 0;
        for (const auto &ids : all_ids) {
            input_ids_numel += ids.numel();
        }
        int64_t unique_numel = batch->unique_node_indices_.defined() ? batch->unique_node_indices_.numel() : 0;
        int64_t duplicate_count = std::max<int64_t>(input_ids_numel - unique_numel, 0);
        double duplicate_ratio = input_ids_numel > 0 ? (double)duplicate_count / (double)input_ids_numel : 0.0;
        SPDLOG_INFO(
            "[stage-debug][edgeSample][batch {}][step 3] map/remap ms={:.3f} unique_nodes={} input_ids={} duplicates={} duplicate_ratio={:.6f} neighbor_sampler={} fast_map={} verify_map={}",
            debug_batch_id, elapsed_ms(step_start, now), unique_numel, input_ids_numel, duplicate_count, duplicate_ratio,
            neighbor_sampler_ != nullptr, fast_map_tensors_enabled(), verify_node_mapping_enabled());
        SPDLOG_INFO(
            "[stage-debug][edgeSample][batch {}][step 3 breakdown] collect_ids_ms={:.3f} map_lookup_ms={:.3f} verify_ms={:.3f} remap_assign_ms={:.3f}",
            debug_batch_id, elapsed_ms(map_collect_start, map_collect_end), map_lookup_ms, map_verify_ms, remap_assign_ms);
        if (has_map_tensor_timing) {
            SPDLOG_INFO(
                "[stage-debug][edgeSample][batch {}][step 3 map_tensors] validate_ms={:.3f} cat_ms={:.3f} unique_ms={:.3f} unique_wall_ms={:.3f} split_ms={:.3f} total_ms={:.3f}",
                debug_batch_id, map_tensor_timing.validate_ms, map_tensor_timing.cat_ms, map_tensor_timing.unique_ms, map_tensor_timing.unique_wall_ms,
                map_tensor_timing.split_ms, map_tensor_timing.total_ms);
            SPDLOG_INFO(
                "[stage-debug][edgeSample][batch {}][step 3 map_tensors backend] requested_backend={} executed_backend={} fallback={} fallback_backend={} fallback_reason={} total_fallbacks={} cuco_compiled={} capture_file={}",
                debug_batch_id, map_tensor_timing.unique_requested_backend, map_tensor_timing.unique_executed_backend,
                map_tensor_timing.unique_used_fallback, map_tensor_timing.unique_fallback_backend, map_tensor_timing.unique_fallback_reason,
                map_tensor_timing.unique_total_fallbacks, map_tensor_timing.unique_cuco_compiled, map_tensor_timing.capture_path);
        }
        step_start = now;
    }

    auto finalize_start = std::chrono::high_resolution_clock::now();
    if (batch->edges_.size(1) == 2) {
        batch->edges_ = torch::stack({src_mapping, dst_mapping}).transpose(0, 1);
    } else if (batch->edges_.size(1) == 3) {
        batch->edges_ = torch::stack({src_mapping, batch->edges_.select(1, 1), dst_mapping}).transpose(0, 1);
    } else {
        throw TensorSizeMismatchException(batch->edges_, "Edge list must be a 3 or 2 column tensor");
    }

    batch->src_neg_indices_mapping_ = src_neg_mapping;
    batch->dst_neg_indices_mapping_ = dst_neg_mapping;
    finalize_elapsed = elapsed_ns(finalize_start, std::chrono::high_resolution_clock::now());

    add_perf_stat(edge_get_edges_ns_, device_edge_get_edges_ns_, device_idx, get_edges_elapsed);
    add_perf_stat(edge_negative_sample_ns_, device_edge_negative_sample_ns_, device_idx, negative_sample_elapsed);
    add_perf_stat(edge_map_collect_ids_ns_, device_edge_map_collect_ids_ns_, device_idx, map_collect_elapsed);
    add_perf_stat(edge_map_lookup_ns_, device_edge_map_lookup_ns_, device_idx, map_lookup_elapsed);
    add_perf_stat(edge_map_verify_ns_, device_edge_map_verify_ns_, device_idx, map_verify_elapsed);
    add_perf_stat(edge_remap_assign_ns_, device_edge_remap_assign_ns_, device_idx, remap_assign_elapsed);
    add_perf_stat(edge_finalize_ns_, device_edge_finalize_ns_, device_idx, finalize_elapsed);

    if (run_stage_debug) {
        auto now = std::chrono::high_resolution_clock::now();
        SPDLOG_INFO("[stage-debug][edgeSample][batch {}][step 4] finalize ms={:.3f} total_ms={:.3f}",
                    debug_batch_id, elapsed_ms(step_start, now), elapsed_ms(edge_sample_start, now));
    }
}

void DataLoader::nodeSample(shared_ptr<Batch> batch, int32_t device_idx) {
    if (batch->task_ == LearningTask::ENCODE) {
        torch::TensorOptions node_opts = torch::TensorOptions().dtype(torch::kInt64).device(graph_storage_->storage_ptrs_.edges->device_);
        batch->root_node_indices_ = torch::arange(batch->start_idx_, batch->start_idx_ + batch->batch_size_, node_opts);
    } else {
        batch->root_node_indices_ = graph_storage_->getNodeIdsRange(batch->start_idx_, batch->batch_size_).to(torch::kInt64);
    }

    if (graph_storage_->storage_ptrs_.node_labels != nullptr) {
        batch->node_labels_ = graph_storage_->getNodeLabels(batch->root_node_indices_).flatten(0, 1);
    }

    if (graph_storage_->current_subgraph_state_->global_to_local_index_map_.defined()) {
        batch->root_node_indices_ = graph_storage_->current_subgraph_state_->global_to_local_index_map_.index_select(0, batch->root_node_indices_);
    }

    if (neighbor_sampler_ != nullptr) {
        batch->dense_graph_ = neighbor_sampler_->getNeighbors(batch->root_node_indices_, graph_storage_->current_subgraph_state_->in_memory_subgraph_);
        batch->unique_node_indices_ = batch->dense_graph_.getNodeIDs();
    } else {
        batch->unique_node_indices_ = batch->root_node_indices_;
    }
}
/*
    @zizhong
    getNodeCorruptNegatives
*/
void DataLoader::negativeSample(shared_ptr<Batch> batch, int32_t device_idx) {
    bool need_src_negatives = batch->edges_.size(1) == 3 && use_inverse_relations_;
    std::tie(batch->src_neg_indices_, batch->src_neg_filter_, batch->dst_neg_indices_, batch->dst_neg_filter_) =
        negative_sampler_->getNodeCorruptNegatives(graph_storage_->current_subgraph_states_[device_idx]->in_memory_subgraph_, batch->edges_,
                                                   need_src_negatives, device_idx);
}

void DataLoader::loadCPUParameters(shared_ptr<Batch> batch) {
    if (graph_storage_->storage_ptrs_.node_embeddings != nullptr) {
        if (graph_storage_->storage_ptrs_.node_embeddings->device_ != torch::kCUDA) {
            batch->node_embeddings_ = graph_storage_->getNodeEmbeddings(batch->unique_node_indices_);
            if (train_) {
                batch->node_embeddings_state_ = graph_storage_->getNodeEmbeddingState(batch->unique_node_indices_);
                if (negative_sampling_method_ == NegativeSamplingMethod::GAN) {
                    batch->node_embeddings_g_ = graph_storage_->getNodeEmbeddingsG(batch->unique_node_indices_);
                    batch->node_embeddings_state_g_ = graph_storage_->getNodeEmbeddingStateG(batch->unique_node_indices_);
                }
            }
        }
    }

    if (graph_storage_->storage_ptrs_.node_features != nullptr) {
        if (graph_storage_->storage_ptrs_.node_features->device_ != torch::kCUDA) {
            if (only_root_features_) {
                batch->node_features_ = graph_storage_->getNodeFeatures(batch->root_node_indices_);
            } else {
                batch->node_features_ = graph_storage_->getNodeFeatures(batch->unique_node_indices_);
            }
        }
    }

    batch->load_timestamp_ = timestamp_;
}

void DataLoader::loadGPUParameters(shared_ptr<Batch> batch, int32_t device_idx) {
    if (graph_storage_->storage_ptrs_.node_embeddings != nullptr) {
        if (graph_storage_->storage_ptrs_.node_embeddings->device_ == torch::kCUDA) {

            batch->node_embeddings_ = graph_storage_->getNodeEmbeddings(batch->unique_node_indices_, device_idx);

            if (train_) {
                batch->node_embeddings_state_ = graph_storage_->getNodeEmbeddingState(batch->unique_node_indices_, device_idx);
                if (negative_sampling_method_ == NegativeSamplingMethod::GAN) {
                    batch->node_embeddings_g_ = graph_storage_->getNodeEmbeddingsG(batch->unique_node_indices_, device_idx);
                    batch->node_embeddings_state_g_ = graph_storage_->getNodeEmbeddingStateG(batch->unique_node_indices_, device_idx);
                }
            }
        }
    }

    if (graph_storage_->storage_ptrs_.node_features != nullptr) {
        if (graph_storage_->storage_ptrs_.node_features->device_ == torch::kCUDA) {
            if (only_root_features_) {
                batch->node_features_ = graph_storage_->getNodeFeatures(batch->root_node_indices_);
            } else {
                batch->node_features_ = graph_storage_->getNodeFeatures(batch->unique_node_indices_);
            }
        }
    }
}

void DataLoader::updateEmbeddings(shared_ptr<Batch> batch, bool gpu, int32_t device_idx) {
    if (gpu) {
        if (graph_storage_->storage_ptrs_.node_embeddings->device_ == torch::kCUDA) {
            graph_storage_->updateAddNodeEmbeddings(batch->unique_node_indices_, batch->node_gradients_, device_idx);
            graph_storage_->updateAddNodeEmbeddingState(batch->unique_node_indices_, batch->node_state_update_, device_idx);
        }
    } else {
        batch->host_transfer_.synchronize();
        if (graph_storage_->storage_ptrs_.node_embeddings->device_ != torch::kCUDA) {
            graph_storage_->updateAddNodeEmbeddings(batch->unique_node_indices_, batch->node_gradients_);
            graph_storage_->updateAddNodeEmbeddingState(batch->unique_node_indices_, batch->node_state_update_);
        }
        batch->clear();
    }
}

void DataLoader::updateEmbeddingsG(shared_ptr<Batch> batch, bool gpu, int32_t device_idx) {
    if (gpu) {
        if (graph_storage_->storage_ptrs_.node_embeddings_g->device_ == torch::kCUDA) {
            graph_storage_->updateAddNodeEmbeddingsG(batch->unique_node_indices_, batch->node_gradients_g_, device_idx);
            graph_storage_->updateAddNodeEmbeddingStateG(batch->unique_node_indices_, batch->node_state_update_g_, device_idx);
        }
    } else {
        batch->host_transfer_.synchronize();
        if (graph_storage_->storage_ptrs_.node_embeddings_g->device_ != torch::kCUDA) {
            graph_storage_->updateAddNodeEmbeddingsG(batch->unique_node_indices_, batch->node_gradients_g_);
            graph_storage_->updateAddNodeEmbeddingStateG(batch->unique_node_indices_, batch->node_state_update_g_);
        }
        batch->clear();
    }
}

void DataLoader::loadStorage() {
    if (negative_sampler_ != nullptr) {
        negative_sampler_->resetPlanCache();
    }
    loaded_subgraphs = 0;
    setBufferOrdering();
    graph_storage_->load();
    if (negative_sampling_method_ == NegativeSamplingMethod::GAN) {
        graph_storage_->load_g();
    }

    batch_id_offset_ = 0;
    total_batches_processed_ = 0;

    all_batches_ = std::vector<std::vector<shared_ptr<Batch>>>(devices_.size());
    batches_left_ = std::vector<int32_t>(devices_.size());
    batch_iterators_ = std::vector<std::vector<shared_ptr<Batch>>::iterator>(devices_.size());
    all_reads_ = std::vector<bool>(devices_.size());

    for (int device_idx = 0; device_idx < devices_.size(); device_idx ++) {
        all_reads_[device_idx] = false;
    }

    if (!buffer_states_.empty()) {
        for(int device_idx = 0; device_idx < devices_.size(); device_idx ++) {
            graph_storage_->initializeInMemorySubGraph(buffer_states_[loaded_subgraphs ++], devices_[device_idx], device_idx);
        }
    } else {
        graph_storage_->initializeInMemorySubGraph(torch::empty({}));
    }

    if (negative_sampler_ != nullptr) {
        if (instance_of<NegativeSampler, CorruptNodeNegativeSampler>(negative_sampler_)) {
            if (std::dynamic_pointer_cast<CorruptNodeNegativeSampler>(negative_sampler_)->filtered_) {
                graph_storage_->sortAllEdges();
            }
        }
    }
}
