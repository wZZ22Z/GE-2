#include "engine/trainer.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <thread>
#include <vector>
#include <nvtx3/nvtx3.hpp>
#include "configuration/options.h"
#include "reporting/logger.h"
#ifdef GEGE_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#endif

using std::get;
using std::tie;

namespace {

int64_t elapsed_ns(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point end) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

double ns_to_ms(int64_t ns) {
    return static_cast<double>(ns) / 1.0e6;
}

double ns_to_ms(double ns) {
    return ns / 1.0e6;
}

std::string format_vector(const std::vector<int64_t> &values) {
    std::ostringstream oss;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << values[i];
    }
    return oss.str();
}

std::string format_ms_vector(const std::vector<int64_t> &values) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << ns_to_ms(values[i]);
    }
    return oss.str();
}

double spread_ms(const std::vector<int64_t> &values) {
    if (values.empty()) {
        return 0.0;
    }
    auto minmax = std::minmax_element(values.begin(), values.end());
    return ns_to_ms(*minmax.second - *minmax.first);
}

// These timers bracket host-side regions in the multi-GPU training loop.
// CUDA kernels may still execute asynchronously after the call returns.
struct DeviceEpochTiming {
    int64_t batch_count = 0;
    int64_t sync_count = 0;
    int64_t batch_fetch_region_ns = 0;
    int64_t gpu_load_region_ns = 0;
    int64_t map_region_ns = 0;
    int64_t compute_region_ns = 0;
    int64_t embedding_update_region_ns = 0;
    int64_t embedding_update_g_region_ns = 0;
    int64_t dense_sync_wait_ns = 0;
    int64_t dense_sync_wait_excl_all_reduce_ns = 0;
    int64_t dense_sync_all_reduce_ns = 0;
    int64_t finalize_region_ns = 0;
};

struct SampleSummary {
    std::size_t count = 0;
    int64_t min = 0;
    int64_t max = 0;
    double median = 0.0;
    double average = 0.0;
};

int64_t sum_member(const std::vector<DeviceEpochTiming> &timings, int64_t DeviceEpochTiming::*member) {
    int64_t total = 0;
    for (const auto &timing : timings) {
        total += timing.*member;
    }
    return total;
}

std::vector<int64_t> collect_ns(const std::vector<DeviceEpochTiming> &timings, int64_t DeviceEpochTiming::*member) {
    std::vector<int64_t> values;
    values.reserve(timings.size());
    for (const auto &timing : timings) {
        values.emplace_back(timing.*member);
    }
    return values;
}

SampleSummary summarize_samples(const std::vector<int64_t> &values) {
    SampleSummary summary;
    summary.count = values.size();
    if (values.empty()) {
        return summary;
    }

    auto minmax = std::minmax_element(values.begin(), values.end());
    summary.min = *minmax.first;
    summary.max = *minmax.second;

    int64_t total = 0;
    for (auto value : values) {
        total += value;
    }
    summary.average = static_cast<double>(total) / static_cast<double>(values.size());

    std::vector<int64_t> sorted(values);
    std::sort(sorted.begin(), sorted.end());
    std::size_t mid = sorted.size() / 2;
    if (sorted.size() % 2 == 0) {
        summary.median = static_cast<double>(sorted[mid - 1] + sorted[mid]) / 2.0;
    } else {
        summary.median = static_cast<double>(sorted[mid]);
    }

    return summary;
}

}  // namespace

SynchronousTrainer::SynchronousTrainer(shared_ptr<DataLoader> dataloader, shared_ptr<Model> model, int logs_per_epoch) {
    dataloader_ = dataloader;
    model_ = model;
    learning_task_ = dataloader_->learning_task_;

    std::string item_name;
    int64_t num_items = 0;
    if (learning_task_ == LearningTask::LINK_PREDICTION) {
        item_name = "Edges";
        num_items = dataloader_->graph_storage_->storage_ptrs_.train_edges->getDim0();
    } else if (learning_task_ == LearningTask::NODE_CLASSIFICATION) {
        item_name = "Nodes";
        num_items = dataloader_->graph_storage_->storage_ptrs_.train_nodes->getDim0();
    }

    progress_reporter_ = std::make_shared<ProgressReporter>(item_name, num_items, logs_per_epoch);
}

void SynchronousTrainer::train(int num_epochs) {
    if (!dataloader_->single_dataset_) {
        dataloader_->setTrainSet();
    }
    dataloader_->initializeBatches(false);
#ifdef GEGE_CUDA
    c10::cuda::CUDACachingAllocator::emptyCache();
#endif

    Timer timer = Timer(false);
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        dataloader_->resetPerfStats();
        timer.start();
        SPDLOG_INFO("################ Starting training epoch {} ################", dataloader_->getEpochsProcessed() + 1);
        while (dataloader_->hasNextBatch()) {
            Timer timer0 = Timer(false);
            timer0.start();

            shared_ptr<Batch> batch = dataloader_->getBatch();

            if (dataloader_->graph_storage_->embeddingsOffDevice()) {
                batch->to(model_->device_);
            } else {
                dataloader_->loadGPUParameters(batch);
            }

            if (batch->node_embeddings_.defined()) {
                batch->node_embeddings_.requires_grad_();
            }

            batch->dense_graph_.performMap();

            model_->train_batch(batch);

            if (batch->node_embeddings_.defined()) {
                if (dataloader_->graph_storage_->embeddingsOffDevice()) {
                    batch->embeddingsToHost();
                } else {
                    dataloader_->updateEmbeddings(batch, true);
                }
                dataloader_->updateEmbeddings(batch, false);
            }

            if (batch->node_embeddings_g_.defined()) {
                if (dataloader_->graph_storage_->embeddingsOffDeviceG()) {
                    batch->embeddingsToHostG();
                } else {
                    dataloader_->updateEmbeddingsG(batch, true);
                }
                dataloader_->updateEmbeddingsG(batch, false);
            }

            batch->clear();
            dataloader_->finishedBatch();
            progress_reporter_->addResult(batch->batch_size_);
        }

        SPDLOG_INFO("################ Finished training epoch {} ################", dataloader_->getEpochsProcessed() + 1);
        timer.stop();

        dataloader_->nextEpoch();
        progress_reporter_->clear();

        std::string item_name;
        int64_t num_items = 0;
        if (learning_task_ == LearningTask::LINK_PREDICTION) {
            item_name = "Edges";
            num_items = dataloader_->graph_storage_->storage_ptrs_.train_edges->getDim0();
        } else if (learning_task_ == LearningTask::NODE_CLASSIFICATION) {
            item_name = "Nodes";
            num_items = dataloader_->graph_storage_->storage_ptrs_.train_nodes->getDim0();
        }

        int64_t epoch_time = timer.getDuration();
        float items_per_second = (float)num_items / ((float)epoch_time / 1000);
        SPDLOG_INFO("Epoch Runtime: {}ms", epoch_time);
        SPDLOG_INFO("{} per Second: {}", item_name, items_per_second);

        auto perf_stats = dataloader_->getPerfStats();
        if (perf_stats.swap_count > 0) {
            SPDLOG_INFO(
                "[perf][epoch {}] swap_count={} swap_barrier_wait_ms={:.3f} swap_update_ms={:.3f} swap_rebuild_ms={:.3f} swap_sync_wait_ms={:.3f}",
                dataloader_->getEpochsProcessed(), perf_stats.swap_count, ns_to_ms(perf_stats.swap_barrier_wait_ns),
                ns_to_ms(perf_stats.swap_update_ns), ns_to_ms(perf_stats.swap_rebuild_ns), ns_to_ms(perf_stats.swap_sync_wait_ns));
            if (!perf_stats.device_swap_count.empty()) {
                SPDLOG_INFO("[perf][epoch {}][device] swap_count={}", dataloader_->getEpochsProcessed(),
                            format_vector(perf_stats.device_swap_count));
                SPDLOG_INFO("[perf][epoch {}][device] swap_barrier_wait_ms={}", dataloader_->getEpochsProcessed(),
                            format_ms_vector(perf_stats.device_swap_barrier_wait_ns));
                SPDLOG_INFO("[perf][epoch {}][device] swap_update_ms={}", dataloader_->getEpochsProcessed(),
                            format_ms_vector(perf_stats.device_swap_update_ns));
                SPDLOG_INFO("[perf][epoch {}][device] swap_rebuild_ms={}", dataloader_->getEpochsProcessed(),
                            format_ms_vector(perf_stats.device_swap_rebuild_ns));
                SPDLOG_INFO("[perf][epoch {}][device] swap_sync_wait_ms={}", dataloader_->getEpochsProcessed(),
                            format_ms_vector(perf_stats.device_swap_sync_wait_ns));
                SPDLOG_INFO(
                    "[perf][epoch {}][spread] swap_barrier_wait_ms={:.3f} swap_update_ms={:.3f} swap_rebuild_ms={:.3f} swap_sync_wait_ms={:.3f}",
                    dataloader_->getEpochsProcessed(), spread_ms(perf_stats.device_swap_barrier_wait_ns),
                    spread_ms(perf_stats.device_swap_update_ns), spread_ms(perf_stats.device_swap_rebuild_ns),
                    spread_ms(perf_stats.device_swap_sync_wait_ns));
            }
        }
    }
}

SynchronousMultiGPUTrainer::SynchronousMultiGPUTrainer(shared_ptr<DataLoader> dataloader, shared_ptr<Model> model, int logs_per_epoch) {
    dataloader_ = dataloader;
    model_ = model;
    learning_task_ = dataloader_->learning_task_;

    std::string item_name;
    int64_t num_items = 0;
    if (learning_task_ == LearningTask::LINK_PREDICTION) {
        item_name = "Edges";
        num_items = dataloader_->graph_storage_->storage_ptrs_.train_edges->getDim0();
    } else if (learning_task_ == LearningTask::NODE_CLASSIFICATION) {
        item_name = "Nodes";
        num_items = dataloader_->graph_storage_->storage_ptrs_.train_nodes->getDim0();
    }

    progress_reporter_ = std::make_shared<ProgressReporter>(item_name, num_items, logs_per_epoch);
}
/*
    original
*/

// void SynchronousMultiGPUTrainer::train(int num_epochs) {
//     if (!dataloader_->single_dataset_) {
//         dataloader_->setTrainSet();
//     }

//     dataloader_->activate_devices_ = model_->device_models_.size();

//     for (int i = 0; i < model_->device_models_.size(); i++) {
//         dataloader_->initializeBatches(false, i);
//     }
// #ifdef GEGE_CUDA
//     c10::cuda::CUDACachingAllocator::emptyCache();
// #endif

//     Timer timer = Timer(false);

//     for (int epoch = 0; epoch < num_epochs; epoch++) {
//         dataloader_->resetPerfStats();
//         timer.start();
//         std::atomic<int64_t> need_sync = 0;
//         std::atomic<int64_t> sync_round = 0;
//         std::atomic<int64_t> all_reduce_ns = 0;
//         std::atomic<int64_t> all_reduce_calls = 0;
//         std::vector<DeviceEpochTiming> device_timings(model_->device_models_.size());
//         std::vector<std::atomic<int64_t>> sync_batch_counts(model_->device_models_.size());
//         std::vector<std::atomic<int64_t>> sync_round_all_reduce_ns(model_->device_models_.size());
//         for (auto &count : sync_batch_counts) {
//             count.store(0);
//         }
//         for (auto &round_all_reduce_ns : sync_round_all_reduce_ns) {
//             round_all_reduce_ns.store(0);
//         }
//         int dense_sync_batches = 1;
//         if (dataloader_->training_config_ != nullptr) {
//             dense_sync_batches = std::max(dataloader_->training_config_->dense_sync_batches, 1);
//         }
//         std::vector<std::thread> threads;

//         SPDLOG_INFO("################ Starting training epoch {} ################", dataloader_->getEpochsProcessed() + 1);
//         SPDLOG_INFO("[perf][epoch {}] dense_sync_batches={} active_devices={}", dataloader_->getEpochsProcessed() + 1, dense_sync_batches,
//                     model_->device_models_.size());
//         for (int32_t device_idx = 0; device_idx < model_->device_models_.size(); device_idx++) {
//             threads.emplace_back(std::thread([this, &need_sync, &sync_round, &all_reduce_ns, &all_reduce_calls, &device_timings, &sync_batch_counts,
//                                               &sync_round_all_reduce_ns, dense_sync_batches, device_idx] {
//                 int64_t local_batches_since_sync = 0;
//                 while (dataloader_->hasNextBatch(device_idx)) {
//                     auto batch_fetch_start = std::chrono::high_resolution_clock::now();
//                     shared_ptr<Batch> batch = dataloader_->getBatch(c10::nullopt, false, device_idx);
//                     auto batch_fetch_end = std::chrono::high_resolution_clock::now();
//                     device_timings[device_idx].batch_fetch_region_ns += elapsed_ns(batch_fetch_start, batch_fetch_end);

//                     bool has_relation = (batch->edges_.size(1) == 3);

//                     auto gpu_load_start = std::chrono::high_resolution_clock::now();
//                     dataloader_->loadGPUParameters(batch, device_idx);
//                     auto gpu_load_end = std::chrono::high_resolution_clock::now();
//                     device_timings[device_idx].gpu_load_region_ns += elapsed_ns(gpu_load_start, gpu_load_end);

//                     if (batch->node_embeddings_.defined()) {
//                         batch->node_embeddings_.requires_grad_();
//                     }

//                     auto map_start = std::chrono::high_resolution_clock::now();
//                     batch->dense_graph_.performMap();
//                     auto map_end = std::chrono::high_resolution_clock::now();
//                     device_timings[device_idx].map_region_ns += elapsed_ns(map_start, map_end);

//                     auto compute_start = std::chrono::high_resolution_clock::now();
//                     model_->device_models_[device_idx]->train_batch(batch, false);
//                     auto compute_end = std::chrono::high_resolution_clock::now();
//                     device_timings[device_idx].compute_region_ns += elapsed_ns(compute_start, compute_end);

//                     if (batch->node_embeddings_.defined()) {
//                         auto embedding_update_start = std::chrono::high_resolution_clock::now();
//                         if (dataloader_->graph_storage_->embeddingsOffDevice()) {
//                             batch->embeddingsToHost();
//                         } else {
//                             dataloader_->updateEmbeddings(batch, true, device_idx);
//                         }
//                         dataloader_->updateEmbeddings(batch, false, device_idx);
//                         auto embedding_update_end = std::chrono::high_resolution_clock::now();
//                         device_timings[device_idx].embedding_update_region_ns += elapsed_ns(embedding_update_start, embedding_update_end);
//                     }

//                     if (batch->node_embeddings_g_.defined()) {
//                         auto embedding_update_g_start = std::chrono::high_resolution_clock::now();
//                         if (dataloader_->graph_storage_->embeddingsOffDeviceG()) {
//                             batch->embeddingsToHostG();
//                         } else {
//                             dataloader_->updateEmbeddingsG(batch, true, device_idx);
//                         }
//                         dataloader_->updateEmbeddingsG(batch, false, device_idx);
//                         auto embedding_update_g_end = std::chrono::high_resolution_clock::now();
//                         device_timings[device_idx].embedding_update_g_region_ns += elapsed_ns(embedding_update_g_start, embedding_update_g_end);
//                     }

//                     if (has_relation) {
//                         local_batches_since_sync++;
//                         bool should_sync = (local_batches_since_sync >= dense_sync_batches) || (dataloader_->batches_left_[device_idx] == 1);
//                         if (should_sync) {
//                             auto sync_start = std::chrono::high_resolution_clock::now();
//                             int64_t round = sync_round.load();
//                             sync_round_all_reduce_ns[device_idx].store(0);
//                             sync_batch_counts[device_idx].store(local_batches_since_sync);
//                             int64_t arrivals = need_sync.fetch_add(1) + 1;
//                             device_timings[device_idx].sync_count++;

//                             if (arrivals == dataloader_->activate_devices_.load()) {
//                                 auto all_reduce_start = std::chrono::high_resolution_clock::now();
//                                 std::vector<int64_t> grad_scales(sync_batch_counts.size(), 1);
//                                 std::vector<int32_t> round_participants;
//                                 for (int i = 0; i < sync_batch_counts.size(); i++) {
//                                     int64_t sync_batches = sync_batch_counts[i].load();
//                                     if (sync_batches > 0) {
//                                         grad_scales[i] = std::max<int64_t>(sync_batches, 1);
//                                         round_participants.emplace_back(i);
//                                     }
//                                 }
//                                 model_->all_reduce(grad_scales);
//                                 int64_t round_all_reduce_elapsed = elapsed_ns(all_reduce_start, std::chrono::high_resolution_clock::now());
//                                 all_reduce_ns.fetch_add(round_all_reduce_elapsed);
//                                 all_reduce_calls.fetch_add(1);
//                                 for (auto &round_all_reduce_ns : sync_round_all_reduce_ns) {
//                                     round_all_reduce_ns.store(round_all_reduce_elapsed);
//                                 }
//                                 for (auto participant_idx : round_participants) {
//                                     sync_batch_counts[participant_idx].store(0);
//                                 }
//                                 need_sync.store(0);
//                                 sync_round.fetch_add(1);
//                             }
//                             while (sync_round.load() == round) {
//                                 std::this_thread::sleep_for(std::chrono::milliseconds(1));
//                             }
//                             int64_t sync_elapsed = elapsed_ns(sync_start, std::chrono::high_resolution_clock::now());
//                             int64_t round_all_reduce_elapsed = sync_round_all_reduce_ns[device_idx].load();
//                             device_timings[device_idx].dense_sync_wait_ns += sync_elapsed;
//                             device_timings[device_idx].dense_sync_all_reduce_ns += round_all_reduce_elapsed;
//                             device_timings[device_idx].dense_sync_wait_excl_all_reduce_ns +=
//                                 std::max<int64_t>(sync_elapsed - round_all_reduce_elapsed, 0LL);
//                             local_batches_since_sync = 0;
//                         }
//                     }

//                     auto finalize_start = std::chrono::high_resolution_clock::now();
//                     batch->clear();
//                     dataloader_->finishedBatch(device_idx);
//                     auto finalize_end = std::chrono::high_resolution_clock::now();
//                     device_timings[device_idx].finalize_region_ns += elapsed_ns(finalize_start, finalize_end);
//                     device_timings[device_idx].batch_count++;
//                 }
//             }));
//         }
//         for (auto &thread : threads) {
//             thread.join();
//         }

//         SPDLOG_INFO("################ Finished training epoch {} ################", dataloader_->getEpochsProcessed() + 1);
//         timer.stop();
//         dataloader_->nextEpoch();
//         progress_reporter_->clear();

//         std::string item_name;
//         int64_t num_items = 0;
//         if (learning_task_ == LearningTask::LINK_PREDICTION) {
//             item_name = "Edges";
//             num_items = dataloader_->graph_storage_->storage_ptrs_.train_edges->getDim0();
//         } else if (learning_task_ == LearningTask::NODE_CLASSIFICATION) {
//             item_name = "Nodes";
//             num_items = dataloader_->graph_storage_->storage_ptrs_.train_nodes->getDim0();
//         }

//         int64_t epoch_time = timer.getDuration();
//         float items_per_second = (float)num_items / ((float)epoch_time / 1000);
//         SPDLOG_INFO("Epoch Runtime: {}ms", epoch_time);
//         SPDLOG_INFO("{} per Second: {}", item_name, items_per_second);

//         auto perf_stats = dataloader_->getPerfStats();
//         bool have_device_swap_stats =
//             perf_stats.device_swap_count.size() == device_timings.size() &&
//             perf_stats.device_swap_barrier_wait_ns.size() == device_timings.size() &&
//             perf_stats.device_swap_update_ns.size() == device_timings.size() &&
//             perf_stats.device_swap_rebuild_ns.size() == device_timings.size() &&
//             perf_stats.device_swap_sync_wait_ns.size() == device_timings.size();
//         bool have_device_batch_fetch_stats =
//             perf_stats.device_get_next_batch_ns.size() == device_timings.size() &&
//             perf_stats.device_edge_sample_ns.size() == device_timings.size() &&
//             perf_stats.device_edge_get_edges_ns.size() == device_timings.size() &&
//             perf_stats.device_edge_negative_sample_ns.size() == device_timings.size() &&
//             perf_stats.device_edge_map_collect_ids_ns.size() == device_timings.size() &&
//             perf_stats.device_edge_map_lookup_ns.size() == device_timings.size() &&
//             perf_stats.device_edge_map_verify_ns.size() == device_timings.size() &&
//             perf_stats.device_edge_remap_assign_ns.size() == device_timings.size() &&
//             perf_stats.device_edge_finalize_ns.size() == device_timings.size() &&
//             perf_stats.device_node_sample_ns.size() == device_timings.size() &&
//             perf_stats.device_load_cpu_parameters_ns.size() == device_timings.size() &&
//             perf_stats.device_get_batch_device_prepare_ns.size() == device_timings.size() &&
//             perf_stats.device_get_batch_perform_map_ns.size() == device_timings.size() &&
//             perf_stats.device_get_batch_overhead_ns.size() == device_timings.size();
//         bool have_device_swap_state_samples =
//             perf_stats.device_swap_active_bucket_samples.size() == device_timings.size() &&
//             perf_stats.device_swap_active_edge_samples.size() == device_timings.size() &&
//             perf_stats.device_swap_batch_count_samples.size() == device_timings.size() &&
//             perf_stats.device_swap_rebuild_samples_ns.size() == device_timings.size();
//         bool have_device_negative_sampler_stats =
//             perf_stats.negative_sampler.device_get_negatives_total_ns.size() == device_timings.size() &&
//             perf_stats.negative_sampler.device_get_negatives_call_count.size() == device_timings.size() &&
//             perf_stats.negative_sampler.device_plan_lock_wait_ns.size() == device_timings.size() &&
//             perf_stats.negative_sampler.device_plan_lock_wait_count.size() == device_timings.size() &&
//             perf_stats.negative_sampler.device_state_pool_hit_count.size() == device_timings.size() &&
//             perf_stats.negative_sampler.device_planned_uniform_fetch_count.size() == device_timings.size() &&
//             perf_stats.negative_sampler.device_cuda_call_count.size() == device_timings.size() &&
//             perf_stats.negative_sampler.device_cpu_call_count.size() == device_timings.size() &&
//             perf_stats.negative_sampler.device_uniform_randint_ns.size() == device_timings.size() &&
//             perf_stats.negative_sampler.device_sample_edge_randint_ns.size() == device_timings.size() &&
//             perf_stats.negative_sampler.device_materialize_ns.size() == device_timings.size() &&
//             perf_stats.negative_sampler.device_filter_ns.size() == device_timings.size() &&
//             perf_stats.negative_sampler.device_get_negatives_samples_ns.size() == device_timings.size() &&
//             perf_stats.negative_sampler.device_plan_lock_wait_samples_ns.size() == device_timings.size();
//         if (!have_device_swap_stats &&
//             (!perf_stats.device_swap_count.empty() || !perf_stats.device_swap_barrier_wait_ns.empty() ||
//              !perf_stats.device_swap_update_ns.empty() || !perf_stats.device_swap_rebuild_ns.empty() ||
//              !perf_stats.device_swap_sync_wait_ns.empty())) {
//             SPDLOG_WARN("[perf][epoch {}] device swap stats are unavailable or size-mismatched for {} GPU timing entries",
//                         dataloader_->getEpochsProcessed(), device_timings.size());
//         }
//         if (!have_device_batch_fetch_stats &&
//             (!perf_stats.device_get_next_batch_ns.empty() || !perf_stats.device_edge_sample_ns.empty() ||
//              !perf_stats.device_edge_get_edges_ns.empty() || !perf_stats.device_edge_negative_sample_ns.empty() ||
//              !perf_stats.device_edge_map_collect_ids_ns.empty() || !perf_stats.device_edge_map_lookup_ns.empty() ||
//              !perf_stats.device_edge_map_verify_ns.empty() || !perf_stats.device_edge_remap_assign_ns.empty() ||
//              !perf_stats.device_edge_finalize_ns.empty() || !perf_stats.device_node_sample_ns.empty() ||
//              !perf_stats.device_load_cpu_parameters_ns.empty() || !perf_stats.device_get_batch_device_prepare_ns.empty() ||
//              !perf_stats.device_get_batch_perform_map_ns.empty() || !perf_stats.device_get_batch_overhead_ns.empty())) {
//             SPDLOG_WARN("[perf][epoch {}] device batch-fetch stats are unavailable or size-mismatched for {} GPU timing entries",
//                         dataloader_->getEpochsProcessed(), device_timings.size());
//         }
//         if (!have_device_swap_state_samples &&
//             (!perf_stats.device_swap_active_bucket_samples.empty() || !perf_stats.device_swap_active_edge_samples.empty() ||
//              !perf_stats.device_swap_batch_count_samples.empty() || !perf_stats.device_swap_rebuild_samples_ns.empty())) {
//             SPDLOG_WARN("[perf][epoch {}] device swap-state samples are unavailable or size-mismatched for {} GPU timing entries",
//                         dataloader_->getEpochsProcessed(), device_timings.size());
//         }
//         if (!have_device_negative_sampler_stats &&
//             (!perf_stats.negative_sampler.device_get_negatives_total_ns.empty() ||
//              !perf_stats.negative_sampler.device_get_negatives_call_count.empty() ||
//              !perf_stats.negative_sampler.device_plan_lock_wait_ns.empty() ||
//              !perf_stats.negative_sampler.device_plan_lock_wait_count.empty() ||
//              !perf_stats.negative_sampler.device_state_pool_hit_count.empty() ||
//              !perf_stats.negative_sampler.device_planned_uniform_fetch_count.empty() ||
//              !perf_stats.negative_sampler.device_cuda_call_count.empty() ||
//              !perf_stats.negative_sampler.device_cpu_call_count.empty() ||
//              !perf_stats.negative_sampler.device_uniform_randint_ns.empty() ||
//              !perf_stats.negative_sampler.device_sample_edge_randint_ns.empty() ||
//              !perf_stats.negative_sampler.device_materialize_ns.empty() ||
//              !perf_stats.negative_sampler.device_filter_ns.empty() ||
//              !perf_stats.negative_sampler.device_get_negatives_samples_ns.empty() ||
//              !perf_stats.negative_sampler.device_plan_lock_wait_samples_ns.empty())) {
//             SPDLOG_WARN("[perf][epoch {}] negative-sampler device stats are unavailable or size-mismatched for {} GPU timing entries",
//                         dataloader_->getEpochsProcessed(), device_timings.size());
//         }
//         SPDLOG_INFO(
//             "[perf][epoch {}] dense_sync_batches={} batches={} sync_points={} all_reduce_calls={} batch_fetch_region_sum_ms={:.3f} gpu_load_region_sum_ms={:.3f} map_region_sum_ms={:.3f} compute_region_sum_ms={:.3f} embedding_update_region_sum_ms={:.3f} embedding_update_g_region_sum_ms={:.3f} dense_sync_wait_sum_ms={:.3f} dense_sync_wait_excl_all_reduce_sum_ms={:.3f} all_reduce_total_ms={:.3f} finalize_region_sum_ms={:.3f} swap_count={} swap_barrier_wait_ms={:.3f} swap_update_ms={:.3f} swap_rebuild_ms={:.3f} swap_sync_wait_ms={:.3f}",
//             dataloader_->getEpochsProcessed(), dense_sync_batches, sum_member(device_timings, &DeviceEpochTiming::batch_count),
//             sum_member(device_timings, &DeviceEpochTiming::sync_count), all_reduce_calls.load(),
//             ns_to_ms(sum_member(device_timings, &DeviceEpochTiming::batch_fetch_region_ns)),
//             ns_to_ms(sum_member(device_timings, &DeviceEpochTiming::gpu_load_region_ns)), ns_to_ms(sum_member(device_timings, &DeviceEpochTiming::map_region_ns)),
//             ns_to_ms(sum_member(device_timings, &DeviceEpochTiming::compute_region_ns)), ns_to_ms(sum_member(device_timings, &DeviceEpochTiming::embedding_update_region_ns)),
//             ns_to_ms(sum_member(device_timings, &DeviceEpochTiming::embedding_update_g_region_ns)), ns_to_ms(sum_member(device_timings, &DeviceEpochTiming::dense_sync_wait_ns)),
//             ns_to_ms(sum_member(device_timings, &DeviceEpochTiming::dense_sync_wait_excl_all_reduce_ns)), ns_to_ms(all_reduce_ns.load()),
//             ns_to_ms(sum_member(device_timings, &DeviceEpochTiming::finalize_region_ns)),
//             perf_stats.swap_count, ns_to_ms(perf_stats.swap_barrier_wait_ns), ns_to_ms(perf_stats.swap_update_ns), ns_to_ms(perf_stats.swap_rebuild_ns),
//             ns_to_ms(perf_stats.swap_sync_wait_ns));
//         SPDLOG_INFO(
//             "[perf][epoch {}][batch_fetch] total_ms={:.3f} get_next_batch_ms={:.3f} edge_sample_ms={:.3f} node_sample_ms={:.3f} load_cpu_parameters_ms={:.3f} device_prepare_ms={:.3f} perform_map_ms={:.3f} overhead_ms={:.3f}",
//             dataloader_->getEpochsProcessed(), ns_to_ms(sum_member(device_timings, &DeviceEpochTiming::batch_fetch_region_ns)),
//             ns_to_ms(perf_stats.get_next_batch_ns), ns_to_ms(perf_stats.edge_sample_ns), ns_to_ms(perf_stats.node_sample_ns),
//             ns_to_ms(perf_stats.load_cpu_parameters_ns), ns_to_ms(perf_stats.get_batch_device_prepare_ns),
//             ns_to_ms(perf_stats.get_batch_perform_map_ns), ns_to_ms(perf_stats.get_batch_overhead_ns));
//         SPDLOG_INFO(
//             "[perf][epoch {}][edge_sample] total_ms={:.3f} get_edges_ms={:.3f} negative_sample_ms={:.3f} collect_ids_ms={:.3f} map_lookup_ms={:.3f} verify_ms={:.3f} remap_assign_ms={:.3f} finalize_ms={:.3f}",
//             dataloader_->getEpochsProcessed(), ns_to_ms(perf_stats.edge_sample_ns), ns_to_ms(perf_stats.edge_get_edges_ns),
//             ns_to_ms(perf_stats.edge_negative_sample_ns), ns_to_ms(perf_stats.edge_map_collect_ids_ns), ns_to_ms(perf_stats.edge_map_lookup_ns),
//             ns_to_ms(perf_stats.edge_map_verify_ns), ns_to_ms(perf_stats.edge_remap_assign_ns), ns_to_ms(perf_stats.edge_finalize_ns));
//         SPDLOG_INFO(
//             "[perf][epoch {}][negative_sampler] calls={} call_ms_total={:.3f} plan_lock_calls={} plan_lock_wait_ms_total={:.3f}",
//             dataloader_->getEpochsProcessed(), perf_stats.negative_sampler.get_negatives_call_count,
//             ns_to_ms(perf_stats.negative_sampler.get_negatives_total_ns), perf_stats.negative_sampler.plan_lock_wait_count,
//             ns_to_ms(perf_stats.negative_sampler.plan_lock_wait_ns));
//         SPDLOG_INFO(
//             "[perf][epoch {}][negative_sampler_breakdown] uniform_randint_ms={:.3f} sample_edge_randint_ms={:.3f} materialize_ms={:.3f} filter_ms={:.3f} state_pool_hits={} planned_uniform_fetches={} cuda_calls={} cpu_calls={}",
//             dataloader_->getEpochsProcessed(), ns_to_ms(perf_stats.negative_sampler.uniform_randint_ns),
//             ns_to_ms(perf_stats.negative_sampler.sample_edge_randint_ns), ns_to_ms(perf_stats.negative_sampler.materialize_ns),
//             ns_to_ms(perf_stats.negative_sampler.filter_ns), perf_stats.negative_sampler.state_pool_hit_count,
//             perf_stats.negative_sampler.planned_uniform_fetch_count, perf_stats.negative_sampler.cuda_call_count,
//             perf_stats.negative_sampler.cpu_call_count);
//         SPDLOG_INFO(
//             "[perf][epoch {}][negative_filter_breakdown] deg_chunk_ids_ms={:.3f} deg_mask_ms={:.3f} deg_nonzero_ms={:.3f} deg_gather_ms={:.3f} deg_finalize_ms={:.3f} gpu_prepare_ms={:.3f} gpu_searchsorted_ms={:.3f} gpu_offsets_ms={:.3f} gpu_repeat_interleave_ms={:.3f} gpu_neighbor_gather_ms={:.3f} gpu_relation_filter_ms={:.3f} gpu_finalize_ms={:.3f}",
//             dataloader_->getEpochsProcessed(), ns_to_ms(perf_stats.negative_sampler.filter_deg_chunk_ids_ns),
//             ns_to_ms(perf_stats.negative_sampler.filter_deg_mask_ns), ns_to_ms(perf_stats.negative_sampler.filter_deg_nonzero_ns),
//             ns_to_ms(perf_stats.negative_sampler.filter_deg_gather_ns), ns_to_ms(perf_stats.negative_sampler.filter_deg_finalize_ns),
//             ns_to_ms(perf_stats.negative_sampler.filter_gpu_prepare_ns), ns_to_ms(perf_stats.negative_sampler.filter_gpu_searchsorted_ns),
//             ns_to_ms(perf_stats.negative_sampler.filter_gpu_offsets_ns),
//             ns_to_ms(perf_stats.negative_sampler.filter_gpu_repeat_interleave_ns),
//             ns_to_ms(perf_stats.negative_sampler.filter_gpu_neighbor_gather_ns),
//             ns_to_ms(perf_stats.negative_sampler.filter_gpu_relation_filter_ns),
//             ns_to_ms(perf_stats.negative_sampler.filter_gpu_finalize_ns));
//         for (int32_t device_idx = 0; device_idx < static_cast<int32_t>(device_timings.size()); device_idx++) {
//             const auto &timing = device_timings[device_idx];
//             int64_t swap_count = have_device_swap_stats ? perf_stats.device_swap_count[device_idx] : 0;
//             int64_t swap_barrier = have_device_swap_stats ? perf_stats.device_swap_barrier_wait_ns[device_idx] : 0;
//             int64_t swap_update = have_device_swap_stats ? perf_stats.device_swap_update_ns[device_idx] : 0;
//             int64_t swap_rebuild = have_device_swap_stats ? perf_stats.device_swap_rebuild_ns[device_idx] : 0;
//             int64_t swap_sync = have_device_swap_stats ? perf_stats.device_swap_sync_wait_ns[device_idx] : 0;
//             int64_t get_next_batch = have_device_batch_fetch_stats ? perf_stats.device_get_next_batch_ns[device_idx] : 0;
//             int64_t edge_sample = have_device_batch_fetch_stats ? perf_stats.device_edge_sample_ns[device_idx] : 0;
//             int64_t edge_get_edges = have_device_batch_fetch_stats ? perf_stats.device_edge_get_edges_ns[device_idx] : 0;
//             int64_t edge_negative_sample = have_device_batch_fetch_stats ? perf_stats.device_edge_negative_sample_ns[device_idx] : 0;
//             int64_t edge_map_collect_ids = have_device_batch_fetch_stats ? perf_stats.device_edge_map_collect_ids_ns[device_idx] : 0;
//             int64_t edge_map_lookup = have_device_batch_fetch_stats ? perf_stats.device_edge_map_lookup_ns[device_idx] : 0;
//             int64_t edge_map_verify = have_device_batch_fetch_stats ? perf_stats.device_edge_map_verify_ns[device_idx] : 0;
//             int64_t edge_remap_assign = have_device_batch_fetch_stats ? perf_stats.device_edge_remap_assign_ns[device_idx] : 0;
//             int64_t edge_finalize = have_device_batch_fetch_stats ? perf_stats.device_edge_finalize_ns[device_idx] : 0;
//             int64_t node_sample = have_device_batch_fetch_stats ? perf_stats.device_node_sample_ns[device_idx] : 0;
//             int64_t load_cpu_parameters = have_device_batch_fetch_stats ? perf_stats.device_load_cpu_parameters_ns[device_idx] : 0;
//             int64_t device_prepare = have_device_batch_fetch_stats ? perf_stats.device_get_batch_device_prepare_ns[device_idx] : 0;
//             int64_t perform_map = have_device_batch_fetch_stats ? perf_stats.device_get_batch_perform_map_ns[device_idx] : 0;
//             int64_t get_batch_overhead = have_device_batch_fetch_stats ? perf_stats.device_get_batch_overhead_ns[device_idx] : 0;
//             SPDLOG_INFO(
//                 "[perf][epoch {}][gpu {}] batches={} sync_points={} batch_fetch_region_ms={:.3f} gpu_load_region_ms={:.3f} map_region_ms={:.3f} compute_region_ms={:.3f} embedding_update_region_ms={:.3f} embedding_update_g_region_ms={:.3f} dense_sync_wait_ms={:.3f} dense_sync_wait_excl_all_reduce_ms={:.3f} dense_sync_all_reduce_ms={:.3f} finalize_region_ms={:.3f} swap_count={} swap_barrier_wait_ms={:.3f} swap_update_ms={:.3f} swap_rebuild_ms={:.3f} swap_sync_wait_ms={:.3f}",
//                 dataloader_->getEpochsProcessed(), device_idx, timing.batch_count, timing.sync_count, ns_to_ms(timing.batch_fetch_region_ns),
//                 ns_to_ms(timing.gpu_load_region_ns), ns_to_ms(timing.map_region_ns), ns_to_ms(timing.compute_region_ns), ns_to_ms(timing.embedding_update_region_ns),
//                 ns_to_ms(timing.embedding_update_g_region_ns), ns_to_ms(timing.dense_sync_wait_ns), ns_to_ms(timing.dense_sync_wait_excl_all_reduce_ns),
//                 ns_to_ms(timing.dense_sync_all_reduce_ns), ns_to_ms(timing.finalize_region_ns), swap_count,
//                 ns_to_ms(swap_barrier), ns_to_ms(swap_update), ns_to_ms(swap_rebuild), ns_to_ms(swap_sync));
//             SPDLOG_INFO(
//                 "[perf][epoch {}][gpu {}][batch_fetch] total_ms={:.3f} get_next_batch_ms={:.3f} edge_sample_ms={:.3f} node_sample_ms={:.3f} load_cpu_parameters_ms={:.3f} device_prepare_ms={:.3f} perform_map_ms={:.3f} overhead_ms={:.3f}",
//                 dataloader_->getEpochsProcessed(), device_idx, ns_to_ms(timing.batch_fetch_region_ns), ns_to_ms(get_next_batch), ns_to_ms(edge_sample),
//                 ns_to_ms(node_sample), ns_to_ms(load_cpu_parameters), ns_to_ms(device_prepare), ns_to_ms(perform_map), ns_to_ms(get_batch_overhead));
//             SPDLOG_INFO(
//                 "[perf][epoch {}][gpu {}][edge_sample] total_ms={:.3f} get_edges_ms={:.3f} negative_sample_ms={:.3f} collect_ids_ms={:.3f} map_lookup_ms={:.3f} verify_ms={:.3f} remap_assign_ms={:.3f} finalize_ms={:.3f}",
//                 dataloader_->getEpochsProcessed(), device_idx, ns_to_ms(edge_sample), ns_to_ms(edge_get_edges), ns_to_ms(edge_negative_sample),
//                 ns_to_ms(edge_map_collect_ids), ns_to_ms(edge_map_lookup), ns_to_ms(edge_map_verify), ns_to_ms(edge_remap_assign), ns_to_ms(edge_finalize));
//             if (have_device_swap_state_samples) {
//                 auto active_bucket_summary = summarize_samples(perf_stats.device_swap_active_bucket_samples[device_idx]);
//                 auto active_edge_summary = summarize_samples(perf_stats.device_swap_active_edge_samples[device_idx]);
//                 auto batch_count_summary = summarize_samples(perf_stats.device_swap_batch_count_samples[device_idx]);
//                 auto rebuild_summary = summarize_samples(perf_stats.device_swap_rebuild_samples_ns[device_idx]);
//                 SPDLOG_INFO(
//                     "[perf][epoch {}][gpu {}][swap_state] states={} active_buckets_min={} active_buckets_med={:.1f} active_buckets_avg={:.1f} active_buckets_max={} active_edges_min={} active_edges_med={:.1f} active_edges_avg={:.1f} active_edges_max={} batches_min={} batches_med={:.1f} batches_avg={:.1f} batches_max={} rebuild_ms_min={:.3f} rebuild_ms_med={:.3f} rebuild_ms_avg={:.3f} rebuild_ms_max={:.3f}",
//                     dataloader_->getEpochsProcessed(), device_idx, active_edge_summary.count, active_bucket_summary.min, active_bucket_summary.median,
//                     active_bucket_summary.average, active_bucket_summary.max, active_edge_summary.min, active_edge_summary.median,
//                     active_edge_summary.average, active_edge_summary.max, batch_count_summary.min, batch_count_summary.median,
//                     batch_count_summary.average, batch_count_summary.max, ns_to_ms(rebuild_summary.min), ns_to_ms(rebuild_summary.median),
//                     ns_to_ms(rebuild_summary.average), ns_to_ms(rebuild_summary.max));
//             }
//             if (have_device_negative_sampler_stats) {
//                 auto negative_call_summary = summarize_samples(perf_stats.negative_sampler.device_get_negatives_samples_ns[device_idx]);
//                 auto negative_lock_summary = summarize_samples(perf_stats.negative_sampler.device_plan_lock_wait_samples_ns[device_idx]);
//                 SPDLOG_INFO(
//                     "[perf][epoch {}][gpu {}][negative_sampler] calls={} call_ms_total={:.3f} call_ms_min={:.3f} call_ms_med={:.3f} call_ms_avg={:.3f} call_ms_max={:.3f} plan_lock_calls={} plan_lock_wait_ms_total={:.3f} plan_lock_wait_ms_min={:.3f} plan_lock_wait_ms_med={:.3f} plan_lock_wait_ms_avg={:.3f} plan_lock_wait_ms_max={:.3f}",
//                     dataloader_->getEpochsProcessed(), device_idx, perf_stats.negative_sampler.device_get_negatives_call_count[device_idx],
//                     ns_to_ms(perf_stats.negative_sampler.device_get_negatives_total_ns[device_idx]), ns_to_ms(negative_call_summary.min),
//                     ns_to_ms(negative_call_summary.median), ns_to_ms(negative_call_summary.average), ns_to_ms(negative_call_summary.max),
//                     perf_stats.negative_sampler.device_plan_lock_wait_count[device_idx],
//                     ns_to_ms(perf_stats.negative_sampler.device_plan_lock_wait_ns[device_idx]), ns_to_ms(negative_lock_summary.min),
//                     ns_to_ms(negative_lock_summary.median), ns_to_ms(negative_lock_summary.average), ns_to_ms(negative_lock_summary.max));
//                 SPDLOG_INFO(
//                     "[perf][epoch {}][gpu {}][negative_sampler_breakdown] uniform_randint_ms={:.3f} sample_edge_randint_ms={:.3f} materialize_ms={:.3f} filter_ms={:.3f} state_pool_hits={} planned_uniform_fetches={} cuda_calls={} cpu_calls={}",
//                     dataloader_->getEpochsProcessed(), device_idx, ns_to_ms(perf_stats.negative_sampler.device_uniform_randint_ns[device_idx]),
//                     ns_to_ms(perf_stats.negative_sampler.device_sample_edge_randint_ns[device_idx]),
//                     ns_to_ms(perf_stats.negative_sampler.device_materialize_ns[device_idx]),
//                     ns_to_ms(perf_stats.negative_sampler.device_filter_ns[device_idx]),
//                     perf_stats.negative_sampler.device_state_pool_hit_count[device_idx],
//                     perf_stats.negative_sampler.device_planned_uniform_fetch_count[device_idx],
//                     perf_stats.negative_sampler.device_cuda_call_count[device_idx],
//                     perf_stats.negative_sampler.device_cpu_call_count[device_idx]);
//                 SPDLOG_INFO(
//                     "[perf][epoch {}][gpu {}][negative_filter_breakdown] deg_chunk_ids_ms={:.3f} deg_mask_ms={:.3f} deg_nonzero_ms={:.3f} deg_gather_ms={:.3f} deg_finalize_ms={:.3f} gpu_prepare_ms={:.3f} gpu_searchsorted_ms={:.3f} gpu_offsets_ms={:.3f} gpu_repeat_interleave_ms={:.3f} gpu_neighbor_gather_ms={:.3f} gpu_relation_filter_ms={:.3f} gpu_finalize_ms={:.3f}",
//                     dataloader_->getEpochsProcessed(), device_idx,
//                     ns_to_ms(perf_stats.negative_sampler.device_filter_deg_chunk_ids_ns[device_idx]),
//                     ns_to_ms(perf_stats.negative_sampler.device_filter_deg_mask_ns[device_idx]),
//                     ns_to_ms(perf_stats.negative_sampler.device_filter_deg_nonzero_ns[device_idx]),
//                     ns_to_ms(perf_stats.negative_sampler.device_filter_deg_gather_ns[device_idx]),
//                     ns_to_ms(perf_stats.negative_sampler.device_filter_deg_finalize_ns[device_idx]),
//                     ns_to_ms(perf_stats.negative_sampler.device_filter_gpu_prepare_ns[device_idx]),
//                     ns_to_ms(perf_stats.negative_sampler.device_filter_gpu_searchsorted_ns[device_idx]),
//                     ns_to_ms(perf_stats.negative_sampler.device_filter_gpu_offsets_ns[device_idx]),
//                     ns_to_ms(perf_stats.negative_sampler.device_filter_gpu_repeat_interleave_ns[device_idx]),
//                     ns_to_ms(perf_stats.negative_sampler.device_filter_gpu_neighbor_gather_ns[device_idx]),
//                     ns_to_ms(perf_stats.negative_sampler.device_filter_gpu_relation_filter_ns[device_idx]),
//                     ns_to_ms(perf_stats.negative_sampler.device_filter_gpu_finalize_ns[device_idx]));
//             }
//         }
//         SPDLOG_INFO(
//             "[perf][epoch {}][spread] batch_fetch_region_ms={:.3f} gpu_load_region_ms={:.3f} map_region_ms={:.3f} compute_region_ms={:.3f} embedding_update_region_ms={:.3f} embedding_update_g_region_ms={:.3f} dense_sync_wait_ms={:.3f} dense_sync_wait_excl_all_reduce_ms={:.3f} dense_sync_all_reduce_ms={:.3f} finalize_region_ms={:.3f}",
//             dataloader_->getEpochsProcessed(), spread_ms(collect_ns(device_timings, &DeviceEpochTiming::batch_fetch_region_ns)),
//             spread_ms(collect_ns(device_timings, &DeviceEpochTiming::gpu_load_region_ns)),
//             spread_ms(collect_ns(device_timings, &DeviceEpochTiming::map_region_ns)),
//             spread_ms(collect_ns(device_timings, &DeviceEpochTiming::compute_region_ns)),
//             spread_ms(collect_ns(device_timings, &DeviceEpochTiming::embedding_update_region_ns)),
//             spread_ms(collect_ns(device_timings, &DeviceEpochTiming::embedding_update_g_region_ns)),
//             spread_ms(collect_ns(device_timings, &DeviceEpochTiming::dense_sync_wait_ns)),
//             spread_ms(collect_ns(device_timings, &DeviceEpochTiming::dense_sync_wait_excl_all_reduce_ns)),
//             spread_ms(collect_ns(device_timings, &DeviceEpochTiming::dense_sync_all_reduce_ns)),
//             spread_ms(collect_ns(device_timings, &DeviceEpochTiming::finalize_region_ns)));
//         if (have_device_batch_fetch_stats) {
//             SPDLOG_INFO(
//                 "[perf][epoch {}][spread][batch_fetch] get_next_batch_ms={:.3f} edge_sample_ms={:.3f} node_sample_ms={:.3f} load_cpu_parameters_ms={:.3f} device_prepare_ms={:.3f} perform_map_ms={:.3f} overhead_ms={:.3f}",
//                 dataloader_->getEpochsProcessed(), spread_ms(perf_stats.device_get_next_batch_ns), spread_ms(perf_stats.device_edge_sample_ns),
//                 spread_ms(perf_stats.device_node_sample_ns), spread_ms(perf_stats.device_load_cpu_parameters_ns),
//                 spread_ms(perf_stats.device_get_batch_device_prepare_ns), spread_ms(perf_stats.device_get_batch_perform_map_ns),
//                 spread_ms(perf_stats.device_get_batch_overhead_ns));
//             SPDLOG_INFO(
//                 "[perf][epoch {}][spread][edge_sample] get_edges_ms={:.3f} negative_sample_ms={:.3f} collect_ids_ms={:.3f} map_lookup_ms={:.3f} verify_ms={:.3f} remap_assign_ms={:.3f} finalize_ms={:.3f}",
//                 dataloader_->getEpochsProcessed(), spread_ms(perf_stats.device_edge_get_edges_ns),
//                 spread_ms(perf_stats.device_edge_negative_sample_ns), spread_ms(perf_stats.device_edge_map_collect_ids_ns),
//                 spread_ms(perf_stats.device_edge_map_lookup_ns), spread_ms(perf_stats.device_edge_map_verify_ns),
//                 spread_ms(perf_stats.device_edge_remap_assign_ns), spread_ms(perf_stats.device_edge_finalize_ns));
//         }
//         if (have_device_negative_sampler_stats) {
//             SPDLOG_INFO(
//                 "[perf][epoch {}][spread][negative_sampler] uniform_randint_ms={:.3f} sample_edge_randint_ms={:.3f} materialize_ms={:.3f} filter_ms={:.3f}",
//                 dataloader_->getEpochsProcessed(), spread_ms(perf_stats.negative_sampler.device_uniform_randint_ns),
//                 spread_ms(perf_stats.negative_sampler.device_sample_edge_randint_ns),
//                 spread_ms(perf_stats.negative_sampler.device_materialize_ns),
//                 spread_ms(perf_stats.negative_sampler.device_filter_ns));
//             SPDLOG_INFO(
//                 "[perf][epoch {}][spread][negative_filter] deg_chunk_ids_ms={:.3f} deg_mask_ms={:.3f} deg_nonzero_ms={:.3f} deg_gather_ms={:.3f} deg_finalize_ms={:.3f} gpu_prepare_ms={:.3f} gpu_searchsorted_ms={:.3f} gpu_offsets_ms={:.3f} gpu_repeat_interleave_ms={:.3f} gpu_neighbor_gather_ms={:.3f} gpu_relation_filter_ms={:.3f} gpu_finalize_ms={:.3f}",
//                 dataloader_->getEpochsProcessed(), spread_ms(perf_stats.negative_sampler.device_filter_deg_chunk_ids_ns),
//                 spread_ms(perf_stats.negative_sampler.device_filter_deg_mask_ns),
//                 spread_ms(perf_stats.negative_sampler.device_filter_deg_nonzero_ns),
//                 spread_ms(perf_stats.negative_sampler.device_filter_deg_gather_ns),
//                 spread_ms(perf_stats.negative_sampler.device_filter_deg_finalize_ns),
//                 spread_ms(perf_stats.negative_sampler.device_filter_gpu_prepare_ns),
//                 spread_ms(perf_stats.negative_sampler.device_filter_gpu_searchsorted_ns),
//                 spread_ms(perf_stats.negative_sampler.device_filter_gpu_offsets_ns),
//                 spread_ms(perf_stats.negative_sampler.device_filter_gpu_repeat_interleave_ns),
//                 spread_ms(perf_stats.negative_sampler.device_filter_gpu_neighbor_gather_ns),
//                 spread_ms(perf_stats.negative_sampler.device_filter_gpu_relation_filter_ns),
//                 spread_ms(perf_stats.negative_sampler.device_filter_gpu_finalize_ns));
//         }
//         if (have_device_swap_stats) {
//             SPDLOG_INFO("[perf][epoch {}][device] swap_count={}", dataloader_->getEpochsProcessed(),
//                         format_vector(perf_stats.device_swap_count));
//             SPDLOG_INFO("[perf][epoch {}][device] swap_barrier_wait_ms={}", dataloader_->getEpochsProcessed(),
//                         format_ms_vector(perf_stats.device_swap_barrier_wait_ns));
//             SPDLOG_INFO("[perf][epoch {}][device] swap_update_ms={}", dataloader_->getEpochsProcessed(),
//                         format_ms_vector(perf_stats.device_swap_update_ns));
//             SPDLOG_INFO("[perf][epoch {}][device] swap_rebuild_ms={}", dataloader_->getEpochsProcessed(),
//                         format_ms_vector(perf_stats.device_swap_rebuild_ns));
//             SPDLOG_INFO("[perf][epoch {}][device] swap_sync_wait_ms={}", dataloader_->getEpochsProcessed(),
//                         format_ms_vector(perf_stats.device_swap_sync_wait_ns));
//             SPDLOG_INFO(
//                 "[perf][epoch {}][spread] swap_barrier_wait_ms={:.3f} swap_update_ms={:.3f} swap_rebuild_ms={:.3f} swap_sync_wait_ms={:.3f}",
//                 dataloader_->getEpochsProcessed(), spread_ms(perf_stats.device_swap_barrier_wait_ns),
//                 spread_ms(perf_stats.device_swap_update_ns), spread_ms(perf_stats.device_swap_rebuild_ns),
//                 spread_ms(perf_stats.device_swap_sync_wait_ns));
//         }
//     }
// }

void SynchronousMultiGPUTrainer::train(int num_epochs) {
    if (!dataloader_->single_dataset_) {
        dataloader_->setTrainSet();
    }

    dataloader_->activate_devices_ = model_->device_models_.size();

    for (int i = 0; i < model_->device_models_.size(); i++) {
        dataloader_->initializeBatches(false, i);
    }
#ifdef GEGE_CUDA
    c10::cuda::CUDACachingAllocator::emptyCache();
#endif

    Timer timer = Timer(false);

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        timer.start();

        std::atomic<int64_t> need_sync = 0;
        std::atomic<int64_t> sync_round = 0;

        int dense_sync_batches = 1;
        if (dataloader_->training_config_ != nullptr) {
            dense_sync_batches = std::max(dataloader_->training_config_->dense_sync_batches, 1);
        }

        std::vector<std::thread> threads;

        SPDLOG_INFO("################ Starting training epoch {} ################", dataloader_->getEpochsProcessed() + 1);
        SPDLOG_INFO("[nvtx][epoch {}] dense_sync_batches={} active_devices={}",
                    dataloader_->getEpochsProcessed() + 1,
                    dense_sync_batches,
                    model_->device_models_.size());

        for (int32_t device_idx = 0; device_idx < model_->device_models_.size(); device_idx++) {
            threads.emplace_back(std::thread([this, &need_sync, &sync_round, dense_sync_batches, device_idx] {
                int64_t local_batches_since_sync = 0;

                while (dataloader_->hasNextBatch(device_idx)) {
                    {
                        std::string range_name = "gpu" + std::to_string(device_idx) + "/getBatch";
                        nvtxRangePushA(range_name.c_str());
                    }
                    shared_ptr<Batch> batch = dataloader_->getBatch(c10::nullopt, false, device_idx);
                    nvtxRangePop();

                    bool has_relation = (batch->edges_.size(1) == 3);

                    {
                        std::string range_name = "gpu" + std::to_string(device_idx) + "/loadGPUParameters";
                        nvtxRangePushA(range_name.c_str());
                    }
                    dataloader_->loadGPUParameters(batch, device_idx);
                    nvtxRangePop();

                    if (batch->node_embeddings_.defined()) {
                        batch->node_embeddings_.requires_grad_();
                    }

                    {
                        std::string range_name = "gpu" + std::to_string(device_idx) + "/performMap";
                        nvtxRangePushA(range_name.c_str());
                    }
                    batch->dense_graph_.performMap();
                    nvtxRangePop();

                    {
                        std::string range_name = "gpu" + std::to_string(device_idx) + "/train_batch";
                        nvtxRangePushA(range_name.c_str());
                    }
                    model_->device_models_[device_idx]->train_batch(batch, false);
                    nvtxRangePop();

                    if (batch->node_embeddings_.defined()) {
                        {
                            std::string range_name = "gpu" + std::to_string(device_idx) + "/updateEmbeddings";
                            nvtxRangePushA(range_name.c_str());
                        }
                        if (dataloader_->graph_storage_->embeddingsOffDevice()) {
                            batch->embeddingsToHost();
                        } else {
                            dataloader_->updateEmbeddings(batch, true, device_idx);
                        }
                        dataloader_->updateEmbeddings(batch, false, device_idx);
                        nvtxRangePop();
                    }

                    if (batch->node_embeddings_g_.defined()) {
                        {
                            std::string range_name = "gpu" + std::to_string(device_idx) + "/updateEmbeddingsG";
                            nvtxRangePushA(range_name.c_str());
                        }
                        if (dataloader_->graph_storage_->embeddingsOffDeviceG()) {
                            batch->embeddingsToHostG();
                        } else {
                            dataloader_->updateEmbeddingsG(batch, true, device_idx);
                        }
                        dataloader_->updateEmbeddingsG(batch, false, device_idx);
                        nvtxRangePop();
                    }

                    if (has_relation) {
                        local_batches_since_sync++;
                        bool should_sync =
                            (local_batches_since_sync >= dense_sync_batches) ||
                            (dataloader_->batches_left_[device_idx] == 1);

                        if (should_sync) {
                            {
                                std::string range_name = "gpu" + std::to_string(device_idx) + "/dense_sync_wait";
                                nvtxRangePushA(range_name.c_str());
                            }

                            int64_t round = sync_round.load();
                            int64_t arrivals = need_sync.fetch_add(1) + 1;

                            if (arrivals == dataloader_->activate_devices_.load()) {
                                {
                                    std::string range_name = "gpu" + std::to_string(device_idx) + "/all_reduce";
                                    nvtxRangePushA(range_name.c_str());
                                }

                                std::vector<int64_t> grad_scales(model_->device_models_.size(), 1);
                                model_->all_reduce(grad_scales);

                                nvtxRangePop();

                                need_sync.store(0);
                                sync_round.fetch_add(1);
                            }

                            while (sync_round.load() == round) {
                                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                            }

                            nvtxRangePop();
                            local_batches_since_sync = 0;
                        }
                    }

                    {
                        std::string range_name = "gpu" + std::to_string(device_idx) + "/finalize_batch";
                        nvtxRangePushA(range_name.c_str());
                    }
                    batch->clear();
                    dataloader_->finishedBatch(device_idx);
                    nvtxRangePop();
                }
            }));
        }

        for (auto &thread : threads) {
            thread.join();
        }

        SPDLOG_INFO("################ Finished training epoch {} ################", dataloader_->getEpochsProcessed() + 1);

        timer.stop();
        dataloader_->nextEpoch();
        progress_reporter_->clear();

        std::string item_name;
        int64_t num_items = 0;
        if (learning_task_ == LearningTask::LINK_PREDICTION) {
            item_name = "Edges";
            num_items = dataloader_->graph_storage_->storage_ptrs_.train_edges->getDim0();
        } else if (learning_task_ == LearningTask::NODE_CLASSIFICATION) {
            item_name = "Nodes";
            num_items = dataloader_->graph_storage_->storage_ptrs_.train_nodes->getDim0();
        }

        int64_t epoch_time = timer.getDuration();
        float items_per_second = (float)num_items / ((float)epoch_time / 1000);

        SPDLOG_INFO("Epoch Runtime: {}ms", epoch_time);
        SPDLOG_INFO("{} per Second: {}", item_name, items_per_second);
    }
}