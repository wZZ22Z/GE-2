#include "gege.h"

#include "common/util.h"
#include "configuration/util.h"
#include "engine/evaluator.h"
#include "engine/graph_encoder.h"
#include "engine/trainer.h"
#include "reporting/logger.h"
#include "storage/checkpointer.h"
#include "storage/io.h"


void encode_and_export(shared_ptr<DataLoader> dataloader, shared_ptr<Model> model, shared_ptr<GegeConfig> gege_config) {
    shared_ptr<GraphEncoder> graph_encoder;
    graph_encoder = std::make_shared<SynchronousGraphEncoder>(dataloader, model);

    string filename = gege_config->storage->model_dir + PathConstants::encoded_nodes_file + PathConstants::file_ext;

    if (fileExists(filename)) {
        remove(filename.c_str());
    }

    int64_t num_nodes = gege_config->storage->dataset->num_nodes;

    int last_stage = gege_config->model->encoder->layers.size() - 1;
    int last_layer = gege_config->model->encoder->layers[last_stage].size() - 1;
    int64_t dim = gege_config->model->encoder->layers[last_stage][last_layer]->output_dim;

    dataloader->graph_storage_->storage_ptrs_.encoded_nodes = std::make_shared<FlatFile>(filename, num_nodes, dim, torch::kFloat32, true);

    graph_encoder->encode();
}

std::tuple<shared_ptr<Model>, shared_ptr<GraphModelStorage>, shared_ptr<DataLoader> > gege_init(shared_ptr<GegeConfig> gege_config, bool train) {
    Timer initialization_timer = Timer(false);
    initialization_timer.start();
    GegeLogger gege_logger = GegeLogger(gege_config->storage->model_dir);
    spdlog::set_default_logger(gege_logger.main_logger_);
    gege_logger.setConsoleLogLevel(gege_config->storage->log_level);

    torch::manual_seed(gege_config->model->random_seed);
    srand(gege_config->model->random_seed);

    std::vector<torch::Device> devices = devices_from_config(gege_config->storage);

    shared_ptr<Model> model;
    shared_ptr<GraphModelStorage> graph_model_storage;

    int epochs_processed = 0;

    if (train) {
        // initialize new model
        if (!gege_config->training->resume_training && gege_config->training->resume_from_checkpoint.empty()) {
            model = initModelFromConfig(gege_config->model, devices, gege_config->storage->dataset->num_relations, true,
                                        gege_config->training->negative_sampling_method, gege_config->training->negative_sampling_selected_ratio);
            graph_model_storage = initializeStorage(model, gege_config->storage, !gege_config->training->resume_training, true,
                                                    gege_config->training->negative_sampling_method);
        } else {
            auto checkpoint_loader = std::make_shared<Checkpointer>();

            string checkpoint_dir = gege_config->storage->model_dir;
            if (!gege_config->training->resume_from_checkpoint.empty()) {
                checkpoint_dir = gege_config->training->resume_from_checkpoint;
            }

            auto tup = checkpoint_loader->load(checkpoint_dir, gege_config, true);
            model = std::get<0>(tup);
            graph_model_storage = std::get<1>(tup);

            CheckpointMeta checkpoint_meta = std::get<2>(tup);
            epochs_processed = checkpoint_meta.num_epochs;
        }
    } else {
        auto checkpoint_loader = std::make_shared<Checkpointer>();

        string checkpoint_dir = gege_config->storage->model_dir;
        if (!gege_config->evaluation->checkpoint_dir.empty()) {
            checkpoint_dir = gege_config->evaluation->checkpoint_dir;
        }
        auto tup = checkpoint_loader->load(checkpoint_dir, gege_config, false);
        model = std::get<0>(tup);
        graph_model_storage = std::get<1>(tup);

        CheckpointMeta checkpoint_meta = std::get<2>(tup);
        epochs_processed = checkpoint_meta.num_epochs;
    }

    bool use_inverse_relations = true;
    if (gege_config->model->decoder != nullptr && gege_config->model->decoder->type != DecoderType::NODE) {
        auto decoder_options = std::dynamic_pointer_cast<EdgeDecoderOptions>(gege_config->model->decoder->options);
        if (decoder_options != nullptr) {
            use_inverse_relations = decoder_options->inverse_edges;
        }
    }

    shared_ptr<DataLoader> dataloader = std::make_shared<DataLoader>(graph_model_storage, model->learning_task_, gege_config->training,
                                                                     gege_config->evaluation, gege_config->model->encoder, devices,
                                                                     gege_config->training->negative_sampling_method, use_inverse_relations);

    dataloader->epochs_processed_ = epochs_processed;

    model->negative_sampler_ = dataloader->training_negative_sampler_;
    for (int i = 1; i < model->device_models_.size(); i++) {
        model->device_models_[i]->negative_sampler_ = model->negative_sampler_;
    }

    initialization_timer.stop();
    int64_t initialization_time = initialization_timer.getDuration();

    SPDLOG_INFO("Initialization Complete: {}s", (double)initialization_time / 1000);

    return std::forward_as_tuple(model, graph_model_storage, dataloader);
}

void gege_train(shared_ptr<GegeConfig> gege_config) {
    auto tup = gege_init(gege_config, true);
    auto model = std::get<0>(tup);
    auto graph_model_storage = std::get<1>(tup);
    auto dataloader = std::get<2>(tup);

    shared_ptr<Trainer> trainer;
    shared_ptr<Evaluator> evaluator;

    shared_ptr<Checkpointer> model_saver;
    CheckpointMeta metadata;
    if (gege_config->training->save_model) {
        model_saver = std::make_shared<Checkpointer>(model, graph_model_storage, gege_config->training->checkpoint);
        metadata.has_state = true;
        metadata.has_encoded = gege_config->storage->export_encoded_nodes;
        metadata.has_model = true;
        metadata.link_prediction = gege_config->model->learning_task == LearningTask::LINK_PREDICTION;
    }
    
    std::vector<torch::Device> devices = devices_from_config(gege_config->storage);

    if (devices.size() == 1 && gege_config->training->dense_sync_batches <= 1) {
        trainer = std::make_shared<SynchronousTrainer>(dataloader, model, gege_config->training->logs_per_epoch);
    } else {
        if (devices.size() == 1) {
            SPDLOG_INFO("SynchronousMultiGPUTrainer single-GPU superstep simulation");
        } else {
            // enable multi-gpu synchronous training by the COVER ordering
            SPDLOG_INFO("SynchronousMultiGPUTrainer");
        }
        trainer = std::make_shared<SynchronousMultiGPUTrainer>(dataloader, model, gege_config->training->logs_per_epoch);
    }

    evaluator = std::make_shared<SynchronousEvaluator>(dataloader, model);

    int checkpoint_interval = gege_config->training->checkpoint->interval;
    for (int epoch = 0; epoch < gege_config->training->num_epochs; epoch++) {
        trainer->train(1);

        if ((epoch + 1) % gege_config->evaluation->epochs_per_eval == 0) {
            if (gege_config->storage->dataset->num_valid != -1) {
                evaluator->evaluate(true);
            }

            if (gege_config->storage->dataset->num_test != -1) {
                evaluator->evaluate(false);
            }
        }

        metadata.num_epochs = dataloader->epochs_processed_;
        if (checkpoint_interval > 0 && (epoch + 1) % checkpoint_interval == 0 && epoch + 1 < gege_config->training->num_epochs) {
            model_saver->create_checkpoint(gege_config->storage->model_dir, metadata, dataloader->epochs_processed_);
        }
    }
        
    if (gege_config->training->save_model) {
        model_saver->save(gege_config->storage->model_dir, metadata);

        if (gege_config->storage->export_encoded_nodes) {
            encode_and_export(dataloader, model, gege_config);
        }
    }
}

void gege_eval(shared_ptr<GegeConfig> gege_config) {
    auto tup = gege_init(gege_config, false);
    auto model = std::get<0>(tup);
    auto graph_model_storage = std::get<1>(tup);
    auto dataloader = std::get<2>(tup);

    shared_ptr<Evaluator> evaluator;
    if (gege_config->evaluation->epochs_per_eval > 0) {
        evaluator = std::make_shared<SynchronousEvaluator>(dataloader, model);
        evaluator->evaluate(false);
    }

    if (gege_config->storage->export_encoded_nodes) {
        encode_and_export(dataloader, model, gege_config);
    }
}

void gege(int argc, char *argv[]) {
    (void)argc;

    bool train = true;
    string command_path = string(argv[0]);
    string config_path = string(argv[1]);
    string command_name = command_path.substr(command_path.find_last_of("/\\") + 1);
    if (strcmp(command_name.c_str(), "gege_eval") == 0) {
        train = false;
    }

    shared_ptr<GegeConfig> gege_config = loadConfig(config_path, true);

    if (train) {
        gege_train(gege_config);
    } else {
        gege_eval(gege_config);
    }
}

int main(int argc, char *argv[]) { gege(argc, argv); }
