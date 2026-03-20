#include "engine/graph_encoder.h"

#include "reporting/logger.h"
#include <cuda_runtime_api.h>

using std::get;
using std::tie;


SynchronousGraphEncoder::SynchronousGraphEncoder(shared_ptr<DataLoader> dataloader, shared_ptr<Model> model, int logs_per_epoch) {
    dataloader_ = dataloader;
    model_ = model;

    std::string item_name = "Nodes";
    int64_t num_items = dataloader_->graph_storage_->getNumNodes();

    progress_reporter_ = std::make_shared<ProgressReporter>(item_name, num_items, logs_per_epoch);
}

void SynchronousGraphEncoder::encode(bool separate_layers) {
    dataloader_->setEncode();
    Timer timer = Timer(false);
    timer.start();
    SPDLOG_INFO("Start full graph encode");

    while (dataloader_->hasNextBatch()) {
        shared_ptr<Batch> batch = dataloader_->getBatch();
        batch->to(model_->device_);
        dataloader_->loadGPUParameters(batch);

        batch->dense_graph_.performMap();
        torch::Tensor encoded_nodes = model_->encoder_->forward(batch->node_embeddings_, batch->node_features_, batch->dense_graph_, false);
        batch->clear();

        encoded_nodes = encoded_nodes.contiguous().to(torch::kCPU);

        if (model_->device_.is_cuda()) {
            cudaDeviceSynchronize();
        }

        dataloader_->graph_storage_->updatePutEncodedNodesRange(batch->start_idx_, batch->batch_size_, encoded_nodes);
        dataloader_->finishedBatch();
    }

    timer.stop();
    SPDLOG_INFO("Encode Complete: {}s", (double)timer.getDuration() / 1000);
}
