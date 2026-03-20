#include "engine/evaluator.h"

#include "configuration/constants.h"
#include "reporting/logger.h"


SynchronousEvaluator::SynchronousEvaluator(shared_ptr<DataLoader> dataloader, shared_ptr<Model> model) {
    dataloader_ = dataloader;
    model_ = model;
}

void SynchronousEvaluator::evaluate(bool validation) {
    if (!dataloader_->single_dataset_) {
        if (validation) {
            dataloader_->setValidationSet();
        } else {
            dataloader_->setTestSet();
        }
    }

    dataloader_->initializeBatches(false);

    if (dataloader_->evaluation_negative_sampler_ != nullptr) {
        if (dataloader_->evaluation_config_->negative_sampling->filtered) {
            dataloader_->graph_storage_->sortAllEdges();
        }
    }
    Timer timer = Timer(false);
    timer.start();
    int num_batches = 0;
    while (dataloader_->hasNextBatch()) {
        
        shared_ptr<Batch> batch = dataloader_->getBatch();
        batch->to(model_->device_);
        dataloader_->loadGPUParameters(batch);
        batch->dense_graph_.performMap();
        model_->evaluate_batch(batch);
        dataloader_->finishedBatch();
        batch->clear();
        num_batches++;
    }
    timer.stop();

    model_->reporter_->report();
}
