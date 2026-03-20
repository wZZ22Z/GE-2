#pragma once

#include "configuration/options.h"
#include "data/dataloader.h"
#include "nn/model.h"
#include "reporting/reporting.h"

/**
  The trainer runs the training process using the given model for the specified number of epochs.
*/
class Trainer {
   public:
    shared_ptr<DataLoader> dataloader_;
    shared_ptr<ProgressReporter> progress_reporter_;
    LearningTask learning_task_;

    virtual ~Trainer(){};
    /**
      Runs training process for embeddings for specified number of epochs.
      @param num_epochs The number of epochs to train for
    */
    virtual void train(int num_epochs = 1) = 0;
};

class SynchronousTrainer : public Trainer {
    std::shared_ptr<Model> model_;

   public:
    SynchronousTrainer(shared_ptr<DataLoader> dataloader, std::shared_ptr<Model> model, int logs_per_epoch = 10);

    void train(int num_epochs = 1) override;
};

class SynchronousMultiGPUTrainer : public Trainer {
    std::shared_ptr<Model> model_;

   public:
    SynchronousMultiGPUTrainer(shared_ptr<DataLoader> dataloader, std::shared_ptr<Model> model, int logs_per_epoch = 10);

    void train(int num_epochs = 1) override;
};
