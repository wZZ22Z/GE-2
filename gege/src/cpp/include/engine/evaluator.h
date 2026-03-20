#pragma once

#include <iostream>

#include "data/dataloader.h"
#include "nn/model.h"

/**
  The evaluator runs the evaluation process using the given model and dataset.
*/
class Evaluator {
   public:
    shared_ptr<DataLoader> dataloader_;

    virtual ~Evaluator(){};

    /**
      Runs evaluation process.
      @param validation If true, evaluate on validation set. Otherwise evaluate on test set
    */
    virtual void evaluate(bool validation) = 0;
};

class SynchronousEvaluator : public Evaluator {
    shared_ptr<Model> model_;

   public:
    SynchronousEvaluator(shared_ptr<DataLoader> dataloader, shared_ptr<Model> model);

    void evaluate(bool validation) override;
};
