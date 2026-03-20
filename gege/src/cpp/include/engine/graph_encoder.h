#pragma once

#include "data/dataloader.h"
#include "nn/model.h"
#include "reporting/reporting.h"

class GraphEncoder {
   public:
    shared_ptr<DataLoader> dataloader_;
    shared_ptr<ProgressReporter> progress_reporter_;

    virtual ~GraphEncoder(){};
    /**
      Encodes all of the nodes in the graph
      @param seperate_layers. If true, all the nodes at each layer will be encoded before moving onto the next layer.
    */
    virtual void encode(bool separate_layers = false) = 0;
};

class SynchronousGraphEncoder : public GraphEncoder {
    std::shared_ptr<Model> model_;

   public:
    SynchronousGraphEncoder(shared_ptr<DataLoader> sampler, std::shared_ptr<Model> model, int logs_per_epoch = 10);

    void encode(bool separate_layers = false) override;
};
