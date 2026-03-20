#pragma once

#include "common/datatypes.h"
#include "configuration/config.h"
#include "data/graph.h"
#include "nn/initialization.h"
#include "nn/layers/layer.h"

class GNNLayer : public Layer {
   public:
    int input_dim_;
    int output_dim_;

    virtual ~GNNLayer(){};

    virtual torch::Tensor forward(torch::Tensor inputs, DENSEGraph dense_graph, bool train) { return torch::Tensor(); };
};
