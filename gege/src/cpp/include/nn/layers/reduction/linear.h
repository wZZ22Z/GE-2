#pragma once

#include "common/datatypes.h"
#include "reduction_layer.h"

class LinearReduction : public ReductionLayer {
   public:
    torch::Tensor weight_matrix_;

    LinearReduction(shared_ptr<LayerConfig> layer_config, torch::Device device);

    torch::Tensor forward(std::vector<torch::Tensor> inputs) override;

    void reset() override;
};
