#pragma once

#include "common/datatypes.h"
#include "reduction_layer.h"

class ConcatReduction : public ReductionLayer {
   public:
    ConcatReduction(shared_ptr<LayerConfig> layer_config, torch::Device device);

    torch::Tensor forward(std::vector<torch::Tensor> inputs) override;

    void reset() override;
};
