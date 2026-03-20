#pragma once

#include "common/datatypes.h"
#include "nn/layers/layer.h"

class FeatureLayer : public Layer {
   public:
    int offset_;

    FeatureLayer(shared_ptr<LayerConfig> layer_config, torch::Device device, int offset = 0);

    torch::Tensor forward(torch::Tensor input);

    void reset() override;
};
