#pragma once

#include "gnn_layer.h"

class GCNLayer : public GNNLayer {
   public:
    shared_ptr<GNNLayerOptions> options_;
    bool use_incoming_;
    bool use_outgoing_;
    torch::Tensor w_;

    GCNLayer(shared_ptr<LayerConfig> layer_config, torch::Device device);

    void reset() override;

    torch::Tensor forward(torch::Tensor inputs, DENSEGraph dense_graph, bool train = true) override;
};
