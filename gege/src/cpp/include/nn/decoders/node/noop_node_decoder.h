#pragma once

#include "nn/decoders/node/node_decoder.h"

class NoOpNodeDecoder : public NodeDecoder, public torch::nn::Cloneable<NoOpNodeDecoder> {
   public:
    NoOpNodeDecoder() { learning_task_ = LearningTask::NODE_CLASSIFICATION; };

    torch::Tensor forward(torch::Tensor node_repr) override;

    void reset() override;
};
