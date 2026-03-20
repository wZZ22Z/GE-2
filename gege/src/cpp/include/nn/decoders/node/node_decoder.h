#pragma once

#include "nn/decoders/decoder.h"

class NodeDecoder : public Decoder {
   public:
    virtual torch::Tensor forward(torch::Tensor node_repr) = 0;
};
