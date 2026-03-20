#pragma once

#include "nn/decoders/edge/edge_decoder.h"

class DistMult : public EdgeDecoder, public torch::nn::Cloneable<DistMult> {
   public:
    DistMult(int num_relations, int embedding_dim, torch::TensorOptions tensor_options = torch::TensorOptions(), bool use_inverse_relations = true,
             EdgeDecoderMethod decoder_method = EdgeDecoderMethod::CORRUPT_NODE);

    void reset() override;
};
