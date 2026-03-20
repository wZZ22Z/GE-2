#pragma once

#include "nn/decoders/edge/edge_decoder.h"

class TransE : public EdgeDecoder, public torch::nn::Cloneable<TransE> {
   public:
    TransE(int num_relations, int embedding_dim, torch::TensorOptions tensor_options = torch::TensorOptions(), bool use_inverse_relations = true,
           EdgeDecoderMethod decoder_method = EdgeDecoderMethod::CORRUPT_NODE);

    void reset() override;
};
