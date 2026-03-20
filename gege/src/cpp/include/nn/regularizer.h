#pragma once

#include "common/datatypes.h"

class Regularizer {
   public:
    virtual ~Regularizer(){};

    virtual torch::Tensor operator()(torch::Tensor src_nodes_embs, torch::Tensor dst_node_embs) = 0;
};

class NormRegularizer : public Regularizer {
   private:
    int norm_;
    float coefficient_;

   public:
    NormRegularizer(int norm, float coefficient);

    torch::Tensor operator()(torch::Tensor src_nodes_embs, torch::Tensor dst_node_embs) override;
};
