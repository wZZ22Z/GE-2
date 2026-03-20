#pragma once

#include <common/datatypes.h>

torch::Tensor segment_sum_csr(torch::Tensor src, torch::Tensor indptr, torch::optional<torch::Tensor> optional_out = torch::nullopt);
torch::Tensor gather_csr(torch::Tensor src, torch::Tensor indptr, torch::optional<torch::Tensor> optional_out = torch::nullopt);
