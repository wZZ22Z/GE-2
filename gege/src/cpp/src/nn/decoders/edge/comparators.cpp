#include "nn/decoders/edge/comparators.h"

#include <cstdlib>
#include <string>

namespace {

bool parse_env_flag(const char *name, bool default_value) {
    const char *raw = std::getenv(name);
    if (raw == nullptr) {
        return default_value;
    }

    std::string value(raw);
    if (value == "0" || value == "false" || value == "False" || value == "FALSE") {
        return false;
    }
    if (value == "1" || value == "true" || value == "True" || value == "TRUE") {
        return true;
    }
    return default_value;
}

torch::Tensor pairwise_l2_distance_2d_3d(torch::Tensor src, torch::Tensor dst, bool squared_distance) {
    src = pad_and_reshape(src, dst.size(0));

    torch::Tensor x2 = src.pow(2).sum(2).unsqueeze(2);
    torch::Tensor y2 = dst.pow(2).sum(2).unsqueeze(1);
    torch::Tensor xy = torch::matmul(src, dst.transpose(1, 2));
    torch::Tensor distance2 = torch::clamp_min(x2 + y2 - 2 * xy, 1e-8);

    if (!squared_distance) {
        distance2 = torch::sqrt(distance2);
    }

    return distance2.flatten(0, 1).clone();
}

torch::Tensor pairwise_l2_distance_2d_4d(torch::Tensor src, torch::Tensor dst, bool squared_distance) {
    int64_t chunk_num = dst.size(0);
    int64_t num_per_chunk = dst.size(1);
    int64_t selected_negatives_num = dst.size(2);
    int64_t embedding_dim = dst.size(3);

    torch::Tensor src_view = pad_and_reshape(src, chunk_num).reshape({chunk_num * num_per_chunk, embedding_dim, 1});
    torch::Tensor dst_view = dst.reshape({chunk_num * num_per_chunk, selected_negatives_num, embedding_dim});

    torch::Tensor x2 = src_view.squeeze(-1).pow(2).sum(1, true);
    torch::Tensor y2 = dst_view.pow(2).sum(2);
    torch::Tensor xy = torch::bmm(dst_view, src_view).squeeze(-1);
    torch::Tensor distance2 = torch::clamp_min(y2 + x2 - 2 * xy, 1e-8);

    if (!squared_distance) {
        distance2 = torch::sqrt(distance2);
    }

    return distance2.reshape({chunk_num * num_per_chunk, selected_negatives_num}).clone();
}

}  // namespace

torch::Tensor pad_and_reshape(torch::Tensor input, int num_chunks) {
    int num_pos = input.size(0);
    int num_per_chunk = (int)ceil((float)num_pos / num_chunks);

    if (num_per_chunk != num_pos / num_chunks) {
        int64_t new_size = num_per_chunk * num_chunks;
        torch::nn::functional::PadFuncOptions options({0, 0, 0, new_size - num_pos});
        input = torch::nn::functional::pad(input, options);
    }

    input = input.view({num_chunks, num_per_chunk, input.size(1)});

    return input;
}

torch::Tensor L2Compare::operator()(torch::Tensor src, torch::Tensor dst) {
    if (!src.defined() || !dst.defined()) {
        throw UndefinedTensorException();
    }

    if (src.sizes() == dst.sizes()) {
        return torch::pairwise_distance(src, dst);
    } else if (src.dim() == 2 && dst.dim() == 3) {
        return pairwise_l2_distance_2d_3d(src, dst, false);
    } else if (src.dim() == 2 && dst.dim() == 4) {
        if (!parse_env_flag("GEGE_L2_SELECTED_BMM", true)) {
            int64_t chunk_num = dst.size(0);
            torch::Tensor src_view = pad_and_reshape(src, chunk_num).unsqueeze(2);
            double tol = 1e-8;
            return torch::sqrt(torch::clamp_min((src_view - dst).pow(2).sum(-1), tol)).reshape({dst.size(0) * dst.size(1), dst.size(2)}).clone();
        }
        return pairwise_l2_distance_2d_4d(src, dst, false);
    }

    throw TensorSizeMismatchException(dst, "L2Compare expects dst rank 2, 3, or 4 when src rank is 2");
}

torch::Tensor CosineCompare::operator()(torch::Tensor src, torch::Tensor dst) {
    if (!src.defined() || !dst.defined()) {
        throw UndefinedTensorException();
    }

    torch::Tensor src_norm = src.norm(2, -1);
    torch::Tensor dst_norm = dst.norm(2, -1);

    torch::Tensor normalized_src = src * src_norm.clamp_min(1e-10).reciprocal().unsqueeze(-1);
    torch::Tensor normalized_dst = dst * dst_norm.clamp_min(1e-10).reciprocal().unsqueeze(-1);

    if (src.sizes() == dst.sizes()) {
        return (src * dst).sum(-1);
    } else if (src.dim() == 2 && dst.dim() == 3) {
        src = pad_and_reshape(src, dst.size(0));
        return src.bmm(dst.transpose(-1, -2)).flatten(0, 1);
    } else if (src.dim() == 2 && dst.dim() == 4) {
        int64_t chunk_num = dst.size(0);
        int64_t num_per_chunk = dst.size(1);
        int64_t selected_negatives_num = dst.size(2);
        int64_t embedding_dim = dst.size(3);
        torch::Tensor src_view = pad_and_reshape(src, chunk_num).reshape({chunk_num * num_per_chunk, embedding_dim, 1});
        torch::Tensor dst_view = dst.reshape({chunk_num * num_per_chunk, selected_negatives_num, embedding_dim});
        return torch::bmm(dst_view, src_view).reshape({chunk_num * num_per_chunk, selected_negatives_num});
    }

    throw TensorSizeMismatchException(dst, "CosineCompare expects dst rank 2, 3, or 4 when src rank is 2");
}

torch::Tensor DotCompare::operator()(torch::Tensor src, torch::Tensor dst) {
    if (!src.defined() || !dst.defined()) {
        throw UndefinedTensorException();
    }

    if (src.dim() == 2 && dst.dim() == 2) {
        return (src * dst).sum(-1);
    } else if (src.dim() == 2 && dst.dim() == 3) {
        src = pad_and_reshape(src, dst.size(0));
        return src.bmm(dst.transpose(-1, -2)).flatten(0, 1);
    } else if (src.dim() == 2 && dst.dim() == 4) {
        int64_t chunk_num = dst.size(0);
        int64_t num_per_chunk = dst.size(1);
        int64_t selected_negatives_num = dst.size(2);
        int64_t embedding_dim = dst.size(3);
        torch::Tensor src_view = pad_and_reshape(src, chunk_num).reshape({chunk_num * num_per_chunk, embedding_dim, 1});
        torch::Tensor dst_view = dst.reshape({chunk_num * num_per_chunk, selected_negatives_num, embedding_dim});
        return torch::bmm(dst_view, src_view).reshape({chunk_num * num_per_chunk, selected_negatives_num});
    }

    throw TensorSizeMismatchException(dst, "DotCompare expects dst rank 2, 3, or 4 when src rank is 2");
}
