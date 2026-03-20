#pragma once

#include <exception>

#include "torch/torch.h"

struct GegeRuntimeException : public std::runtime_error {
   public:
    GegeRuntimeException(const std::string &message) : runtime_error(message) {}
};

struct UndefinedTensorException : public GegeRuntimeException {
   public:
    UndefinedTensorException() : GegeRuntimeException("Tensor undefined") {}
};

struct NANTensorException : public GegeRuntimeException {
   public:
    NANTensorException() : GegeRuntimeException("Tensor contains NANs") {}
};

struct OOMTensorException : public GegeRuntimeException {
   public:
    OOMTensorException() : GegeRuntimeException("Tensor results in OOM") {}
};

struct TensorSizeMismatchException : public GegeRuntimeException {
   public:
    //    TensorSizeMismatchException(torch::Tensor input, std::string message) : GegeRuntimeException((std::stringstream("Tensor size mismatch. Size: ") <<
    //    input.sizes() << " " << message).str()) {}
    TensorSizeMismatchException(torch::Tensor input, std::string message) : GegeRuntimeException(message) {}
};

struct UnexpectedNullPtrException : public GegeRuntimeException {
   public:
    UnexpectedNullPtrException(std::string message = "") : GegeRuntimeException(message) {}
};
