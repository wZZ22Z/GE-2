#pragma once

#include "common/datatypes.h"
#include "configuration/config.h"

torch::Tensor apply_activation(ActivationFunction activation_function, torch::Tensor input);
