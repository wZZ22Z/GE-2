#pragma once

#include "config.h"

std::vector<torch::Device> devices_from_config(std::shared_ptr<StorageConfig> storage_config);
