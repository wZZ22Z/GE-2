#pragma once

#include <configuration/options.h>

#include "common/datatypes.h"

class Decoder {
   public:
    LearningTask learning_task_;

    virtual ~Decoder(){};
};
