#pragma once
#define OMP_MIN_VALUE \
    512  // 65536/16 // should probably find a real number ot use

#define JUMP_LOOP(numel, jump) for (; i < numel; i += jump)

#include <immintrin.h>
#include <omp.h>

#include "Tensor.h"
#include "utils.h"

using Tensor = sail::Tensor;
namespace sail {

class Kernel {
   public:
    Kernel() = default;

    virtual ~Kernel() = default;

    Kernel(const Kernel &) = delete;
    Kernel(Kernel &&) = delete;
    Kernel &operator=(const Kernel &) = delete;
    Kernel &operator=(Kernel &&) = delete;
};

}  // namespace sail
