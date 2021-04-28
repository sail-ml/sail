#pragma once
#define OMP_MIN_VALUE 512  // should probably find a real number ot use

#include <immintrin.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <vector>
#include "../Tensor.h"
#include "../dtypes.h"
#include "../utils.h"

inline Dtype avx_support[3] = {Dtype::sInt32, Dtype::sFloat32, Dtype::sFloat64};

template <typename... TensorPack>
inline bool allow_avx(TensorPack... tensors) {
    for (Tensor x : {tensors...}) {
        Dtype *foo =
            std::find(std::begin(avx_support), std::end(avx_support), x.dtype);
        if (foo == std::end(avx_support)) {
            return false;
        }
        return true;
    }
}
