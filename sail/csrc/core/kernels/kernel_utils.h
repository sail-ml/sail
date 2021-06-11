#pragma once

#include <immintrin.h>
#include <omp.h>
#include <algorithm>
#include <vector>
#include "Tensor.h"
#include "dtypes.h"
#include "utils.h"

inline Dtype avx_support[3] = {Dtype::sInt32, Dtype::sFloat32, Dtype::sFloat64};

template <typename... TensorPack>
inline bool allow_avx(TensorPack... tensors) {
    for (Tensor x : {tensors...}) {
        Dtype* foo = std::find(std::begin(avx_support), std::end(avx_support),
                               x.get_dtype());
        if (foo == std::end(avx_support)) {
            return false;
        }
        return true;
    }
}

template <std::size_t N, typename T, typename... types>
struct get_Nth_type {
    using type = typename get_Nth_type<N - 1, types...>::type;
};

template <typename T, typename... types>
struct get_Nth_type<0, T, types...> {
    using type = T;
};
