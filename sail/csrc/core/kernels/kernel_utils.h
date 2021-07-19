// allow-no-source
#pragma once

#include <immintrin.h>
#include <omp.h>
#include <algorithm>
#include <vector>
#include "Tensor.h"
#include "dtypes.h"
#include "utils.h"

inline Dtype avx_support[3] = {Dtype::sInt32, Dtype::sFloat32, Dtype::sFloat64};

template <std::size_t N, typename T, typename... types>
struct get_Nth_type {
    using type = typename get_Nth_type<N - 1, types...>::type;
};

template <typename T, typename... types>
struct get_Nth_type<0, T, types...> {
    using type = T;
};
