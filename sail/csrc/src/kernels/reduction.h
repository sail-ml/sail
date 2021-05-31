#pragma once

#include <immintrin.h>
#include <omp.h>
#include <algorithm>
#include <vector>
#include "../Tensor.h"
#include "../dtypes.h"
#include "../utils.h"
#include "kernel_utils.h"

using Tensor = sail::Tensor;

template <std::size_t N, typename... Args>
using get = typename get_Nth_type<N, Args...>::type;

namespace sail {

namespace inner_reduction {

template <typename... Ts, typename Op>
void launch_reduction(Op op, const Tensor& input, const Tensor& out) {
    int numel = input.get_shape().numel();
    int jump = input.get_info().jump;
    int i = 0;

    get<0, Ts...> __restrict__* p1;
    get<1, Ts...> __restrict__* p2;

    p1 = static_cast<decltype(p1)>(input.get_data());
    p2 = static_cast<decltype(p2)>(out.get_data());

    for (i = 0; i < numel; i += 1) {
        op.call_base(p1[i], p2[0]);
    }
}

}  // namespace inner_reduction

template <typename... Ts, typename Op>
void Reduction(Op op, const Tensor& input, const Tensor& out) {
    bool allows_avx = false;

    inner_reduction::launch_reduction<Ts...>(op, input, out);
}

}  // namespace sail