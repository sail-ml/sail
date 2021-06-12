#pragma once

#include <immintrin.h>
#include <omp.h>
#include <algorithm>
#include <vector>
#include "Tensor.h"
#include "dtypes.h"
#include "kernel_utils.h"
#include "ops/ops.h"
#include "tensor_shape.h"
#include "utils.h"

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

template <typename... Ts, typename Op>
void launch_reduction_axis(Op op, const Tensor& input, const Tensor& out,
                           int axis) {
    TensorShape s = TensorShape(input.get_shape());
    int numel = out.get_shape().numel();
    s.recompute();
    s.move_axis(axis, -1);

    get<0, Ts...> __restrict__* p1;
    get<1, Ts...> __restrict__* p2;

    p1 = static_cast<decltype(p1)>(input.get_data());
    p2 = static_cast<decltype(p2)>(out.get_data());

    bool init = true;
    int count = 0;
    int idx = 0;

    for (int i = 0; i < out.numel(); i++) {
        count = 0;
        p2[idx] = 0;
        while (count != s.shape[s.ndim() - 1]) {  // s.numel_avoid(0)) {
            op.call_base(p1[s.d_ptr], p2[idx]);
            s.next();
            count += 1;
        }
        idx += 1;
        init = false;
    }
}

}  // namespace inner_reduction

template <typename... Ts, typename Op>
void Reduction(Op op, const Tensor& input, const Tensor& out) {
    bool allows_avx = false;

    inner_reduction::launch_reduction<Ts...>(op, input, out);
}
template <typename... Ts, typename Op>
void Reduction(Op op, const Tensor& input, const Tensor& out, int index) {
    bool allows_avx = false;

    inner_reduction::launch_reduction_axis<Ts...>(op, input, out, index);
}

}  // namespace sail