#pragma once
#define OMP_MIN_VALUE 512  // should probably find a real number ot use

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

namespace inner_unary {

template <typename... Ts, typename Op>
void launch_unary(Op op, const Tensor& t1, const Tensor& out) {
    int numel = t1.get_shape().numel();
    int jump = t1.get_info().jump;
    int i = 0;

    bool omp = numel >= OMP_MIN_VALUE;

    get<0, Ts...> __restrict__* p1;
    get<1, Ts...> __restrict__* p2;

    p1 = static_cast<decltype(p1)>(t1.get_data());
    p2 = static_cast<decltype(p2)>(out.get_data());

    if (t1.is_view()) {
        TensorShape sh = t1.get_shape();
        sh.reset();
        for (i = 0; i < numel; i += 1) {
            op.call_base(p1[sh.d_ptr], p2[i]);
            sh.next();
        }
        sh.reset();
    } else {
        if (omp) {
#pragma omp parallel for
            for (i = 0; i < numel; i += 1) {
                op.call_base(p1[i], p2[i]);
            }

        } else {
            for (i = 0; i < numel; i += 1) {
                op.call_base(p1[i], p2[i]);
            }
        }
    }
}

}  // namespace inner_unary

template <typename... Ts, typename Op>
void Unary(Op op, const Tensor& t1,
           const Tensor& out) {  // TensorPack &... args) {
    bool allows_avx = false;

    inner_unary::launch_unary<Ts...>(op, t1, out);
}

}  // namespace sail