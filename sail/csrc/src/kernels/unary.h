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
#include "kernel_utils.h"

using namespace std::chrono;

using Tensor = sail::Tensor;

template <std::size_t N, typename... Args>
using get = typename get_Nth_type<N, Args...>::type;

namespace sail {

namespace inner_unary {

template <typename... Ts, typename... TensorPack, typename Op>
void launch_unary(Op op, const TensorPack... args) {
    std::vector<Tensor> vec = {args...};

    int numel = vec[0].numel();
    int jump = vec[0].info.jump;
    int i = 0;

    bool omp = numel >= OMP_MIN_VALUE;

    get<0, Ts...> __restrict__ *p1;
    get<1, Ts...> __restrict__ *p2;

    p1 = static_cast<decltype(p1)>(vec[0].data);
    p2 = static_cast<decltype(p2)>(vec[1].data);

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

}  // namespace inner_unary

template <typename... Ts, typename... TensorPack, typename Op>
void Unary(Op op, TensorPack &... args) {
    bool allows_avx = false;
    static_assert(sizeof...(Ts) == sizeof...(args),
                  "Data types must be specified for each Tensor. ");

    // // get dtype to cast to

    // #ifdef USE_AVX2

    // allows_avx = allow_avx(std::forward<TensorPack>(args)...);
    // // std::cout << allows_avx << std::endl;
    // if (allows_avx) {
    //     // inner_elementwise::launch_binary_elementwise_avx<Ts...>(op,
    //     args...); return;
    // }
    // #else
    // #endif
    inner_unary::launch_unary<Ts...>(op, args...);
}

}  // namespace sail