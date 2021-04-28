#pragma once

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

namespace inner_elementwise {

template <typename... Ts, typename... TensorPack, typename Op>
void launch_binary_elementwise(Op op, const TensorPack... args) {
    std::vector<Tensor> vec = {args...};
    // using T = {Ts...}[0];
    int numel = vec[0].numel();
    int jump = vec[0].info.jump;
    int i = 0;
    bool omp = numel >= OMP_MIN_VALUE;

    get<0, Ts...> __restrict__ *p1;
    get<1, Ts...> __restrict__ *p2;
    get<2, Ts...> __restrict__ *p3;

    p1 = static_cast<decltype(p1)>(vec[0].data);
    p2 = static_cast<decltype(p2)>(vec[1].data);
    p3 = static_cast<decltype(p3)>(vec[2].data);

    if (omp) {
#pragma omp parallel for
        for (i = 0; i < numel; i += 1) {
            op.call_base(p1[i], p2[i], p3[i]);
        }

    } else {
        for (i = 0; i < numel; i += 1) {
            op.call_base(p1[i], p2[i], p3[i]);
        }
    }
}

template <typename... Ts, typename... TensorPack, typename Op>
void launch_binary_elementwise_avx(Op op, const TensorPack &... args) {
    std::vector<Tensor> vec = {args...};
    // using T = {Ts...}[0];
    int numel = vec[0].numel();
    bool aligned = true;  // is_aligned_vec(vec);
    int jump = vec[0].info.jump;
    int i = 0;
    bool omp = numel >= OMP_MIN_VALUE;

    get<0, Ts...> __restrict__ *p1;
    get<1, Ts...> __restrict__ *p2;
    get<2, Ts...> __restrict__ *p3;

    p1 = static_cast<decltype(p1)>(vec[0].data);
    p2 = static_cast<decltype(p2)>(vec[1].data);
    p3 = static_cast<decltype(p3)>(vec[2].data);

    if (omp) {
        if (aligned) {
#pragma omp parallel for
            for (int i = 0; i < numel; i += jump) {
                op.call_avx_aligned(&p1[i], &p2[i], &p3[i]);
            }
        } else {
#pragma omp parallel for
            for (int i = 0; i < numel; i += jump) {
                op.call_avx_non_aligned(&p1[i], &p2[i], &p3[i]);
            }
        }
    } else {
        if (aligned) {
            for (int i = 0; i < numel; i += jump) {
                op.call_avx_aligned(&p1[i], &p2[i], &p3[i]);
            }
        } else {
            for (int i = 0; i < numel; i += jump) {
                op.call_avx_non_aligned(&p1[i], &p2[i], &p3[i]);
            }
        }
    }
}

template <typename... Ts, typename... TensorPack, typename Op>
void launch_binary_elementwise_scalar(Op op, const TensorPack... args) {
    std::vector<Tensor> vec = {args...};
    // using T = {Ts...}[0];
    int numel = vec[0].numel();
    int jump = vec[0].info.jump;
    int i = 0;
    bool omp = numel >= OMP_MIN_VALUE;
    get<0, Ts...> __restrict__ *p1;
    get<1, Ts...> __restrict__ *p2;
    get<2, Ts...> __restrict__ *p3;

    p1 = static_cast<decltype(p1)>(vec[0].data);
    p2 = static_cast<decltype(p2)>(vec[1].data);
    p3 = static_cast<decltype(p3)>(vec[2].data);

    if (omp) {
#pragma omp parallel for
        for (i = 0; i < numel; i += 1) {
            op.call_base(p1[i], p2[i], p3[i]);
        }

    } else {
        for (i = 0; i < numel; i += 1) {
            op.call_base(p1[i], p2[i], p3[i]);
        }
    }
}

template <typename... Ts, typename... TensorPack, typename Op>
void launch_binary_elementwise_avx_scalar(Op op, const TensorPack &... args) {
    std::vector<Tensor> vec = {args...};
    // using T = {Ts...}[0];
    int numel = vec[0].numel();
    bool aligned = true;  // is_aligned_vec(vec);
    int jump = vec[0].info.jump;
    int i = 0;
    bool omp = numel >= OMP_MIN_VALUE;
    get<0, Ts...> __restrict__ *p1;
    get<1, Ts...> __restrict__ *p2;
    get<2, Ts...> __restrict__ *p3;

    p1 = static_cast<decltype(p1)>(vec[0].data);
    p2 = static_cast<decltype(p2)>(vec[1].data);
    p3 = static_cast<decltype(p3)>(vec[2].data);

    if (omp) {
        if (aligned) {
#pragma omp parallel for
            for (int i = 0; i < numel; i += jump) {
                op.call_avx_aligned(&p1[i], &p2[0], &p3[i]);
            }
        } else {
#pragma omp parallel for
            for (int i = 0; i < numel; i += jump) {
                op.call_avx_non_aligned(&p1[i], &p2[0], &p3[i]);
            }
        }
    } else {
        if (aligned) {
            for (int i = 0; i < numel; i += jump) {
                op.call_avx_aligned(&p1[i], &p2[0], &p3[i]);
            }
        } else {
            for (int i = 0; i < numel; i += jump) {
                op.call_avx_non_aligned(&p1[i], &p2[0], &p3[i]);
            }
        }
    }
}

}  // namespace inner_elementwise

template <typename... Ts, typename... TensorPack, typename Op>
void BinaryElementwise(Op op, TensorPack &... args) {
    bool allows_avx = false;
    static_assert(sizeof...(Ts) == sizeof...(args),
                  "Data types must be specified for each Tensor. ");

    // // get dtype to cast to

#ifdef USE_AVX2

    allows_avx = allow_avx(std::forward<TensorPack>(args)...);
    // std::cout << allows_avx << std::endl;
    if (allows_avx) {
        inner_elementwise::launch_binary_elementwise_avx<Ts...>(op, args...);
        return;
    }
// #else
#endif
    inner_elementwise::launch_binary_elementwise<Ts...>(op, args...);
}

template <typename... Ts, typename... TensorPack, typename Op>
void BinaryElementwiseScalar(Op op, TensorPack &... args) {
    bool allows_avx = false;
    static_assert(sizeof...(Ts) == sizeof...(args),
                  "Data types must be specified for each Tensor. ");

    // // get dtype to cast to

#ifdef USE_AVX2

    allows_avx = allow_avx(std::forward<TensorPack>(args)...);
    // std::cout << allows_avx << std::endl;
    if (allows_avx) {
        inner_elementwise::launch_binary_elementwise_avx_scalar<Ts...>(op,
                                                                       args...);
        return;
    }
// #else
#endif
    inner_elementwise::launch_binary_elementwise_scalar<Ts...>(op, args...);
}

}  // namespace sail