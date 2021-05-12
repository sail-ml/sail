#pragma once

#include <immintrin.h>
#include <omp.h>
#include <algorithm>
#include <cassert>  // needed for xsimd
#include <vector>
#include "../Tensor.h"
#include "../dtypes.h"
#include "../tensor_shape.h"
#include "../utils.h"
#include "kernel_utils.h"
#include "xsimd/xsimd.hpp"

using Tensor = sail::Tensor;

template <std::size_t N, typename... Args>
using get = typename get_Nth_type<N, Args...>::type;

namespace sail {

namespace inner_elementwise {

template <typename... Ts, typename Op>
void launch_binary_elementwise(Op op, const Tensor &t1, const Tensor &t2,
                               const Tensor &out) {
    // using T = {Ts...}[0];
    int numel = t1.get_shape().numel();
    int jump = t1.info.jump;
    int i = 0;
    bool omp = numel >= OMP_MIN_VALUE;

    get<0, Ts...> __restrict__ *p1;
    get<1, Ts...> __restrict__ *p2;
    get<2, Ts...> __restrict__ *p3;

    p1 = static_cast<decltype(p1)>(t1.get_data());
    p2 = static_cast<decltype(p2)>(t2.get_data());
    p3 = static_cast<decltype(p3)>(out.get_data());

    TensorShape s1 = t1.get_shape();
    TensorShape s2 = t2.get_shape();
    TensorShape sOut = out.get_shape();

    if (omp) {
#pragma omp parallel for
        for (i = 0; i < numel; i += 1) {
            op.call_base(p1[s1.d_ptr], p2[s2.d_ptr], p3[sOut.d_ptr]);
#pragma omp critical
            {
                s1.next();
                s2.next();
                sOut.next();
            }
        }

    } else {
        for (i = 0; i < numel; i += 1) {
            op.call_base(p1[s1.d_ptr], p2[s2.d_ptr], p3[sOut.d_ptr]);
            s1.next();
            s2.next();
            sOut.next();
        }
    }
    s1.reset();
    s2.reset();
    sOut.reset();
}

template <typename... Ts, typename Op>
void launch_binary_elementwise_avx(Op op, const Tensor &t1, const Tensor &t2,
                                   const Tensor &out) {
    // std::vector<Tensor> vec = {args...};
    int numel = t1.get_shape().numel();
    int jump = t1.get_info().jump;
    int i = 0;
    bool omp = numel >= OMP_MIN_VALUE;

    get<0, Ts...> __restrict__ *p1;
    get<1, Ts...> __restrict__ *p2;
    get<2, Ts...> __restrict__ *p3;

    p1 = static_cast<decltype(p1)>(t1.get_data());
    p2 = static_cast<decltype(p2)>(t2.get_data());
    p3 = static_cast<decltype(p3)>(out.get_data());

    bool aligned = true;  // is_aligned_vec(vec);

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
void launch_binary_elementwise_broadcast(Op op, const TensorPack &... args) {
    std::vector<Tensor> vec = {args...};
    // using T = {Ts...}[0];
    int numel = vec[2].numel();
    int jump = vec[2].info.jump;
    int i = 0;
    bool omp = numel >= OMP_MIN_VALUE;
    get<0, Ts...> __restrict__ *p1;
    get<1, Ts...> __restrict__ *p2;
    get<2, Ts...> __restrict__ *p3;

    p1 = static_cast<decltype(p1)>(vec[0].get_data());
    p2 = static_cast<decltype(p2)>(vec[1].get_data());
    p3 = static_cast<decltype(p3)>(vec[2].get_data());

    TensorShape vec0_shape = vec[0].get_shape();
    TensorShape vec1_shape = vec[1].get_shape();

    if (vec0_shape.ndim() < vec[1].get_shape().ndim()) {
        while (vec0_shape.shape.size() < vec1_shape.ndim()) {
            vec0_shape.shape.insert(vec0_shape.shape.begin(), 1);
            vec0_shape.strides.insert(vec0_shape.strides.begin(), 0);
        }
    } else {
        while (vec1_shape.shape.size() < vec0_shape.ndim()) {
            vec1_shape.shape.insert(vec1_shape.shape.begin(), 1);
            vec1_shape.strides.insert(vec1_shape.strides.begin(), 0);
        }
    }

    TensorSize shape1 = vec0_shape.shape;
    TensorSize shape2 = vec1_shape.shape;
    for (int i = 0; i < vec0_shape.ndim(); i++) {
        if (shape1[i] != shape2[i]) {
            if (shape1[i] == 1) {
                vec0_shape.strides[i] = 0;
                vec0_shape.shape[i] = shape2[i];

            } else if (shape2[i] == 1) {
                vec1_shape.strides[i] = 0;
                vec1_shape.shape[i] = shape1[i];
            }
        }
    }

    vec0_shape.recompute();
    vec1_shape.recompute();

    TensorShape s0 = vec0_shape;
    TensorShape s1 = vec1_shape;

    for (i = 0; i < numel; i += 1) {
        op.call_base(p1[s0.d_ptr], p2[s1.d_ptr], p3[i]);
        s0.next();
        s1.next();
    }
    s0.reset();
    s1.reset();
    // }
}
template <typename... Ts, typename... TensorPack, typename Op>
void launch_binary_elementwise_scalar(Op op, const TensorPack &... args) {
    std::vector<Tensor> vec = {args...};
    // using T = {Ts...}[0];
    int numel = vec[0].numel();
    int jump = vec[0].info.jump;
    int i = 0;
    bool omp = numel >= OMP_MIN_VALUE;
    get<0, Ts...> __restrict__ *p1;
    get<1, Ts...> __restrict__ *p2;
    get<2, Ts...> __restrict__ *p3;

    p1 = static_cast<decltype(p1)>(vec[0].get_data());
    p2 = static_cast<decltype(p2)>(vec[1].get_data());
    p3 = static_cast<decltype(p3)>(vec[2].get_data());

    if (omp) {
#pragma omp parallel for
        for (i = 0; i < numel; i += 1) {
            op.call_base(p1[i], p2[0], p3[i]);
            // small_idx = small_shape.next();
        }

    } else {
        for (i = 0; i < numel; i += 1) {
            op.call_base(p1[i], p2[0], p3[i]);
            // small_idx = small_shape.next();
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

    p1 = static_cast<decltype(p1)>(vec[0].get_data());
    p2 = static_cast<decltype(p2)>(vec[1].get_data());
    p3 = static_cast<decltype(p3)>(vec[2].get_data());

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
void BinaryElementwiseScalar(Op op, TensorPack &... args) {
    bool allows_avx = false;
    static_assert(sizeof...(Ts) == sizeof...(args),
                  "Data types must be specified for each Tensor. ");

    // // get dtype to cast to

#ifdef USE_AVX2

    allows_avx = true;  // allow_avx(std::forward<TensorPack>(args)...);
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

template <typename... Ts, typename Op>
void BinaryElementwise(Op op, bool broadcast, const Tensor &t1,
                       const Tensor &t2, const Tensor &t3) {
    bool allows_avx = false;

    // if (t1.view || t2.view) {
    //     inner_elementwise::launch_binary_elementwise<Ts...>(op, t1, t2, t3);
    //     return;
    // }
    // static_assert(sizeof...(Ts) == sizeof...(args),
    //               "Data types must be specified for each Tensor. ");

    // // get dtype to cast to
    // if (broadcast) {
    //     inner_elementwise::launch_binary_elementwise_broadcast<Ts...>(op,
    //                                                                   args...);
    //     return;
    // }

#ifdef USE_AVX2

    allows_avx = true;  // allow_avx(std::forward<TensorPack>(args)...);
    // std::cout << allows_avx << std::endl;
    if (allows_avx) {
        inner_elementwise::launch_binary_elementwise_avx<Ts...>(op, t1, t2, t3);
        return;
    }
// #else
#endif
    inner_elementwise::launch_binary_elementwise<Ts...>(op, t1, t2, t3);
}

}  // namespace sail