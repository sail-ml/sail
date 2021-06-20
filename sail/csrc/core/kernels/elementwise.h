#pragma once

#include <immintrin.h>
#include <omp.h>
#include <algorithm>
#include <cassert>  // needed for xsimd
#include <vector>
#include "Tensor.h"
#include "dtypes.h"
#include "kernel_utils.h"
#include "tensor_iterator.h"
#include "tensor_shape.h"
#include "utils.h"
#include "xsimd/xsimd.hpp"

#include <chrono>
using namespace std::chrono;

// Use auto keyword to avoid typing long
// type definitions to get the timepoint
// at this instant use function now()
using Tensor = sail::Tensor;

template <std::size_t N, typename... Args>
using get = typename get_Nth_type<N, Args...>::type;

namespace sail {

namespace inner_elementwise {

// <typename T, typename Op>
// void loop(Op op, T* p1, T* p2, T* p3, TensorShape t1, TensorShape t2,
// TensorShape t3) {
//     int t1_nop = t1.numel();
//     int t2_nop = t2.numel();

// }

template <typename... Ts, typename Op>
void launch_binary_elementwise(Op op, const Tensor &t1, const Tensor &t2,
                               const Tensor &out) {
    // using T = {Ts...}[0];
    int i = 0;
    const int jump = t1.get_info().jump;

    get<0, Ts...> __restrict__ *p1;
    get<1, Ts...> __restrict__ *p2;
    get<2, Ts...> __restrict__ *p3;

    p1 = static_cast<decltype(p1)>(t1.get_data());
    p2 = static_cast<decltype(p2)>(t2.get_data());
    p3 = static_cast<decltype(p3)>(out.get_data());

    TensorShape s1 = t1.get_shape();
    TensorShape s2 = t2.get_shape();
    TensorShape s3 = out.get_shape();

    // std::cout << t1.get_shape().get_string() << std::endl;
    // std::cout << t1.is_view() << std::endl;

    // TensorIterator test = TensorIterator(s1);
    // std::cout << test.out_loop_size() << std::endl;
    // std::cout << test.inner_loop_size() << std::endl;

    MultiTensorIterator test =
        MultiTensorIterator(s1).add_input(s2);  //.add_input(s3);

    bool scalar_0 = (test.strides.at_back(0) == 0) ? true : false;
    bool scalar_1 = (test.strides.at_back(1) == 0) ? true : false;
    int z = 0;
    for (int i = 0; i < test.out_loop_size(); i++) {
        for (int j = 0; j < test.inner_loop_size(); j++) {
            op.call_base(p1[test.d_ptrs[0]], p2[test.d_ptrs[1]], p3[z]);
            test.advance_d_ptr(1);
            z += 1;
        }
        test.backup_d_ptr();
        test.next();
    }
}

template <typename... Ts, typename Op>
void launch_binary_elementwise2(Op op, const Tensor &t1, const Tensor &t2,
                                const Tensor &out) {
    // using T = {Ts...}[0];
    int i = 0;
    const int jump = t1.get_info().jump;

    get<0, Ts...> __restrict__ *p1;
    get<1, Ts...> __restrict__ *p2;
    get<2, Ts...> __restrict__ *p3;

    p1 = static_cast<decltype(p1)>(t1.get_data());
    p2 = static_cast<decltype(p2)>(t2.get_data());
    p3 = static_cast<decltype(p3)>(out.get_data());

    TensorShape s1 = t1.get_shape();
    TensorShape s2 = t2.get_shape();

    MultiTensorIterator test =
        MultiTensorIterator(s1).add_input(s2);  //.add_input(s3);

    bool scalar_0 = (test.strides.at_back(0) == 0) ? true : false;
    bool scalar_1 = (test.strides.at_back(1) == 0) ? true : false;

    int inner_loop_size = test.inner_loop_size();
    int inner_steps = inner_loop_size / jump;
    int outer_steps = test.out_loop_size();
    std::vector<long> strides = test.get_strides();
    int z = 0;
    // auto start = high_resolution_clock::now();
    for (int i = 0; i < outer_steps; i++) {
        int inner = 0;
        if (scalar_0 && !scalar_1) {
            op.set_scalar_val(p1[test.d_ptrs[0]]);
        } else if (!scalar_0 && scalar_1) {
            op.set_scalar_val(p2[test.d_ptrs[1]]);
        }
        for (int j = 0; j < inner_steps; j += 1) {
            if (scalar_0 && !scalar_1) {
                op.iterator_avx(p2 + test.d_ptrs[1], p3 + z);
            } else if (!scalar_0 && scalar_1) {
                op.iterator_avx(p1 + test.d_ptrs[0], p3 + z);
            } else {
                op.call_avx_aligned(p1 + test.d_ptrs[0], p2 + test.d_ptrs[1],
                                    p3 + z);
            }
            z += jump;
            inner += jump;
            test.advance_d_ptr(jump);
        }
        for (; inner < inner_loop_size; inner++) {
            op.call_base(p1[test.d_ptrs[0]], p2[test.d_ptrs[1]], p3[z]);
            test.advance_d_ptr(1);
            z += 1;
        }
        test.backup_d_ptr();
        test.next();
    }
    // auto stop = high_resolution_clock::now();
    // auto duration = duration_cast<nanoseconds>(stop - start);

    // To get the value of duration use the count()
    // member function on the duration object
    // std::cout << duration.count() << std::endl;
}
template <typename... Ts, typename Op>
void launch_parallel_binary_elementwise(Op op, const Tensor &t1,
                                        const Tensor &t2, const Tensor &out) {
    // using T = {Ts...}[0];

    get<0, Ts...> __restrict__ *p1;
    get<1, Ts...> __restrict__ *p2;
    get<2, Ts...> __restrict__ *p3;

    p1 = static_cast<decltype(p1)>(t1.get_data());
    p2 = static_cast<decltype(p2)>(t2.get_data());
    p3 = static_cast<decltype(p3)>(out.get_data());

    TensorShape s1 = t1.get_shape();
    TensorShape s2 = t2.get_shape();
    TensorShape s3 = out.get_shape();

    s1.recompute();
    s2.recompute();
    s3.recompute();

    int numel = out.numel();
    // #pragma omp parallel for firstprivate(s1, s2, s3) num_threads(1)
    for (int i = 0; i < numel; i += 1) {
        op.call_base(p1[s1.d_ptr], p2[s2.d_ptr], p3[s3.d_ptr]);
        s1.next(1);
        // s1.next(omp_get_thread_num() + 1);
        s2.next(1);
        // s2.next(omp_get_thread_num() + 1);
        s3.next(1);
        // s3.next(omp_get_thread_num() + 1);
    }

    s1.reset();
    s2.reset();
    s3.reset();
}

template <typename... Ts, typename Op>
void launch_binary_elementwise_avx(Op op, const Tensor &t1, const Tensor &t2,
                                   const Tensor &out) {
    // std::vector<Tensor> vec = {args...};
    int numel = t1.get_shape().numel();
    int jump = t1.get_info().jump;
    int i = 0;

    get<0, Ts...> __restrict__ *p1;
    get<1, Ts...> __restrict__ *p2;
    get<2, Ts...> __restrict__ *p3;

    p1 = static_cast<decltype(p1)>(t1.get_data());
    p2 = static_cast<decltype(p2)>(t2.get_data());
    p3 = static_cast<decltype(p3)>(out.get_data());

    bool aligned = true;  // is_aligned_vec(vec);

    if (aligned) {
        for (int i = 0; i < numel; i += jump) {
            op.call_avx_aligned(p1 + i, p2 + i, p3 + i);
        }
    } else {
        for (int i = 0; i < numel; i += jump) {
            op.call_avx_non_aligned(p1 + i, p2 + i, p3 + i);
        }
    }
}

template <typename... Ts, typename... TensorPack, typename Op>
void launch_binary_elementwise_scalar(Op op, const TensorPack &... args) {
    std::vector<Tensor> vec = {args...};
    // using T = {Ts...}[0];
    int numel = vec[0].numel();
    int jump = vec[0].get_info().jump;
    int i = 0;

    get<0, Ts...> __restrict__ *p1;
    get<1, Ts...> __restrict__ *p2;
    get<2, Ts...> __restrict__ *p3;

    p1 = static_cast<decltype(p1)>(vec[0].get_data());
    p2 = static_cast<decltype(p2)>(vec[1].get_data());
    p3 = static_cast<decltype(p3)>(vec[2].get_data());

    for (i = 0; i < numel; i += 1) {
        op.call_base(p1[i], p2[0], p3[i]);
        // small_idx = small_shape.next();
    }
}

template <typename... Ts, typename... TensorPack, typename Op>
void launch_binary_elementwise_avx_scalar(Op op, const TensorPack &... args) {
    std::vector<Tensor> vec = {args...};
    // using T = {Ts...}[0];
    int numel = vec[0].numel();
    bool aligned = true;  // is_aligned_vec(vec);
    int jump = vec[0].get_info().jump;
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
                       const Tensor &t2, const Tensor &t3, bool test = false) {
    bool allows_avx = false;

    // if (t1.is_view() || t2.is_view()) {
    //     inner_elementwise::launch_binary_elementwise<Ts...>(op, t1, t2, t3);
    //     return;
    // }
    // static_assert(sizeof...(Ts) == sizeof...(args),
    //               "Data types must be specified for each Tensor. ");
    if (test == true) {
        inner_elementwise::launch_binary_elementwise2<Ts...>(op, t1, t2, t3);
        return;
    }

    // // get dtype to cast to
    if (broadcast) {
        inner_elementwise::launch_binary_elementwise<Ts...>(op, t1, t2, t3);
        return;
    }

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

template <typename... Ts, typename Op>
void BinaryElementwiseNoAvx(Op op, bool parallel, const Tensor &t1,
                            const Tensor &t2, const Tensor &t3) {
    // if (broad) {
    //     inner_elementwise::launch_parallel_binary_elementwise<Ts...>(op, t1,
    //     t2,
    //                                                                  t3);
    //     return;
    // }

    inner_elementwise::launch_binary_elementwise<Ts...>(op, t1, t2, t3);
}

}  // namespace sail