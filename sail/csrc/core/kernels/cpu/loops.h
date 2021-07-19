// allow-no-source

#pragma once

#include <immintrin.h>
#include <omp.h>
#include <algorithm>
#include <cassert>
#include <vector>
#include "Tensor.h"
#include "dtypes.h"
#include "error.h"
#include "kernels/kernel_utils.h"
#include "kernels/native/loops.h"
#include "tensor_iterator.h"
#include "tensor_shape.h"
#include "xsimd/xsimd.hpp"

using Tensor = sail::Tensor;
template <std::size_t N, typename... Args>
using get = typename get_Nth_type<N, Args...>::type;

namespace sail {

namespace cpu {

namespace inner_elementwise {

template <typename T, typename Op>
void launch_binary_elementwise(Op op, const Tensor &t1, const Tensor &t2,
                               const Tensor &out) {
    int i = 0;
    int jump = op.size;

    T *p1;
    T *p2;
    T *p3;

    p1 = static_cast<T *>(t1.get_data());
    p2 = static_cast<T *>(t2.get_data());
    p3 = static_cast<T *>(out.get_data());

    TensorShape s1 = t1.get_shape();
    TensorShape s2 = t2.get_shape();

    MultiTensorIterator iter = MultiTensorIterator(s1).add_input(s2);

    int inner_loop_size = iter.inner_loop_size();
    int inner_steps = inner_loop_size / jump;
    int outer_steps = iter.out_loop_size();

    int z = 0;
    for (int i = 0; i < outer_steps; i++) {
        for (int j = 0; j < inner_loop_size; j += 1) {
            op.call_base(p1[iter.d_ptrs[0]], p2[iter.d_ptrs[1]], p3[z]);
            iter.advance_d_ptr(1);
            z += 1;
        }
        iter.backup_d_ptr();
        iter.next();
    }
}
template <typename T, typename Op>
void launch_binary_elementwise_avx(Op op, const Tensor &t1, const Tensor &t2,
                                   const Tensor &out) {
    int i = 0;
    const int jump = op.size;

    T *p1;
    T *p2;
    T *p3;

    p1 = static_cast<T *>(t1.get_data());
    p2 = static_cast<T *>(t2.get_data());
    p3 = static_cast<T *>(out.get_data());

    TensorShape s1 = t1.get_shape();
    TensorShape s2 = t2.get_shape();

    MultiTensorIterator test = MultiTensorIterator(s1).add_input(s2);

    bool scalar_0 = (test.strides.at_back(0) == 0) ? true : false;
    bool scalar_1 = (test.strides.at_back(1) == 0) ? true : false;

    int inner_loop_size = test.inner_loop_size();
    int inner_steps = inner_loop_size / jump;
    int outer_steps = test.out_loop_size();

    int z = 0;
    for (int i = 0; i < outer_steps; i++) {
        int inner = 0;
        int j = 0;
        if (scalar_0 && !scalar_1) {
            op.set_scalar_val(p1[test.d_ptrs[0]]);
        } else if (!scalar_0 && scalar_1) {
            op.set_scalar_val(p2[test.d_ptrs[1]]);
        } else if (scalar_0 && scalar_1) {
            T v;
            op.call_base(p1[test.d_ptrs[0]], p2[test.d_ptrs[1]], v);
            op.set_scalar_val(v);
            for (; j < inner_steps; j += 1) {
                op.store_scal(p3 + z);
                z += jump;
                inner += jump;
                test.advance_d_ptr(jump);
            }
        }
        for (; j < inner_steps; j += 1) {
            if (scalar_0 && !scalar_1) {
                op.iterator_avx_first(p2 + test.d_ptrs[1], p3 + z);
            } else if (!scalar_0 && scalar_1) {
                op.iterator_avx_second(p1 + test.d_ptrs[0], p3 + z);
            } else {
                op.call_avx_non_aligned(p1 + test.d_ptrs[0],
                                        p2 + test.d_ptrs[1], p3 + z);
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
}

template <typename T, typename Op>
void launch_binary_elementwise_avx_contiguous(Op op, const Tensor &t1,
                                              const Tensor &t2,
                                              const Tensor &out) {
    int numel = t1.get_shape().numel();
    int jump = op.size;
    int i = 0;

    T *p1;
    T *p2;
    T *p3;

    p1 = static_cast<T *>(t1.get_data());
    p2 = static_cast<T *>(t2.get_data());
    p3 = static_cast<T *>(out.get_data());

    bool aligned = true;

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

template <typename T, typename Op>
void launch_unary_elementwise_avx_contiguous(Op op, const Tensor &t1,
                                             const Tensor &out) {
    int numel = t1.get_shape().numel();
    int jump = op.size;
    int i = 0;

    T *p1;
    T *p2;

    p1 = static_cast<T *>(t1.get_data());
    p2 = static_cast<T *>(out.get_data());

    bool aligned = true;

    if (aligned) {
        for (int i = 0; i < numel; i += jump) {
            op.call_avx_aligned(p1 + i, p2 + i);
        }
    } else {
        for (int i = 0; i < numel; i += jump) {
            op.call_avx_non_aligned(p1 + i, p2 + i);
        }
    }
}

template <typename T, typename Op>
void launch_unary_elementwise_avx(Op op, const Tensor &t1, const Tensor &out) {
    int numel = t1.get_shape().numel();
    TensorShape s = t1.get_shape();
    int jump = op.size;
    int i = 0;

    T *p1;
    T *p2;

    p1 = static_cast<T *>(t1.get_data());
    p2 = static_cast<T *>(out.get_data());

    MultiTensorIterator iter = MultiTensorIterator(s);
    int inner_loop_size = iter.inner_loop_size();
    int inner_steps = inner_loop_size / jump;
    int outer_steps = iter.out_loop_size();

    bool scalar = (iter.strides.at_back(0) == 0) ? true : false;
    if (scalar) {
        int z = 0;
        for (int i = 0; i < outer_steps; i++) {
            int inner = 0;
            op.set_scalar_val(p1[iter.d_ptrs[0]]);
            for (int j = 0; j < inner_steps; j += 1) {
                op.call_avx_scalar(p2 + z);
                z += jump;
                inner += jump;
                iter.advance_d_ptr(jump);
            }
            for (; inner < inner_loop_size; inner++) {
                op.call_base(p1[iter.d_ptrs[0]], p2[z]);
                iter.advance_d_ptr(1);
                z += 1;
            }
            iter.backup_d_ptr();
            iter.next();
        }
    } else if (iter.contiguous_at(0)) {
        int z = 0;
        for (int i = 0; i < outer_steps; i++) {
            int inner = 0;
            for (int j = 0; j < inner_steps; j += 1) {
                op.call_avx_non_aligned(p1 + iter.d_ptrs[0], p2 + z);
                z += jump;
                inner += jump;
                iter.advance_d_ptr(jump);
            }
            for (; inner < inner_loop_size; inner++) {
                op.call_base(p1[iter.d_ptrs[0]], p2[z]);
                iter.advance_d_ptr(1);
                z += 1;
            }
            iter.backup_d_ptr();
            iter.next();
        }
    } else {
        int z = 0;
        for (int i = 0; i < outer_steps; i++) {
            for (int j = 0; j < inner_loop_size; j += 1) {
                op.call_base(p1[iter.d_ptrs[0]], p2[z]);
                iter.advance_d_ptr(1);
                z += 1;
            }
            iter.backup_d_ptr();
            iter.next();
        }
    }
}

}  // namespace inner_elementwise

template <typename T, typename Op>
void BinaryElementwise(Op op, bool broadcast, const Tensor &t1,
                       const Tensor &t2, const Tensor &t3,
                       bool allow_avx = true) {
#ifdef USE_AVX

    if (broadcast) {
        if (allow_avx) {
            inner_elementwise::launch_binary_elementwise_avx<T>(op, t1, t2, t3);
            return;
        }
        inner_elementwise::launch_binary_elementwise<T>(op, t1, t2, t3);
        return;
    }

    inner_elementwise::launch_binary_elementwise_avx_contiguous<T>(op, t1, t2,
                                                                   t3);
    return;
#endif
    inner_elementwise::launch_binary_elementwise<T>(op, t1, t2, t3);
}

template <typename T, typename Op>
void UnaryElementwise(Op op, bool view, const Tensor &t1, Tensor &t2,
                      bool allow_avx = true) {
#ifdef USE_AVX

    if (view) {
        if (allow_avx) {
            inner_elementwise::launch_unary_elementwise_avx<T>(op, t1, t2);
            return;
        }
        sail::internal::native::inner_elementwise::launch_unary_elementwise<T>(
            op, t1, t2);
        return;
    }

    inner_elementwise::launch_unary_elementwise_avx_contiguous<T>(op, t1, t2);
    return;
#endif
    sail::internal::native::inner_elementwise::launch_unary_elementwise<T>(
        op, t1, t2);
}

}  // namespace cpu
}  // namespace sail