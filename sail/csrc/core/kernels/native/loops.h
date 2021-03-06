// allow-no-source

#pragma once

#include <immintrin.h>
#include <omp.h>
#include <algorithm>
#include <cassert>  // needed for xsimd
#include <vector>
#include "Tensor.h"
#include "dtypes.h"
#include "kernels/kernel_utils.h"
#include "tensor_iterator.h"
#include "tensor_shape.h"
#include "xsimd/xsimd.hpp"

using Tensor = sail::Tensor;

namespace sail {

namespace internal {

namespace native {

namespace inner_elementwise {

template <typename T, typename Op>
void launch_binary_elementwise(Op op, const Tensor &t1, const Tensor &t2,
                               const Tensor &out) {
    int i = 0;
    const int jump = t1.get_info().jump;

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
void launch_binary_elementwise_inplace(Op op, Tensor &t1, Tensor &t2) {
    int i = 0;
    const int jump = t1.get_info().jump;

    T *p1;
    T *p2;

    p1 = static_cast<T *>(t1.get_data());
    p2 = static_cast<T *>(t2.get_data());

    TensorShape s1 = t1.get_shape();
    TensorShape s2 = t2.get_shape();

    MultiTensorIterator iter = MultiTensorIterator(s1).add_input(s2);

    int inner_loop_size = iter.inner_loop_size();
    int outer_steps = iter.out_loop_size();

    for (int i = 0; i < outer_steps; i++) {
        for (int j = 0; j < inner_loop_size; j += 1) {
            op.call_base(p1[iter.d_ptrs[0]], p2[iter.d_ptrs[1]]);
            iter.advance_d_ptr(1);
        }
        iter.backup_d_ptr();
        iter.next();
    }
}
template <typename T, typename Op>
void launch_unary_elementwise(Op op, const Tensor &t1, const Tensor &out) {
    int numel = t1.get_shape().numel();
    int jump = t1.get_info().jump;
    int i = 0;

    T *p1;
    T *p2;

    p1 = static_cast<T *>(t1.get_data());
    p2 = static_cast<T *>(out.get_data());

    if (t1.is_view()) {
        TensorShape s = t1.get_shape();
        MultiTensorIterator iter = MultiTensorIterator(s);
        int inner_loop_size = iter.inner_loop_size();
        int outer_steps = iter.out_loop_size();

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
    } else {
        for (i = 0; i < numel; i += 1) {
            op.call_base(p1[i], p2[i]);
        }
    }
}
}  // namespace inner_elementwise

template <typename T, typename Op>
void BinaryElementwiseInPlace(Op op, Tensor &t1, Tensor &t2) {
    inner_elementwise::launch_binary_elementwise_inplace<T>(op, t1, t2);
}
template <typename T, typename Op>
void BinaryElementwise(Op op, bool broadcast, const Tensor &t1,
                       const Tensor &t2, const Tensor &t3) {
    inner_elementwise::launch_binary_elementwise<T>(op, t1, t2, t3);
}
template <typename T, typename Op>
void UnaryElementwise(Op op, const Tensor &t1, const Tensor &t3) {
    inner_elementwise::launch_unary_elementwise<T>(op, t1, t3);
}

namespace inner_reduction {

template <typename T, typename Op>
void launch_reduction(Op op, const Tensor &input, const Tensor &out) {
    int numel = input.get_shape().numel();
    int jump = input.get_info().jump;
    int i = 0;

    T *p1;
    T *p2;

    p1 = static_cast<T *>(input.get_data());
    p2 = static_cast<T *>(out.get_data());

    for (i = 0; i < numel; i += 1) {
        op.call_base(p1[i], p2[0]);
    }
}

template <typename T, typename Op>
void launch_reduction_axis(Op op, const Tensor &input, const Tensor &out,
                           int axis, T initial = 0) {
    TensorShape s = TensorShape(input.get_shape());
    int numel = out.get_shape().numel();
    s.recompute();
    s.move_axis(axis, -1);

    T *p1;
    T *p2;

    p1 = static_cast<T *>(input.get_data());
    p2 = static_cast<T *>(out.get_data());

    bool init = true;
    int count = 0;
    int idx = 0;

    for (int i = 0; i < out.numel(); i++) {
        count = 0;
        p2[idx] = initial;
        while (count != s.shape[s.ndim() - 1]) {
            op.call_base(p1[s.d_ptr], p2[idx]);
            s.next();
            count += 1;
        }
        idx += 1;
        init = false;
    }
}

template <typename T, typename Op>
void launch_reduction_multi_axis(Op op, const Tensor &input, const Tensor &out,
                                 std::vector<long> axis, T initial = 0) {
    TensorShape s = TensorShape(input.get_shape());
    int numel = out.get_shape().numel();
    long stop = 1;
    s.recompute();
    std::reverse(axis.begin(), axis.end());
    for (long a : axis) {
        s.move_axis(a, -1);
        stop *= s.shape.back();
    }

    T *p1;
    T *p2;

    p1 = static_cast<T *>(input.get_data());
    p2 = static_cast<T *>(out.get_data());

    bool init = true;
    int count = 0;
    int idx = 0;

    for (int i = 0; i < out.numel(); i++) {
        count = 0;
        p2[idx] = initial;
        while (count != stop) {
            op.call_base(p1[s.d_ptr], p2[idx]);
            s.next();
            count += 1;
        }
        idx += 1;
        init = false;
    }
}

}  // namespace inner_reduction

template <typename T, typename Op>
void Reduction(Op op, const Tensor &input, const Tensor &out) {
    bool allows_avx = false;

    inner_reduction::launch_reduction<T>(op, input, out);
}
template <typename T, typename Op>
void Reduction(Op op, const Tensor &input, const Tensor &out, int index,
               T initial = 0) {
    bool allows_avx = false;

    inner_reduction::launch_reduction_axis<T>(op, input, out, index, initial);
}

template <typename T, typename Op>
void Reduction(Op op, const Tensor &input, const Tensor &out,
               std::vector<long> axes, T initial = 0) {
    bool allows_avx = false;

    inner_reduction::launch_reduction_multi_axis<T>(op, input, out, axes,
                                                    initial);
}

}  // namespace native

}  // namespace internal
}  // namespace sail