#pragma once

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <tuple>

#include "../../Tensor.h"
#include "../../autograd/autograd.h"
#include "../../dtypes.h"
#include "../../error.h"
#include "../../kernels/kernel.h"

namespace sail {

namespace ops {
using TensorVector = std::vector<Tensor>;

std::tuple<LongVec, TensorShape> GetTensorDotRollAxes(
    const TensorShape& shape, const LongVec& reduce_axes,
    bool reduced_axes_first) {
    bool to_reduce[25]{};  // Initialized with false.
    LongVec remain_dims;
    LongVec remain_dims2;
    LongVec roll_axes;
    LongVec roll_axes2;
    LongVec not_in;
    for (int8_t i = 0; i < reduce_axes.size(); ++i) {
        to_reduce[reduce_axes[i]] = true;
        // gsl::at(to_reduce, reduce_axes[i]) = true;
    }

    for (int i = 0; i < shape.ndim(); i++) {
        if (std::find(reduce_axes.begin(), reduce_axes.end(), i) ==
            reduce_axes.end()) {
            not_in.push_back(i);
            remain_dims2.emplace_back(shape.shape[i]);
        }
    }

    if (reduced_axes_first) {
        roll_axes2 = reduce_axes;
        roll_axes2.insert(roll_axes2.end(), not_in.begin(), not_in.end());
    } else {
        roll_axes2 = not_in;
        roll_axes2.insert(roll_axes2.end(), reduce_axes.begin(),
                          reduce_axes.end());
    }

    return std::make_tuple(roll_axes2, TensorShape(remain_dims2));
}
Tensor tensordot(const Tensor& t1, const Tensor& t2, LongVec t1_dim,
                 LongVec t2_dim) {
    if (t1_dim.size() != t2_dim.size()) {
        throw SailCError("Size of axes must match");
    }
    int64_t axis_total_size = 1;
    int axis_ndim = t1_dim.size();
    for (int8_t i = 0; i < axis_ndim; ++i) {
        int64_t a_dim = t1.get_shape().shape[t1_dim[i]];
        axis_total_size *= a_dim;
    }

    auto a_tup = GetTensorDotRollAxes(t1.get_shape(), t1_dim, false);
    auto b_tup = GetTensorDotRollAxes(t2.get_shape(), t2_dim, true);
    LongVec a_roll_axes = std::get<0>(a_tup);
    LongVec b_roll_axes = std::get<0>(b_tup);
    TensorShape a_remain_dims = std::get<1>(a_tup);
    TensorShape b_remain_dims = std::get<1>(b_tup);
    int64_t a_remain_total_size = a_remain_dims.numel();
    int64_t b_remain_total_size = b_remain_dims.numel();
    TensorShape a_shape = TensorShape({a_remain_total_size, axis_total_size});
    TensorShape b_shape = TensorShape({axis_total_size, b_remain_total_size});

    // if (a_shape.numel() != b_shape.numel()) {
    //     throw SailCError("Shape mismatch for tensordot");
    // }

    TensorShape dot_shape =
        TensorShape({a_remain_total_size, b_remain_total_size});
    Tensor dot_out = empty(dot_shape.ndim(), t1.get_dtype(), dot_shape);

    DotTTKernel().execute(t1.transpose(a_roll_axes).reshape(a_shape),
                          t2.transpose(b_roll_axes).reshape(b_shape), dot_out);
    LongVec out_shape = a_remain_dims.shape;
    std::copy(b_remain_dims.shape.begin(), b_remain_dims.shape.end(),
              std::back_inserter(out_shape));
    TensorShape ret_shape = TensorShape(out_shape);
    // ret_shape.reset();
    return dot_out._inplace_reshape(ret_shape);
}

Tensor matmul(const Tensor& t1, const Tensor& t2) {
    Tensor casted;
    bool cast;
    // NEED TO CHECK NDIM, TYPE, AND SHAPES SO THAT IT WORKS
    // ALSO NO SCALARS

    if (t1.requires_grad || t2.requires_grad) {
        TensorVector vec;
        vec.emplace_back(t1);
        vec.emplace_back(t2);
        Tensor empty_tensor = (new autograd::Matmul())->apply(vec);
        return empty_tensor;
    }

    if (t1.is_scalar() || t2.is_scalar()) {
        throw SailCError("Cannot pass scalars to matmul");
    }

    if (t1.get_ndim() != t2.get_ndim()) {
        throw SailCError("Number of dimensions must match");
    }

    if (t1.get_shape().shape[1] != t2.get_shape().shape[0]) {
        throw SailCError("Inner dimensions must match");
    }

    if (t1.is_view() || t2.is_view()) {
        // TODO(tgs266): This needs to not use clone. Instead, switch to
        // tensorshape iterator in kernel
        t1 = clone(t1);
        t2 = clone(t2);
    }

    if (t1.get_dtype() != t2.get_dtype()) {
        cast = true;
        casted = t2.cast(t1.get_dtype());
    } else {
        casted = t2;
    }

    TensorSize new_shape;
    new_shape.push_back(t1.get_shape().shape[0]);
    new_shape.push_back(t2.get_shape().shape[1]);
    TensorShape s = TensorShape(new_shape);
    Tensor empty_tensor = empty(t1.get_ndim(), t1.get_dtype(), s);

    DotTTKernel().execute(t1, t2, empty_tensor);

    return empty_tensor;
}

Tensor addmm(const Tensor& t1, const Tensor& t2, const Tensor& add) {
    Tensor casted;
    bool cast;
    // NEED TO CHECK NDIM, TYPE, AND SHAPES SO THAT IT WORKS
    // ALSO NO SCALARS

    if (t1.requires_grad || t2.requires_grad) {
        TensorVector vec;
        vec.emplace_back(t1);
        vec.emplace_back(t2);
        Tensor empty_tensor = (new autograd::Matmul())->apply(vec);
        return empty_tensor;
    }

    if (t1.is_scalar() || t2.is_scalar()) {
        throw SailCError("Cannot pass scalars to matmul");
    }

    if (t1.get_ndim() != t2.get_ndim()) {
        throw SailCError("Number of dimensions must match");
    }

    if (t1.get_shape().shape[1] != t2.get_shape().shape[0]) {
        throw SailCError("Inner dimensions must match");
    }

    if (t1.is_view() || t2.is_view()) {
        t1 = clone(t1);
        t2 = clone(t2);
    }

    // if (t1.get_dtype() != t2.get_dtype()) {
    //     cast = true;
    //     casted = t2.cast(t1.get_dtype());
    // } else {
    //     casted = t2;
    // }

    TensorSize new_shape;
    new_shape.push_back(t1.get_shape().shape[0]);
    new_shape.push_back(t2.get_shape().shape[1]);
    TensorShape s = TensorShape(new_shape);
    Tensor broadcasted_add = ops::broadcast_to(add, s);
    Tensor empty_tensor = clone(broadcasted_add);

    DotTTKernel().execute(t1, t2, empty_tensor);

    return empty_tensor;
}
}  // namespace ops

}  // namespace sail
