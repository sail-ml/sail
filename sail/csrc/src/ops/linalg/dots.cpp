#pragma once

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <tuple>

#include "../../Tensor.h"
#include "../../dtypes.h"
#include "../../error.h"
#include "../../kernels/kernel.h"

namespace sail {

namespace ops {

std::tuple<LongVec, TensorShape> GetTensorDotRollAxes(
    const TensorShape& shape, const LongVec& reduce_axes,
    bool reduced_axes_first) {
    bool to_reduce[25]{};  // Initialized with false.
    LongVec remain_dims;
    LongVec roll_axes;
    for (int8_t i = 0; i < reduce_axes.size(); ++i) {
        to_reduce[reduce_axes[i]] = true;
        // gsl::at(to_reduce, reduce_axes[i]) = true;
    }

    // There are two steps:
    // A. Insert axes to be reduced to roll_axes.
    // B. Insert non-reduced axes to roll_axes.
    // The order of these steps depends on reduced_axes_first.
    for (int step = 0; step < 2; ++step) {
        if ((step == 0) == reduced_axes_first) {
            // Step A.
            for (int8_t i = 0; i < shape.ndim(); ++i) {
                if (to_reduce[i]) {  // gsl::at(to_reduce, i)) {
                    roll_axes.emplace_back(i);
                }
            }
        } else {
            // Step B.
            for (int8_t i = 0; i < shape.ndim(); ++i) {
                if (!to_reduce[i]) {
                    roll_axes.emplace_back(i);
                    remain_dims.emplace_back(shape.shape[i]);
                }
            }
        }
    }
    return std::make_tuple(roll_axes, TensorShape(remain_dims));
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
    const LongVec& a_roll_axes = std::get<0>(a_tup);
    const LongVec& b_roll_axes = std::get<0>(b_tup);
    const TensorShape& a_remain_dims = std::get<1>(a_tup);
    const TensorShape& b_remain_dims = std::get<1>(b_tup);
    int64_t a_remain_total_size = a_remain_dims.numel();
    int64_t b_remain_total_size = b_remain_dims.numel();
    TensorShape a_shape = TensorShape({a_remain_total_size, axis_total_size});
    TensorShape b_shape = TensorShape({axis_total_size, b_remain_total_size});

    if (a_shape.numel() != b_shape.numel()) {
        throw SailCError("Shape mismatch for tensordot");
    }

    TensorShape dot_shape =
        TensorShape({a_remain_total_size, b_remain_total_size});
    Tensor dot_out = empty(dot_shape.ndim(), t1.get_dtype(), dot_shape);

    std::cout << a_shape.get_string() << std::endl;
    std::cout << b_shape.get_string() << std::endl;

    DotTTKernel().execute(t1.transpose(a_roll_axes).reshape(a_shape),
                          t2.transpose(b_roll_axes).reshape(b_shape), dot_out);
    std::cout << "execute success" << std::endl;
    LongVec out_shape = a_remain_dims.shape;
    std::copy(b_remain_dims.shape.begin(), b_remain_dims.shape.end(),
              std::back_inserter(out_shape));
    TensorShape ret_shape = TensorShape(out_shape);
    return dot_out.reshape(ret_shape);
}

Tensor matmul(const Tensor& t1, const Tensor& t2) {
    Tensor casted;
    bool cast;
    // NEED TO CHECK NDIM, TYPE, AND SHAPES SO THAT IT WORKS
    // ALSO NO SCALARS

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
        throw SailCError("Matmul currently does not support views");
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

    Tensor empty_tensor =
        empty(t1.get_ndim(), t1.get_dtype(), TensorShape(new_shape));

    DotTTKernel().execute(t1, t2, empty_tensor);

    return empty_tensor;
}
}  // namespace ops

}  // namespace sail
