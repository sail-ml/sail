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

    // std::cout << getVectorString(remain_dims2) << std::endl;
    // std::cout << getVectorString(roll_axes2) << std::endl;

    // // There are two steps:
    // // A. Insert axes to be reduced to roll_axes.
    // // B. Insert non-reduced axes to roll_axes.
    // // The order of these steps depends on reduced_axes_first.
    // for (int step = 0; step < 2; ++step) {
    //     if ((step == 0) == reduced_axes_first) {
    //         // Step A.
    //         for (int8_t i = 0; i < shape.ndim(); ++i) {
    //             if (to_reduce[i]) {  // gsl::at(to_reduce, i)) {
    //                 roll_axes.emplace_back(i);
    //             }
    //         }
    //     } else {
    //         // Step B.
    //         for (int8_t i = 0; i < shape.ndim(); ++i) {
    //             if (!to_reduce[i]) {
    //                 roll_axes.emplace_back(i);
    //                 remain_dims.emplace_back(shape.shape[i]);
    //             }
    //         }
    //     }
    // }
    // std::cout << getVectorString(remain_dims) << std::endl;
    // std::cout << getVectorString(roll_axes) << std::endl;
    // std::cout << "D" << std::endl;
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

    std::cout << a_shape.get_string() << std::endl;
    std::cout << b_shape.get_string() << std::endl;
    std::cout << getVectorString(a_roll_axes) << std::endl;
    std::cout << getVectorString(b_roll_axes) << std::endl;

    Tensor t1_a = t1.transpose(a_roll_axes);
    std::cout << "A" << std::endl;
    t1_a = t1_a.reshape(a_shape);
    std::cout << t1_a << std::endl;

    Tensor t2_b = t2.transpose(b_roll_axes);
    std::cout << "B" << std::endl;
    // std::cout << t2_b << std::endl;
    t2_b = t2_b.reshape(b_shape);
    // std::cout << t2_b << std::endl;

    DotTTKernel().execute(t1_a, t2_b, dot_out);
    std::cout << "execute success" << std::endl;
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
