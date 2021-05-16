#pragma once

#include <iostream>

#include "../../Tensor.h"
#include "../../dtypes.h"
#include "../../error.h"
#include "../../kernels/kernel.h"

namespace sail {

namespace ops {

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

    MatmulTTKernel().execute(t1, t2, empty_tensor);

    return empty_tensor;
}
}  // namespace ops

}  // namespace sail
