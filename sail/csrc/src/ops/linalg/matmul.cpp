#pragma once

#include <iostream>

#include "../../Tensor.h"
#include "../../dtypes.h"
#include "../../error.h"
#include "../../kernels/kernel.h"

namespace sail {

namespace ops {

Tensor matmul(Tensor& t1, Tensor& t2) {
    Tensor empty_tensor;
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

    if (t1.get_dtype() != t2.get_dtype()) {
        cast = true;
        casted = t2.cast(t1.get_dtype());
    } else {
        casted = t2;
    }

    TensorSize new_shape;
    new_shape.push_back(t1.get_shape().shape[0]);
    new_shape.push_back(t2.get_shape().shape[1]);

    TensorSize new_strides;
    long dt_size = GetDtypeSize(t1.get_dtype());
    for (long s : new_shape) {
        new_strides.push_back(dt_size * s);
    }
    new_strides.pop_back();
    new_strides.push_back(dt_size);

    empty_tensor = empty(t1.get_ndim(), t1.get_dtype(),
                         TensorShape(new_shape, new_strides));

    MatmulTTKernel().execute(t1, t2, empty_tensor);

    if (cast) {
        casted.free();
    }

    return empty_tensor;
}
}  // namespace ops

}  // namespace sail
