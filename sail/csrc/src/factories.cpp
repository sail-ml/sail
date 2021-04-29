
#pragma once

#include "factories.h"

#include <iostream>
#include <vector>

#include "Tensor.h"
#include "dtypes.h"
#include "error.h"
#include "types.h"
#include "utils.h"

namespace sail {

Tensor empty(int ndims, Dtype dt, TensorSize strides, TensorSize shape) {
    auto size = GetDtypeSize(dt);
    for (long value : shape) {
        size = size * value;
    }
    alignemnt_information info = getAlignment(dt);
    void* data = _malloc_align(size, info.alignment, info.dtype_size);

    // TensorSize new_strides;
    // size_t dt_size = GetDtypeSize(dt);
    // for (size_t s : shape) {
    //     new_strides.push_back(dt_size * s);
    // }
    // new_strides.pop_back();
    // new_strides.push_back(dt_size);

    // std::cout << "NEW STRIDES " << getVectorString(new_strides) << std::endl;

    Tensor _empty = Tensor::move(ndims, data, dt, strides, shape);

    return _empty;
}
Tensor copy(Tensor t) {
    auto size = GetDtypeSize(t.dtype);
    for (long value : t.shape) {
        size = size * value;
    }
    alignemnt_information info = getAlignment(t.dtype);
    void* data = _realloc_align(t.data, size, info.alignment, info.dtype_size);

    Tensor _empty = Tensor(t.ndim, data, t.dtype, t.strides, t.shape);

    return _empty;
    return Tensor();
}

Tensor empty_scalar(Dtype dt) {
    alignemnt_information info = getAlignment(dt);
    void* data = _malloc_align(1, info.alignment, info.dtype_size);
    int zero = 0;
    TensorSize strides = {info.dtype_size};
    TensorSize shape = {};
    Tensor _empty = Tensor(zero, data, dt, strides, shape);

    return _empty;
}

}  // namespace sail