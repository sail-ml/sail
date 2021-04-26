
#pragma once

#include "factories.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <iostream>
#include <vector>

#include "Tensor.h"
#include "Tensor_storage.h"
#include "dtypes.h"
#include "error.h"
#include "types.h"
#include "utils.h"

namespace sail {

Tensor empty(int ndims, Dtype dt, TensorSize strides, TensorSize shape) {
    auto size = GetDtypeSize(dt);
    for (py::ssize_t value : shape) {
        size = size * value;
    }
    alignemnt_information info = getAlignment(dt);
    void* data = _malloc_align(size, info.alignment, info.dtype_size);

    TensorStorage store =
        TensorStorage::createEmpty(ndims, data, 0, dt, strides, shape, info);

    return Tensor(store);
}
Tensor copy(Tensor t) {
    auto size = GetDtypeSize(t.storage.dtype);
    for (py::ssize_t value : t.storage.shape) {
        size = size * value;
    }
    alignemnt_information info = getAlignment(t.storage.dtype);
    void* data =
        _realloc_align(t.storage.data, size, info.alignment, info.dtype_size);

    TensorStorage store =
        TensorStorage::createEmpty(t.storage.ndim, data, 0, t.storage.dtype,
                                   t.storage.strides, t.storage.shape, info);

    return Tensor(store);
}

Tensor empty_scalar(Dtype dt) {
    alignemnt_information info = getAlignment(dt);
    void* data = _malloc_align(1, info.alignment, info.dtype_size);
    TensorStorage store =
        TensorStorage::createEmpty(0, data, 0, dt, {info.dtype_size}, {}, info);
    return Tensor(store);
}

}  // namespace sail