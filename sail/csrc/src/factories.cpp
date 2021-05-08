
#pragma once

#include "factories.h"

#include <iostream>
#include <vector>

#include "Tensor.h"
#include "dtypes.h"
#include "error.h"
#include "tensor_shape.h"
#include "types.h"
#include "utils.h"

namespace sail {

Tensor empty(int ndims, Dtype dt, TensorShape shape) {
    auto size = shape.getTotalSize(GetDtypeSize(dt));

    alignemnt_information info = getAlignment(dt);
    void* data = _malloc_align(size, info.alignment, info.dtype_size);

    Tensor _empty = Tensor::move(ndims, data, dt, shape);

    return _empty;
}
Tensor copy(Tensor t) {
    auto size = t.shape_details.getTotalSize(GetDtypeSize(t.dtype));

    alignemnt_information info = getAlignment(t.dtype);
    void* data =
        _realloc_align(t.get_data(), size, info.alignment, info.dtype_size);

    Tensor _empty = Tensor(t.ndim, data, t.dtype, t.shape_details);

    return _empty;
    return Tensor();
}

Tensor empty_scalar(Dtype dt) {
    alignemnt_information info = getAlignment(dt);
    void* data = _malloc_align(1, info.alignment, info.dtype_size);
    int zero = 0;
    TensorSize shape = {};
    TensorShape ts = TensorShape(shape);
    Tensor _empty = Tensor(zero, data, dt, ts);

    return _empty;
}

Tensor one_scalar(Dtype dt) {
    alignemnt_information info = getAlignment(dt);
    void* data = _malloc_align(1, info.alignment, info.dtype_size);
    launch_arithmetic(dt, [&](auto pt) {
        using T = typename decltype(pt)::type;
        T data2 = (T)1;
        memcpy(data, (void*)(&data2), sizeof(T));
    });
    // memset(data, 1, info.dtype_size);
    int zero = 0;
    TensorSize shape = {1};
    TensorShape ts = TensorShape(shape);
    Tensor _empty = Tensor(zero, data, dt, ts);

    return _empty;
}

}  // namespace sail