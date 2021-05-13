
#pragma once

#include "factories.h"

#include <iostream>
#include <vector>

#include "Tensor.h"
#include "TensorBody.h"
#include "dtypes.h"
#include "error.h"
#include "tensor_shape.h"
#include "types.h"
#include "utils.h"

namespace sail {

Tensor empty(int ndims, Dtype dt, TensorShape shape) {
    std::cout << "EMPTY SHAPE " << shape.get_string() << std::endl;
    TensorBody::pointer body =
        TensorBody::pointer(new TensorBody(dt, shape), true);

    Tensor _empty = Tensor(body, false);

    return _empty;
}

Tensor clone(Tensor& t) {
    auto size = t.get_shape().getTotalSize(GetDtypeSize(t.get_dtype()));
    std::cout << "ALLOC SIZE " << size << std::endl;
    void* data;
    TensorShape s = t.get_shape();
    alignemnt_information info = getAlignment(t.get_dtype());
    if (t.is_view()) {
        data = _malloc_align(s.numel(), info.alignment, info.dtype_size);
        launch_arithmetic(t.get_dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            T* base_data = (T*)(t.get_data());
            T* set_data = (T*)data;
            for (int i = 0; i < s.numel(); i++) {
                set_data[i] = base_data[s.d_ptr];
                s.next();
            }
            s.reset();
            s = TensorShape(s.shape);
        });
    } else {
        data = _realloc_align(t.get_data(), t.numel(), info.alignment,
                              info.dtype_size);
    }

    TensorBody::pointer body = new TensorBody(data, t.get_dtype(), s);

    Tensor _empty = Tensor(body, t.requires_grad);
    return _empty;
}

Tensor make_view(int ndims, void* data, Dtype dt, TensorShape shape) {
    TensorBody::pointer b =
        TensorBody::pointer((new TensorBody(data, dt, shape, true)));
    Tensor _empty = Tensor(b, false);
    return _empty;
}

// Tensor copy(Tensor t) {
//     auto size = t.get_shape().getTotalSize(GetDtypeSize(t.get_dtype()));

//     alignemnt_information info = getAlignment(t.get_dtype());
//     void* data = _realloc_align(t.get_data(), t.numel(), info.alignment,
//                                 info.dtype_size);

//     Tensor _empty = Tensor(t.ndim, data, t.get_dtype(), t.get_shape());

//     return _empty;
// }

Tensor empty_scalar(Dtype dt) {
    alignemnt_information info = getAlignment(dt);
    TensorSize shape = {};
    TensorShape ts = TensorShape(shape);
    TensorBody::pointer b = new TensorBody(dt, ts);
    Tensor _empty = Tensor(b, false);

    return _empty;
}

Tensor one_scalar(Dtype dt) {
    alignemnt_information info = getAlignment(dt);
    std::cout << "one scalar" << std::endl;
    void* data = _malloc_align(1, info.alignment, info.dtype_size);
    launch_arithmetic(dt, [&](auto pt) {
        using T = typename decltype(pt)::type;
        // T data2 = (T)1;
        // memcpy(data, (void*)(&data2), sizeof(T));
        T x = (T)1;
        *(T*)data = x;
    });
    // memset(data, 1, info.get_dtype()_size);
    int zero = 0;
    TensorSize shape = {1};
    TensorShape ts = TensorShape(shape);
    TensorBody::pointer b = new TensorBody(data, dt, ts);

    Tensor _empty = Tensor(b, false);

    return _empty;
}

Tensor from_data(void* data, Dtype dt, TensorShape s) {
    alignemnt_information info = getAlignment(dt);
    void* new_data =
        _realloc_align(data, s.numel(), info.alignment, info.dtype_size);
    TensorBody::pointer b = new TensorBody(new_data, dt, s);
    return Tensor(b, false);
}

}  // namespace sail