
#pragma once

#include "factories.h"

#include <iostream>
#include <random>
#include <vector>

#include "Tensor.h"
#include "TensorBody.h"
#include "dtypes.h"
#include "error.h"
#include "tensor_shape.h"
#include "types.h"
#include "utils.h"

namespace sail {

std::random_device rd;
std::mt19937 gen(rd());

Tensor empty(const int ndims, const Dtype& dt, const TensorShape& shape) {
    TensorBody::pointer body =
        TensorBody::pointer(new TensorBody(dt, shape), true);

    Tensor _empty = Tensor(body, false);

    return _empty;
}

Tensor empty_like(const Tensor& tensor) {
    TensorBody::pointer body = TensorBody::pointer(
        new TensorBody(tensor.get_dtype(), tensor.get_shape()));

    Tensor _empty = Tensor(body, tensor.requires_grad);
    _empty.requires_grad = tensor.requires_grad;
    return _empty;
}

Tensor* create_grad(Dtype dt) {
    alignemnt_information info = getAlignment(dt);
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

    Tensor* _empty = new Tensor(b, false);

    return _empty;
}

Tensor clone(const Tensor& t) {
    auto size = t.get_shape().getTotalSize(GetDtypeSize(t.get_dtype()));
    void* data;
    TensorShape s = t.get_shape();
    alignemnt_information info = getAlignment(t.get_dtype());
    if (t.is_view()) {
        int numel = s.numel();
        data = _malloc_align(numel, info.alignment, info.dtype_size);
        launch_arithmetic(t.get_dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            T* base_data = (T*)(t.get_data());
            T* set_data = (T*)data;
            s.recompute();
            for (int i = 0; i < numel; i++) {
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

Tensor make_view(void* data, Dtype dt, TensorShape shape) {
    TensorBody::pointer b =
        TensorBody::pointer((new TensorBody(data, dt, shape, true)));
    Tensor _empty = Tensor(b, false);
    return _empty;
}

Tensor make_view(const Tensor& t) {
    TensorBody::pointer b = TensorBody::pointer(
        (new TensorBody(t.get_data(), t.get_dtype(), t.get_shape(), true)));
    Tensor _empty = Tensor(b, t.requires_grad);
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
Tensor zero_scalar(Dtype dt) {
    alignemnt_information info = getAlignment(dt);
    void* data = _malloc_align(1, info.alignment, info.dtype_size);
    launch_arithmetic(dt, [&](auto pt) {
        using T = typename decltype(pt)::type;
        // T data2 = (T)1;
        // memcpy(data, (void*)(&data2), sizeof(T));
        T x = (T)0;
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

Tensor zeros(TensorShape size, Dtype dt) {
    alignemnt_information info = getAlignment(dt);
    void* new_data =
        _calloc_align(size.numel(), info.alignment, info.dtype_size);
    TensorBody::pointer b = new TensorBody(new_data, dt, size);
    return Tensor(b, false);
}

namespace random {  // probably want to refactor factories to be in their own
                    // namespace but rolling with this for now

// need to be able to instantiate random tensors
Tensor uniform(TensorShape size, Dtype dt, double min = 0, double max = 1) {
    alignemnt_information info = getAlignment(dt);
    int numel = size.numel();
    void* data = _malloc_align(numel, info.alignment, info.dtype_size);
    launch_arithmetic(dt, [&](auto pt) {
        using T = typename decltype(pt)::type;

        T* data_rand = (T*)data;

        std::uniform_real_distribution<> dis((T)min, (T)max);

        for (int i = 0; i < numel; i++) {
            data_rand[i] = dis(gen);
        }
    });

    TensorBody::pointer b = new TensorBody(data, dt, size);
    return Tensor(b, false);
}
Tensor uniform(TensorShape size, double min = 0, double max = 1) {
    return uniform(size, default_dtype, min, max);
}
Tensor uniform_like(Tensor tensor, double min = 0, double max = 1) {
    TensorShape s = tensor.get_shape();
    Tensor ret = uniform(s, tensor.get_dtype(), min, max);
    ret.requires_grad = tensor.requires_grad;
    return ret;
}

Tensor normal(TensorShape size, Dtype dt, double mean = 0, double std = 1) {
    if (std < 0) {
        throw SailCError("Standard deviation cannot be less than 0");
    }
    alignemnt_information info = getAlignment(dt);
    int numel = size.numel();
    void* data = _malloc_align(numel, info.alignment, info.dtype_size);
    launch_arithmetic(dt, [&](auto pt) {
        using T = typename decltype(pt)::type;

        T* data_rand = (T*)data;

        std::normal_distribution<> dis((T)mean, (T)std);

        for (int i = 0; i < numel; i++) {
            data_rand[i] = dis(gen);
        }
    });

    TensorBody::pointer b = new TensorBody(data, dt, size);
    return Tensor(b, false);
}
Tensor normal(TensorShape size, double mean = 0, double std = 1) {
    return normal(size, default_dtype, mean, std);
}
Tensor normal_like(Tensor tensor, double mean = 0, double std = 1) {
    TensorShape s = tensor.get_shape();
    Tensor ret = normal(s, tensor.get_dtype(), mean, std);
    ret.requires_grad = tensor.requires_grad;
    return ret;
}
}  // namespace random

}  // namespace sail