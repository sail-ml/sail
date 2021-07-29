
#pragma once

#include "factories.h"

#include <iostream>
#include <random>
#include <vector>

#include <chrono>
#include "Tensor.h"
#include "TensorBody.h"
#include "dtypes.h"
#include "exception.h"
#include "kernels/Kernel.h"
#include "tensor_iterator.h"
#include "tensor_shape.h"
#include "types.h"
#include "utils.h"
using namespace std::chrono;

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
    TensorBody::pointer body = TensorBody::pointer(new TensorBody(
        tensor.get_dtype(), TensorShape(tensor.get_shape().shape)));

    Tensor _empty = Tensor(body, tensor.requires_grad);
    _empty.requires_grad = tensor.requires_grad;
    return _empty;
}
Tensor empty_like(const Tensor& tensor, Dtype& dt) {
    TensorBody::pointer body = TensorBody::pointer(
        new TensorBody(dt, TensorShape(tensor.get_shape().shape)));

    Tensor _empty = Tensor(body, tensor.requires_grad);
    _empty.requires_grad = tensor.requires_grad;
    return _empty;
}

Tensor clone(const Tensor& t) {
    void* data;
    TensorShape s = t.get_shape();
    alignemnt_information info = getAlignment(t.get_dtype());
    if (t.is_view()) {
        int numel = s.numel();
        data = _malloc_align(numel, info.alignment, info.dtype_size);
        s = TensorShape(s.shape);
    } else {
        data = _realloc_align(t.get_data(), t.numel(), info.alignment,
                              info.dtype_size);
    }

    TensorBody::pointer body = new TensorBody(data, t.get_dtype(), s);

    Tensor _empty = Tensor(body, t.requires_grad);
    if (t.is_view()) {
        sail::internal::copy_stub(t, _empty);
    }
    return _empty;
}

Tensor make_view(void* data, Dtype dt, TensorShape shape) {
    TensorBody::pointer b =
        TensorBody::pointer((new TensorBody(data, dt, shape, true)));
    Tensor _empty = Tensor(b, false);
    return _empty;
}

Tensor as_strided(const Tensor& t, TensorShape s) {
    return make_view(t.get_data(), t.get_dtype(), s);
}

Tensor one_hot(const Tensor& t, const int size, Dtype dt) {
    if (t.get_dtype() != Dtype::sInt16 && t.get_dtype() != Dtype::sInt32 &&
        t.get_dtype() != Dtype::sInt64) {
        throw SailCError("inputs must be integers");
    }
    long total_data = size * t.numel();
    alignemnt_information info = getAlignment(dt);
    void* data = _calloc_align(total_data, info.alignment, info.dtype_size);
    dispatch_all_numeric_types(dt, [&](auto pt) {
        using T = typename decltype(pt)::type;

        T* t_data = (T*)data;
        int start = 0;
        for (int i = 0; i < t.numel(); i++) {
            int jump = ((int*)t[i].get_data())[0];
            int leftover = size - jump;
            start += jump;
            t_data[start] = 1;
            start += leftover;
        }
    });
    TensorShape shape = TensorShape({t.numel(), size});
    TensorBody::pointer b =
        TensorBody::pointer((new TensorBody(data, dt, shape, false)));
    Tensor _empty = Tensor(b, t.requires_grad);
    return _empty;
}

Tensor make_view(const Tensor& t) {
    TensorBody::pointer b = TensorBody::pointer(
        (new TensorBody(t.get_data(), t.get_dtype(), t.get_shape(), true)));
    Tensor _empty = Tensor(b, t.requires_grad);
    return _empty;
}

Tensor one_scalar(Dtype dt) {
    alignemnt_information info = getAlignment(dt);
    void* data = _malloc_align(1, info.alignment, info.dtype_size);

    dispatch_all_numeric_types(dt, [&](auto pt) {
        using T = typename decltype(pt)::type;
        T x = (T)1;
        *(T*)data = x;
    });

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
Tensor ones(TensorShape size, Dtype dt) {
    alignemnt_information info = getAlignment(dt);
    int numel = size.numel();
    void* new_data =
        _malloc_align(size.numel(), info.alignment, info.dtype_size);
    dispatch_all_numeric_types(dt, [&](auto pt) {
        using T = typename decltype(pt)::type;

        T* data_fill = (T*)new_data;

        for (int i = 0; i < numel; i++) {
            data_fill[i] = (T)1;
        }
    });

    TensorBody::pointer b = new TensorBody(new_data, dt, size);
    return Tensor(b, false);
}

Tensor full(Numeric n, TensorShape size) {
    Tensor v = Tensor(n.get(), false);

    Dtype dt = v.get_dtype();

    alignemnt_information info = getAlignment(dt);
    int numel = size.numel();
    void* new_data =
        _malloc_align(size.numel(), info.alignment, info.dtype_size);
    dispatch_all_numeric_types(dt, [&](auto pt) {
        using T = typename decltype(pt)::type;

        T* data_fill = (T*)new_data;
        T d = v.get<T>();

        for (int i = 0; i < numel; i++) {
            data_fill[i] = d;
        }
    });

    TensorBody::pointer b = new TensorBody(new_data, dt, size);
    return Tensor(b, false);
}

namespace random {

Tensor uniform(TensorShape size, Dtype dt, double min, double max) {
    alignemnt_information info = getAlignment(dt);
    int numel = size.numel();
    void* data = _malloc_align(numel, info.alignment, info.dtype_size);
    dispatch_all_numeric_types(dt, [&](auto pt) {
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
Tensor uniform(TensorShape size, double min, double max) {
    return uniform(size, default_dtype, min, max);
}
Tensor uniform_like(Tensor tensor, double min, double max) {
    TensorShape s = tensor.get_shape();
    Tensor ret = uniform(s, tensor.get_dtype(), min, max);
    ret.requires_grad = tensor.requires_grad;
    return ret;
}
Tensor uniform_fill(Tensor tensor, double min, double max) {
    Dtype dt = tensor.get_dtype();
    int numel = tensor.numel();
    void* data = tensor.get_data();
    dispatch_all_numeric_types(dt, [&](auto pt) {
        using T = typename decltype(pt)::type;

        T* data_rand = (T*)data;

        std::uniform_real_distribution<> dis((T)min, (T)max);

        for (int i = 0; i < numel; i++) {
            data_rand[i] = dis(gen);
        }
    });
    return tensor;
}

Tensor normal(TensorShape size, Dtype dt, double mean, double std) {
    if (std < 0) {
        throw SailCError("Standard deviation cannot be less than 0");
    }
    alignemnt_information info = getAlignment(dt);
    int numel = size.numel();
    void* data = _malloc_align(numel, info.alignment, info.dtype_size);
    dispatch_all_numeric_types(dt, [&](auto pt) {
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
Tensor normal(TensorShape size, double mean, double std) {
    return normal(size, default_dtype, mean, std);
}
Tensor normal_like(Tensor tensor, double mean, double std) {
    TensorShape s = tensor.get_shape();
    Tensor ret = normal(s, tensor.get_dtype(), mean, std);
    ret.requires_grad = tensor.requires_grad;
    return ret;
}
Tensor normal_fill(Tensor tensor, double mean, double std) {
    Dtype dt = tensor.get_dtype();
    int numel = tensor.numel();
    void* data = tensor.get_data();
    dispatch_all_numeric_types(dt, [&](auto pt) {
        using T = typename decltype(pt)::type;

        T* data_rand = (T*)data;

        std::normal_distribution<> dis((T)mean, (T)std);

        for (int i = 0; i < numel; i++) {
            data_rand[i] = dis(gen);
        }
    });
    return tensor;
}
}  // namespace random

}  // namespace sail