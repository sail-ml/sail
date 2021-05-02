#include "Tensor.h"

#include <chrono>
#include <iostream>
#include <vector>

#include "autograd/autograd.h"
#include "cuda/cuda_math.h"
#include "dtypes.h"
#include "factories.h"
#include "kernels/kernel.h"
#include "ops/ops.h"
#include "types.h"
#include "utils.h"

#define MIN_NUMEL_PAR 4096

using Tensor = sail::Tensor;
using namespace std::chrono;

namespace sail {

// CONSTRUCTORS

Tensor::Tensor(int& _ndims, void*& _data, Dtype& _dt, TensorSize& _strides,
               TensorSize& _shape) {
    info = getAlignment(_dt);

    dtype = _dt;
    ndim = _ndims;
    strides = _strides;
    shape = _shape;
    arr_numel = _numel(shape);

    data = _realloc_align(_data, arr_numel, info.alignment, info.dtype_size);
}
Tensor::Tensor(int& _ndims, void*& _data, Dtype& _dt, TensorSize& _strides,
               TensorSize& _shape, bool rq) {
    info = getAlignment(_dt);

    dtype = _dt;
    ndim = _ndims;
    strides = _strides;
    shape = _shape;
    arr_numel = _numel(shape);
    requires_grad = rq;

    data = _realloc_align(_data, arr_numel, info.alignment, info.dtype_size);
}
static Tensor Tensor::move(int& _ndims, void*& _data, Dtype& _dt,
                           TensorSize& _strides, TensorSize& _shape) {
    Tensor t = Tensor();
    t.info = getAlignment(_dt);

    t.dtype = _dt;
    t.ndim = _ndims;
    t.strides = _strides;
    t.shape = _shape;
    t.arr_numel = _numel(t.shape);

    t.data = std::move(_data);
    return t;
}

long Tensor::getTotalSize() {
    long size = GetDtypeSize(dtype);
    for (long value : shape) {
        size = size * value;
    }
    return size;
}

Tensor Tensor::reshape(const TensorSize new_shape) {
    int s = prod_size_vector(new_shape);
    if (s != arr_numel) {
        throw DimensionError{"Cannot reshape tensor of shape ",
                             getVectorString(shape), " to ",
                             getVectorString(new_shape)};
    }

    shape = new_shape;
    TensorSize new_strides;
    long dt_size = GetDtypeSize(dtype);
    for (long s : shape) {
        new_strides.push_back(dt_size * s);
    }
    new_strides.pop_back();
    new_strides.push_back(dt_size);

    strides = new_strides;
    ndim = shape.size();
    return *this;
}

Tensor Tensor::expand_dims(const int dim) {
    TensorSize s = shape;
    s.insert(s.begin() + dim, 1);
    reshape(s);
    return *this;
}

Tensor Tensor::squeeze(const int dim) {
    TensorSize s = shape;
    ops::squeeze(*this, dim);
    return *this;
}

bool Tensor::is_scalar() {
    if (arr_numel == 1) {
        return true;
    }
    return false;
}

int Tensor::numel() const { return arr_numel; }

void Tensor::register_op(autograd::Function* new_func) { fcn = new_func; }

void Tensor::free() {
    // std::cout << "FREEING TENSOR" << std::endl;
    std::free(data);
    data = NULL;
}

long int* Tensor::get_shape_ptr() { return &shape[0]; }

int Tensor::get_np_type_num() { return get_np_type_numFromDtype(dtype); }

// todo - move to op
Tensor Tensor::cast(const Dtype dt) {
    Tensor casted = ops::cast(*this, dt);
    return casted;
}

// // operators

Tensor Tensor::operator[](const int index) {
    auto new_ptr = ((void*)data) + ((index * strides[0]));

    TensorSize new_strides;
    for (int i = 1; i < ndim; i++) {
        new_strides.push_back(strides[i]);
    }
    std::cout << getVectorString(new_strides) << std::endl;
    TensorSize new_shape;
    for (int i = 1; i < ndim; i++) {
        new_shape.push_back(shape[i]);
    }

    int new_ndim = (ndim)-1;
    Tensor e = empty(new_ndim, dtype, new_strides, new_shape);
    e.data = std::move(new_ptr);

    return e;
}

Tensor Tensor::operator+(Tensor& other) { return ops::add(*this, other); }
Tensor Tensor::operator-(Tensor& other) { return ops::subtract(*this, other); }
Tensor Tensor::operator*(Tensor& other) { return ops::multiply(*this, other); }
Tensor Tensor::operator/(Tensor& other) { return ops::divide(*this, other); }

// UNARY OPS

Tensor Tensor::sum() { return ops::sum(*this); }

void Tensor::backward() {
    std::cout << fcn->getName() << std::endl;
    // for (Tensor i : fcn->)
}

}  // namespace sail
