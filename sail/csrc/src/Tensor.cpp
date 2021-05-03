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
#include "tensor_shape.h"
#include "types.h"
#include "utils.h"

#define MIN_NUMEL_PAR 4096

using Tensor = sail::Tensor;
using namespace std::chrono;

namespace sail {

// CONSTRUCTORS

Tensor::Tensor(int& _ndims, void*& _data, Dtype& _dt, TensorShape shape_data) {
    info = getAlignment(_dt);

    dtype = _dt;
    ndim = _ndims;
    // arr_numel = _numel(shape);
    shape_details = shape_data;
    ndim = shape_details.ndim();
    arr_numel = shape_details.numel();

    data = _realloc_align(_data, arr_numel, info.alignment, info.dtype_size);
}
Tensor::Tensor(int& _ndims, void*& _data, Dtype& _dt, TensorShape shape_data,
               bool rq) {
    info = getAlignment(_dt);

    dtype = _dt;
    ndim = _ndims;
    // arr_numel = _numel(shape);
    requires_grad = rq;
    shape_details = shape_data;
    ndim = shape_details.ndim();
    arr_numel = shape_details.numel();

    data = _realloc_align(_data, arr_numel, info.alignment, info.dtype_size);
}

static Tensor Tensor::move(int& _ndims, void*& _data, Dtype& _dt,
                           TensorShape shape_data) {
    Tensor t = Tensor();
    t.info = getAlignment(_dt);

    t.dtype = _dt;
    t.ndim = _ndims;
    // t.arr_numel = _numel(t.shape);

    t.data = std::move(_data);
    t.shape_details = shape_data;
    t.arr_numel = t.shape_details.numel();
    return t;
}

long Tensor::getTotalSize() {
    long size = GetDtypeSize(dtype);
    for (long value : shape_details.shape) {
        size = size * value;
    }
    return size;
}

Tensor Tensor::reshape(const TensorShape new_shape) {
    int s = new_shape.numel();
    if (s != arr_numel) {
        throw DimensionError{"Cannot reshape tensor of shape ",
                             shape_details.get_string(), " to ",
                             new_shape.get_string()};
    }

    shape_details = new_shape;
    return *this;
}

Tensor Tensor::expand_dims(const int dim) {
    TensorShape s = shape_details;
    s.insert_one(dim);
    // TensorSize s = shape;
    // s.insert(s.begin() + dim, 1);
    reshape(s);
    return *this;
}

Tensor Tensor::squeeze(const int dim) {
    shape_details.remove_one(dim);
    // ops::squeeze(*this, dim);
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

long int* Tensor::get_shape_ptr() { return shape_details.get_shape_ptr(); }

int Tensor::get_np_type_num() { return get_np_type_numFromDtype(dtype); }

// todo - move to op
Tensor Tensor::cast(const Dtype dt) {
    Tensor casted = ops::cast(*this, dt);
    return casted;
}

int Tensor::get_ndim() { return shape_details.ndim(); }

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
    Tensor e = empty(new_ndim, dtype, TensorShape(new_shape, new_strides));
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
