#include "Tensor.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <iostream>
#include <vector>
#include <chrono>

#include "Tensor_storage.h"
#include "cuda/cuda_math.h"
#include "dtypes.h"
#include "kernels/kernel.h"
#include "types.h"
#include "utils.h"
#include "ops/elementwise.h"
#include "ops/transformations/expand_dims.h"
#include "ops/reduction.h"
#include "factories.h"

namespace py = pybind11;
using Tensor = sail::Tensor;
using namespace std::chrono;

namespace sail {

// CONSTRUCTORS

Tensor::Tensor(py::array arr) {
    auto dtype = GetDtypeFromNumpyDtype(arr.dtype());
    auto buff = arr.request();

    storage = TensorStorage(
        buff.ndim,
        static_cast<void*>(buff.ptr),
        0,
        dtype,
        buff.strides,
        buff.shape);
}

// TODO(tucker) - define std::vector<py::ssize_t> as a type

Tensor::Tensor(
    int& ndims,
    void*& data,
    Dtype& dt,
    std::vector<py::ssize_t>& strides,
    std::vector<py::ssize_t>& shape) {
    storage = TensorStorage(ndims, data, 0, dt, strides, shape);
}

Tensor::Tensor(TensorStorage storage_) {
    storage = storage_;
}

void Tensor::reshape(const TensorSize s) {
    storage.reshape(s);
}

Tensor Tensor::expand_dims(const int dim) {
    TensorSize s = storage.shape;
    s.insert(s.begin() + dim, 1);
    reshape(s);
    return *this;
}


void* Tensor::data() {
    return storage.data;
}

bool Tensor::isScalar() {
    if (storage.numel() == 1) {
        return true;
    }
    return false;
}

py::array Tensor::getBuffer() {
    auto shape = storage.shape;
    auto strides = storage.strides;
    auto ndim = storage.ndim;
    if (storage.ndim <= 0) {
        ndim = 1;
        shape = {1};
        strides = {storage.getDtypeSize()};
    }
    auto buffer = py::buffer_info(
        storage.data,
        storage.getDtypeSize(),
        storage.getFormat(),
        ndim,
        shape,
        strides);
    return py::array(buffer);
}

Dtype Tensor::dtype() {
    return storage.dtype;
}

void Tensor::free() {
    storage.free_data();
}

long int* Tensor::getShapePtr() {
	// std::vector<int> sh(storage.shape.begin(), storage.shape.end());
	return &storage.shape[0];
}

int Tensor::getNPTypeNum() {
	return GetNPTypeNumFromDtype(storage.dtype);
}

// operators

Tensor Tensor::operator[](const int index) {
    // auto new_ptr = storage[index];
    // auto out = storage.step_back();
    // int new_ndim = (storage.ndim) - 1;
    // Dtype dt = storage.dtype;
    Tensor e = empty(storage.ndim, storage.dtype, storage.strides, storage.shape);
    e.storage.data = storage.data;

    return e;
}

Tensor Tensor::operator+(const Tensor& other) {
    return ops::add(*this, other);
}
Tensor Tensor::operator-(const Tensor& other) {
    return ops::subtract(*this, other);
}
Tensor Tensor::operator*(const Tensor& other) {
    return ops::multiply(*this, other);
}
Tensor Tensor::operator/(const Tensor& other) {
    return ops::divide(*this, other);
}

// UNARY OPS

Tensor Tensor::sum() {
    return ops::sum(*this);
}

} // namespace sail
