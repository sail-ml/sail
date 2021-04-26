#include "Tensor.h"

#include <chrono>
#include <iostream>
#include <vector>

#include "Tensor_storage.h"
#include "cuda/cuda_math.h"
#include "dtypes.h"
#include "factories.h"
#include "kernels/kernel.h"
#include "ops/elementwise.h"
#include "ops/reduction.h"
#include "ops/transformations/expand_dims.h"
#include "types.h"
#include "utils.h"

using Tensor = sail::Tensor;
using namespace std::chrono;

namespace sail {

// CONSTRUCTORS

Tensor::Tensor(int& ndims, void*& data, Dtype& dt, TensorSize& strides,
               TensorSize& shape) {
    storage = TensorStorage(ndims, data, 0, dt, strides, shape);
}

Tensor::Tensor(TensorStorage storage_) { storage = storage_; }

void Tensor::reshape(const TensorSize s) { storage.reshape(s); }

Tensor Tensor::expand_dims(const int dim) {
    TensorSize s = storage.shape;
    s.insert(s.begin() + dim, 1);
    reshape(s);
    return *this;
}

void* Tensor::data() { return storage.data; }

bool Tensor::isScalar() {
    if (storage.numel() == 1) {
        return true;
    }
    return false;
}

Dtype Tensor::dtype() { return storage.dtype; }

void Tensor::free() { storage.free_data(); }

long int* Tensor::getShapePtr() {
    // std::vector<int> sh(storage.shape.begin(), storage.shape.end());
    return &storage.shape[0];
}

int Tensor::getNPTypeNum() { return GetNPTypeNumFromDtype(storage.dtype); }

// operators

Tensor Tensor::operator[](const int index) {
    // auto new_ptr = storage[index];
    // auto out = storage.step_back();
    // int new_ndim = (storage.ndim) - 1;
    // Dtype dt = storage.dtype;
    Tensor e =
        empty(storage.ndim, storage.dtype, storage.strides, storage.shape);
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

Tensor Tensor::sum() { return ops::sum(*this); }

}  // namespace sail
