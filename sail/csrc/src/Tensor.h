#pragma once

#include <iostream>
#include <vector>

#include "Tensor_storage.h"
#include "dtypes.h"
#include "error.h"
#include "types.h"

namespace sail {

class Tensor {
   public:
    explicit Tensor(TensorStorage storage);
    explicit Tensor(){};

    Tensor(int& ndims, void*& data, Dtype& dt, TensorSize& strides,
           TensorSize& shape);

    void reshape(const TensorSize s);

    Tensor expand_dims(const int dim);

    static Tensor createEmptyScalar(Dtype dt);

    void* data();

    bool isScalar();

    Dtype dtype();

    void free();

    long int* getShapePtr();
    int getNPTypeNum();

    Tensor operator+(const Tensor& t);
    Tensor operator-(const Tensor& t);
    Tensor operator*(const Tensor& t);
    Tensor operator/(const Tensor& t);
    Tensor operator[](const int t);

    Tensor sum();

    TensorStorage storage;
};
}  // namespace sail
