
#pragma once

#include <iostream>
#include <vector>

#include "Tensor.h"
#include "dtypes.h"
#include "error.h"
#include "types.h"

namespace sail {

Tensor empty(const int ndims, const Dtype& dt, const TensorShape& shape);
Tensor empty_like(const Tensor& tensor);
Tensor make_view(void* data, Dtype dt, TensorShape shape);
Tensor make_view(const Tensor& t);
Tensor copy(Tensor t);
Tensor clone(const Tensor& t);
// Tensor empty_like(int ndims, void* data, Dtype dt, TensorSize strides,
// TensorSize shape);

Tensor empty_scalar(Dtype dt);
Tensor one_scalar(Dtype dt);
Tensor zero_scalar(Dtype dt);
Tensor* create_grad(Dtype dt);

Tensor from_data(void* data, Dtype dt, TensorShape s);

Tensor zeros(TensorShape size, Dtype dt);

namespace random {  // probably want to refactor factories to be in their own
                    // namespace but rolling with this for now

// need to be able to instantiate random tensors
Tensor uniform(TensorShape size, Dtype dt, double min = 0, double max = 1);
Tensor uniform_like(Tensor tensor, double min = 0, double max = 1);
Tensor normal(TensorShape size, Dtype dt, double mean = 0, double std = 1);
Tensor normal_like(Tensor tensor, double mean = 0, double std = 1);
}  // namespace random

}  // namespace sail