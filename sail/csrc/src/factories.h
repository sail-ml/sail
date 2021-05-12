
#pragma once

#include <iostream>
#include <vector>

#include "Tensor.h"
#include "dtypes.h"
#include "error.h"
#include "types.h"

namespace sail {

Tensor empty(int ndims, Dtype dt, TensorShape shape);
Tensor make_view(int ndims, void* data, Dtype dt, TensorShape shape);
Tensor copy(Tensor t);
Tensor clone(Tensor& t);
// Tensor empty_like(int ndims, void* data, Dtype dt, TensorSize strides,
// TensorSize shape);

Tensor empty_scalar(Dtype dt);
Tensor one_scalar(Dtype dt);

Tensor from_data(void* data, Dtype dt, TensorShape s);

}  // namespace sail