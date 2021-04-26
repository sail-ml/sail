
#pragma once

#include <iostream>
#include <vector>

#include "Tensor.h"
#include "Tensor_storage.h"
#include "dtypes.h"
#include "error.h"
#include "types.h"

namespace sail {

Tensor empty(int ndims, Dtype dt, TensorSize strides, TensorSize shape);
Tensor copy(Tensor t);
// Tensor empty_like(int ndims, void* data, Dtype dt, TensorSize strides,
// TensorSize shape);

Tensor empty_scalar(Dtype dt);
}  // namespace sail