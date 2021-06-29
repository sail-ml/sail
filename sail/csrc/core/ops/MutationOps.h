#pragma once

#include <iostream>

#include "Tensor.h"
#include "factories.h"
#include "tensor_shape.h"
#include "types.h"

namespace sail {

namespace ops {

Tensor expand_dims(const Tensor& tensor1, const int dim);
Tensor squeeze(const Tensor& tensor1, const int dim);
Tensor reshape(const Tensor& tensor1, const TensorShape& new_shape);
Tensor rollaxis(const Tensor& tensor1, const int axis, const int position = 0);
Tensor moveaxis(const Tensor& tensor1, const int axis, const int position = 0);
Tensor transpose(const Tensor& tensor1);
Tensor transpose(const Tensor& tensor1, const LongVec& dims);

Tensor cat(std::vector<Tensor> tensors, const int axis = 0);
Tensor stack(std::vector<Tensor> tensors, const int axis = 0);

}  // namespace ops

}  // namespace sail
