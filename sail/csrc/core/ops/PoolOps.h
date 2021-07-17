#pragma once

#include <iostream>

#include "Tensor.h"
#include "dtypes.h"
#include "tensor_shape.h"

namespace sail {

namespace ops {

Tensor max_pool_2d(Tensor& input, TensorShape kernel_size,
                   std::vector<long> strides, std::vector<long> padding);
}  // namespace ops

}  // namespace sail
