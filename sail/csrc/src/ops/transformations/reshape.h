#pragma once

#include <iostream>

#include "../../Tensor.h"
#include "../../tensor_shape.h"

namespace sail {

namespace ops {

Tensor reshape(const Tensor& tensor1, const TensorShape& new_shape);
}

}  // namespace sail
