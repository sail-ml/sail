#pragma once

#include <iostream>

#include "../../Tensor.h"
#include "../../tensor_shape.h"

namespace sail {

namespace ops {

Tensor transpose(const Tensor& tensor1);
Tensor transpose(const Tensor& tensor1, const LongVec& dims);
}  // namespace ops

}  // namespace sail
