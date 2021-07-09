#pragma once

#include <iostream>

#include "Tensor.h"
#include "constants.h"
#include "factories.h"
#include "kernels/Kernel.h"

namespace sail {
namespace ops {

Tensor sum(const Tensor& tensor1, int axis = NULLDIM, bool keepdims = false);
Tensor mean(const Tensor& tensor1, int axis = NULLDIM, bool keepdims = false);
Tensor max(const Tensor& tensor1, int axis = NULLDIM, bool keepdims = false);
Tensor min(const Tensor& tensor1, int axis = NULLDIM, bool keepdims = false);

}  // namespace ops
}  // namespace sail
