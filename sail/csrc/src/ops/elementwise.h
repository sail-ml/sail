#pragma once

#include <iostream>

#include "../Tensor.h"
#include "../kernels/kernel.h"

namespace sail {

namespace ops {

Tensor add(Tensor& tensor1, Tensor& tensor2);
Tensor subtract(Tensor& tensor1, Tensor& tensor2);
Tensor multiply(Tensor& tensor1, Tensor& tensor2);
Tensor divide(Tensor& tensor1, Tensor& tensor2);

}  // namespace ops

}  // namespace sail
