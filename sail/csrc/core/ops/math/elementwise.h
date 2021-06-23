#pragma once

#include <iostream>

#include "Tensor.h"
#include "kernels/Kernel.h"

namespace sail {

namespace ops {

Tensor add(const Tensor& tensor1, const Tensor& tensor2);
Tensor iadd(Tensor& tensor1, const Tensor& tensor2);
Tensor subtract(const Tensor& tensor1, const Tensor& tensor2);
Tensor multiply(const Tensor& tensor1, const Tensor& tensor2);
Tensor multiply(const Tensor& tensor1, const Tensor& tensor2);
Tensor divide(const Tensor& tensor1, const Tensor& tensor2);

}  // namespace ops

}  // namespace sail
