#pragma once

#include <iostream>

#include "Tensor.h"

namespace sail {

namespace ops {

Tensor exp(const Tensor& tensor1);
Tensor log(const Tensor& tensor1);
Tensor power(const Tensor& tensor1, const Tensor& tensor2);

}  // namespace ops

}  // namespace sail
