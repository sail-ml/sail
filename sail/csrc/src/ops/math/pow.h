#pragma once

#include <iostream>

#include "../../Tensor.h"
#include "../../dtypes.h"
#include "../../kernels/kernel.h"

namespace sail {

namespace ops {

Tensor pow(Tensor& tensor1, Tensor& power);
Tensor exp(Tensor& tensor1);

}  // namespace ops

}  // namespace sail
