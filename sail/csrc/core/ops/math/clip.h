#pragma once

#include <iostream>

#include "Tensor.h"
#include "dtypes.h"
#include "kernels/Kernel.h"

namespace sail {

namespace ops {

Tensor clip(Tensor& tensor1, double min);
Tensor clip(Tensor& tensor1, double min, double max);

}  // namespace ops

}  // namespace sail
