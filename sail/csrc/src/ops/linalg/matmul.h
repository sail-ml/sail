#pragma once

#include <iostream>

#include "../../Tensor.h"
#include "../../dtypes.h"
#include "../../kernels/kernel.h"

namespace sail {

namespace ops {

Tensor matmul(const Tensor& t1, const Tensor& t2);

}  // namespace ops

}  // namespace sail
