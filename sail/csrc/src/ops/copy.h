#pragma once

#include <iostream>

#include "../Tensor.h"
#include "../dtypes.h"
#include "../kernels/kernel.h"

namespace sail {

namespace ops {

Tensor copy(Tensor& tensor1);
Tensor cast(Tensor& tensor1, Dtype dt);

}  // namespace ops

}  // namespace sail
