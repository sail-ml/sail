#pragma once

#include <iostream>

#include "Tensor.h"
#include "dtypes.h"
#include "kernels/Kernel.h"

namespace sail {

namespace ops {

Tensor copy(Tensor& tensor1);
Tensor view(Tensor& tensor1);
Tensor cast(Tensor& tensor1, Dtype dt);
Tensor internal_fast_cast(Tensor& tensor1, Dtype dt);

}  // namespace ops

}  // namespace sail