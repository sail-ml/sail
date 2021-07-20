#pragma once

#include <iostream>

#include "Tensor.h"
#include "dtypes.h"
#include "kernels/Kernel.h"

namespace sail {

namespace ops {

Tensor copy(Tensor& tensor1);
void copy(Tensor& dest, const Tensor& source);
Tensor view(Tensor& tensor1);
Tensor cast(const Tensor& tensor1, Dtype dt);
Tensor internal_fast_cast(Tensor& tensor1, Dtype dt);
Tensor pad(const Tensor& t1, std::vector<std::vector<long>> x);
}  // namespace ops

}  // namespace sail
