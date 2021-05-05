#pragma once

#include <iostream>

#include "../Tensor.h"
#include "../factories.h"
#include "../kernels/kernel.h"

namespace sail {
namespace ops {

Tensor sum(Tensor& tensor1);
Tensor sum(const Tensor& tensor1, int axis);
Tensor mean(const Tensor& tensor1);

}  // namespace ops
}  // namespace sail
