#pragma once

#include <iostream>

#include "../../Tensor.h"
#include "../../factories.h"
#include "../../kernels/kernel.h"

namespace sail {
namespace ops {

Tensor softmax(Tensor& input, int axis = 1);
Tensor log_softmax(Tensor& input, int axis = 1);

}  // namespace ops
}  // namespace sail
