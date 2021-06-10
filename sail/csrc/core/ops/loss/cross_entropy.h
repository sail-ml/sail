#pragma once

#include <iostream>

#include "../../Tensor.h"
#include "../../factories.h"
#include "../../kernels/kernel.h"

namespace sail {
namespace ops {

Tensor softmax_cross_entropy(Tensor& logits, Tensor& targets);

}  // namespace ops
}  // namespace sail
