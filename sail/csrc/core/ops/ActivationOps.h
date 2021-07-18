#pragma once

#include <iostream>

#include "Tensor.h"

namespace sail {

namespace ops {

Tensor relu(const Tensor& tensor1);
Tensor softmax(Tensor& tensor1, const int axis = 1);
Tensor log_softmax(Tensor& tensor1, const int axis = 1);
Tensor sigmoid(const Tensor& tensor1);

}  // namespace ops

}  // namespace sail
