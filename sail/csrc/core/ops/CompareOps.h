#pragma once

#include <iostream>

#include "Tensor.h"

namespace sail {

namespace ops {

Tensor clip(const Tensor& tensor1, const double min, const double max);
Tensor clip_min(const Tensor& tensor1, const double min);
Tensor clip_max(const Tensor& tensor1, const double max);

Tensor elementwise_equal(const Tensor& tensor1, const Tensor& tensor2);
Tensor elementwise_lt(const Tensor& tensor1, const Tensor& tensor2);
Tensor elementwise_lte(const Tensor& tensor1, const Tensor& tensor2);
Tensor elementwise_gt(const Tensor& tensor1, const Tensor& tensor2);
Tensor elementwise_gte(const Tensor& tensor1, const Tensor& tensor2);
Tensor elementwise_ne(const Tensor& tensor1, const Tensor& tensor2);

}  // namespace ops

}  // namespace sail
