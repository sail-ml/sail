#pragma once

#include <iostream>

#include "Tensor.h"
#include "factories.h"
#include "types.h"

namespace sail {

namespace ops {

Tensor expand_dims(const Tensor& tensor1, const int dim);
}  // namespace ops

}  // namespace sail
