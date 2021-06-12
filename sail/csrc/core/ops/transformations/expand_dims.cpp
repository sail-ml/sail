#pragma once

#include <iostream>

#include "Tensor.h"
#include "factories.h"
#include "types.h"

namespace sail {

namespace ops {

Tensor expand_dims(const Tensor& tensor1, const int dim) {
    Tensor t2 = tensor1.expand_dims(dim);
    return t2;
}
}  // namespace ops

}  // namespace sail
