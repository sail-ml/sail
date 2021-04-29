#pragma once

#include <iostream>

#include "../../Tensor.h"
#include "../../factories.h"
#include "../../types.h"

namespace sail {

namespace ops {

inline Tensor expand_dims(const Tensor& tensor1, const int dim) {
    TensorSize s = tensor1.shape;
    s.insert(s.begin() + dim, 1);
    tensor1.reshape(s);
    return tensor1;
}
}  // namespace ops

}  // namespace sail
