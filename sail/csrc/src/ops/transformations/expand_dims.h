#pragma once

#include <iostream>

#include "../../Tensor.h"
#include "../../factories.h"
#include "../../types.h"

namespace sail {

namespace ops {

inline Tensor expand_dims(const Tensor& tensor1, const int dim) {
    tensor1.get_shape().insert_one(dim);
    tensor1.reshape(tensor1.get_shape());
    return tensor1;
}
}  // namespace ops

}  // namespace sail
