#pragma once

#include <iostream>

#include "../Tensor.h"
#include "../types.h"

namespace sail {

inline Tensor reshape(const Tensor& tensor1, const TensorSize new_shape) {
    tensor1.storage.reshape(new_shape);
    return tensor1;
}

} // end sail
