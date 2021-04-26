#pragma once

#include <iostream>

#include "../../Tensor.h"
#include "../../types.h"
#include "../../factories.h"

namespace sail {

namespace ops {

inline Tensor expand_dims(const Tensor& tensor1, const int dim) {
    TensorSize s = tensor1.storage.shape;
    tensor1.reshape(s);
    return tensor1;
}
}


} // end sail
