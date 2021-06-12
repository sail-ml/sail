#pragma once

#include <algorithm>
#include <iostream>

#include "Tensor.h"
#include "error.h"
#include "factories.h"
#include "types.h"

namespace sail {

namespace ops {

inline Tensor squeeze(const Tensor& tensor1, const int dim) {
    tensor1.squeeze(dim);
    return tensor1;
}

// inline Tensor squeeze(const Tensor& tensor1) {
//     TensorSize s = tensor1.shape;
//     s.erase(std::remove(s.begin(), s.end(), 1), s.end());
//     tensor1.reshape(s);
//     return tensor1;
// }
}  // namespace ops

}  // namespace sail
