#pragma once

#include <algorithm>
#include <iostream>

#include "../../Tensor.h"
#include "../../error.h"
#include "../../factories.h"
#include "../../types.h"

namespace sail {

namespace ops {

inline Tensor squeeze(const Tensor& tensor1, const int dim) {
    // TensorSize s = tensor1.shape;
    // if (s[dim] != 1) {
    //     throw SailCError("squeeze dimension shape must be 1");
    // }
    // s.erase(s.begin() + dim);
    tensor1.shape_details.remove_one(dim);
    tensor1.reshape(tensor1.shape_details);
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
