#pragma once

#include <iostream>

#include "../../Tensor.h"
#include "../../error.h"
#include "../../tensor_shape.h"
#include "reshape.h"

namespace sail {

namespace ops {

Tensor reshape(const Tensor& tensor1, const TensorShape& new_shape) {
    int s = new_shape.numel();
    if (s != tensor1.numel()) {
        throw DimensionError{"Cannot reshape tensor of shape ",
                             tensor1.get_shape().get_string(), " to ",
                             new_shape.get_string()};
    }
    TensorBody::pointer new_body = TensorBody::pointer(new TensorBody(
        tensor1.get_body()->get_data(), tensor1.get_body()->get_dtype(),
        new_shape, /*is_view*/ true));
    Tensor new_tensor = Tensor(new_body, tensor1.requires_grad);
    return new_tensor;
}

}  // namespace ops

}  // namespace sail
