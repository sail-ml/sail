#pragma once

// #include <iostream>

#include "reshape.h"
#include "Tensor.h"
#include "exception.h"
#include "factories.h"
#include "tensor_shape.h"

namespace sail {

namespace ops {

Tensor reshape(const Tensor& tensor1, const TensorShape& new_shape) {
    int s = new_shape.numel();
    if (s != tensor1.numel()) {
        THROW_ERROR_DETAILED(DimensionError, "Cannot reshape tensor of shape ",
                             tensor1.get_shape().get_string(), " to ",
                             new_shape.get_string());
    }
    Tensor new_tensor;
    if (tensor1.is_view()) {  // reshaping a view gets messy
        new_tensor = clone(tensor1);
        new_tensor.set_shape(TensorShape(new_shape));
    } else {
        TensorBody::pointer new_body = TensorBody::pointer(new TensorBody(
            tensor1.get_body()->get_data(), tensor1.get_body()->get_dtype(),
            TensorShape(new_shape), /*is_view*/ true));
        new_tensor = Tensor(new_body, tensor1.requires_grad);
    }
    return new_tensor;
}
}  // namespace ops

}  // namespace sail
