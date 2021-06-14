#include <algorithm>
#include <iostream>

#include "Tensor.h"
#include "error.h"
#include "factories.h"
#include "roll_axis.h"
#include "tensor_shape.h"
#include "types.h"

namespace sail {

namespace ops {

Tensor rollaxis(const Tensor& tensor1, const int axis, const int position = 0) {
    TensorShape new_shape = TensorShape(tensor1.get_shape());
    new_shape = new_shape.roll_axis(axis, position);
    new_shape.contiguous = false;
    TensorBody::pointer new_body = TensorBody::pointer(new TensorBody(
        tensor1.get_body()->get_data(), tensor1.get_body()->get_dtype(),
        new_shape, /*is_view*/ true));
    Tensor new_tensor = Tensor(new_body, tensor1.requires_grad);
    return new_tensor;
}
Tensor moveaxis(const Tensor& tensor1, const int axis, const int position = 0) {
    TensorShape new_shape = TensorShape(tensor1.get_shape());
    new_shape = new_shape.move_axis(axis, position);
    new_shape.contiguous = false;
    TensorBody::pointer new_body = TensorBody::pointer(new TensorBody(
        tensor1.get_body()->get_data(), tensor1.get_body()->get_dtype(),
        new_shape, /*is_view*/ true));
    Tensor new_tensor = Tensor(new_body, tensor1.requires_grad);
    return new_tensor;
}

}  // namespace ops

}  // namespace sail