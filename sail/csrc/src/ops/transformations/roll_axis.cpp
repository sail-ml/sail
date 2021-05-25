#pragma once

#include <algorithm>
#include <iostream>

#include "../../Tensor.h"
#include "../../error.h"
#include "../../factories.h"
#include "../../tensor_shape.h"
#include "../../types.h"

namespace sail {

namespace ops {

Tensor rollaxis(const Tensor& tensor1, const int axis, const int position = 0) {
    // NEED ERROR CHECKING

    TensorShape new_shape = TensorShape(tensor1.get_shape());
    new_shape = new_shape.move_axis(axis, position);
    TensorBody::pointer new_body = TensorBody::pointer(new TensorBody(
        tensor1.get_body()->get_data(), tensor1.get_body()->get_dtype(),
        new_shape, /*is_view*/ true));
    Tensor new_tensor = Tensor(new_body, tensor1.requires_grad);
    return new_tensor;
}

}  // namespace ops

}  // namespace sail
