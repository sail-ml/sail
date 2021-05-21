#include <iostream>

#include "../../Tensor.h"
#include "../../error.h"
#include "../../tensor_shape.h"
#include "transpose.h"

namespace sail {

namespace ops {

Tensor transpose(const Tensor& tensor1) {
    TensorShape new_shape = TensorShape(tensor1.get_shape());
    new_shape.reverse();
    TensorBody::pointer new_body = TensorBody::pointer(new TensorBody(
        tensor1.get_body()->get_data(), tensor1.get_body()->get_dtype(),
        new_shape, /*is_view*/ true));
    Tensor new_tensor = Tensor(new_body, tensor1.requires_grad);
    return new_tensor;
}

Tensor transpose(const Tensor& tensor1, LongVec& dims) {
    if (tensor1.get_ndim() != dims.size()) {
        throw SailCError("Transpose axes must have same length as tensor");
    }

    TensorShape new_shape = TensorShape(tensor1.get_shape());
    new_shape.reorder(dims);
    TensorBody::pointer new_body = TensorBody::pointer(new TensorBody(
        tensor1.get_body()->get_data(), tensor1.get_body()->get_dtype(),
        new_shape, /*is_view*/ true));
    Tensor new_tensor = Tensor(new_body, tensor1.requires_grad);
    return new_tensor;
}
}  // namespace ops

}  // namespace sail
