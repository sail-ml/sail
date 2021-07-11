#pragma once

#include <iostream>

#include "Tensor.h"
#include "autograd/autograd.h"
#include "factories.h"
#include "kernels/Kernel.h"
#include "tensor_iterator.h"
#include "tensor_shape.h"
#include "types.h"

namespace sail {

namespace ops {
using TensorVector = std::vector<Tensor>;
Tensor expand_dims(const Tensor& tensor1, const int dim) {
    Tensor t2 = tensor1.expand_dims(dim);
    return t2;
}
Tensor squeeze(const Tensor& tensor1, const int dim) {
    Tensor t2 = tensor1.squeeze(dim);
    return t2;
}

Tensor reshape(const Tensor& tensor1, const TensorShape& new_shape) {
    if (tensor1.requires_grad) {
        TensorVector vec;
        vec.emplace_back(tensor1);
        Tensor empty_tensor = (new autograd::Reshape(new_shape))->apply(vec);

        return empty_tensor;
    }
    int s = new_shape.numel();
    if (s != tensor1.numel()) {
        THROW_ERROR_DETAILED(DimensionError, "Cannot reshape tensor of shape ",
                             tensor1.get_shape().get_string(), " to ",
                             new_shape.get_string());
    }
    Tensor new_tensor;
    if (tensor1.is_view()) {  // reshaping a view gets messy
        new_tensor = clone(tensor1);
        new_tensor.set_shape(new_shape);
    } else {
        TensorBody::pointer new_body = TensorBody::pointer(new TensorBody(
            tensor1.get_body()->get_data(), tensor1.get_body()->get_dtype(),
            new_shape, /*is_view*/ true));
        new_tensor = Tensor(new_body, tensor1.requires_grad);
    }
    return new_tensor;
}

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

Tensor transpose(const Tensor& tensor1) {
    TensorShape new_shape = TensorShape(tensor1.get_shape());
    new_shape.reverse();
    new_shape.contiguous = false;
    TensorBody::pointer new_body = TensorBody::pointer(new TensorBody(
        tensor1.get_body()->get_data(), tensor1.get_body()->get_dtype(),
        new_shape, /*is_view*/ true));
    Tensor new_tensor = Tensor(new_body, tensor1.requires_grad);
    return new_tensor;
}
Tensor transpose(const Tensor& tensor1, const LongVec& dims) {
    if (dims.size() == 0) {
        return transpose(tensor1);
    }
    if (tensor1.get_ndim() != dims.size()) {
        throw SailCError("Transpose axes must have same length as tensor");
    }

    TensorShape new_shape = TensorShape(tensor1.get_shape());
    new_shape = new_shape.reorder(dims);
    new_shape.contiguous = false;
    TensorBody::pointer new_body = TensorBody::pointer(new TensorBody(
        tensor1.get_body()->get_data(), tensor1.get_body()->get_dtype(),
        new_shape, /*is_view*/ true));
    Tensor new_tensor = Tensor(new_body, tensor1.requires_grad);
    return new_tensor;
}

Tensor cat(std::vector<Tensor> tensors, const int axis = 0) {
    Tensor out = sail::internal::cat_stub(tensors, axis, 1);
    return out;
}

Tensor stack(std::vector<Tensor> tensors, const int axis = 0) {
    TensorShape check = tensors[0].get_shape();
    for (int i = 0; i < tensors.size(); i++) {
        SAIL_CHECK(check.shape == tensors[i].get_shape().shape,
                   "Shapes must match");
    }
    Tensor out = sail::internal::stack_stub(tensors, axis);
    return out;
}

}  // namespace ops

}  // namespace sail
