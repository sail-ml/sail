#pragma once

#include <iostream>

#include "../Tensor.h"
#include "../factories.h"
#include "../kernels/kernel.h"
#include "math/tools.h"

namespace sail {
namespace ops {

Tensor elementwise_equal(const Tensor& tensor1, const Tensor& tensor2) {
    Tensor empty_tensor;  //= empty_like(tensor1);
    Tensor t1, t2;

    Dtype dt = tensor1.get_dtype();

    bool broadcast = must_broadcast(tensor1, tensor2);
    if (broadcast) {
        std::vector<long> new_ =
            merge_shapes(tensor1.get_shape().shape, tensor2.get_shape().shape);
        TensorShape s = TensorShape(new_);
        empty_tensor = empty(s.ndim(), dt, s);
        empty_tensor.requires_grad = t1.requires_grad;
        t1 = ops::broadcast_to(tensor1, s);
        t2 = ops::broadcast_to(tensor2, s);
    } else {
        t1 = tensor1;
        t2 = tensor2;
        empty_tensor = empty_like(t1);
    }

    ElementwiseEquality().execute(t1, t2, empty_tensor, broadcast);
    return empty_tensor;
}

}  // namespace ops
}  // namespace sail
