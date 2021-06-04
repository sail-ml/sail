#pragma once

#include <iostream>

#include "../../Tensor.h"
#include "../../autograd/autograd.h"
#include "../../dtypes.h"
#include "../../kernels/kernel.h"
#include "pow.h"
#include "tools.h"

namespace sail {

namespace ops {
using TensorVector = std::vector<Tensor>;

Tensor power(Tensor& tensor1, Tensor& tensor2) {
    if (tensor1.requires_grad) {
        TensorVector vec;
        vec.emplace_back(tensor1);
        vec.emplace_back(tensor2);
        Tensor empty_tensor = (new autograd::Pow())->apply(vec);
        return empty_tensor;
    }
    Tensor empty_tensor = empty_like(tensor1);
    bool broadcast = must_broadcast(tensor1, tensor2);
    if (broadcast) {
        std::vector<long> new_ =
            merge_shapes(tensor1.get_shape().shape, tensor2.get_shape().shape);
        TensorShape s = TensorShape(new_);
        empty_tensor.set_shape(s);
    }
    PowerKernel().execute(tensor1, tensor2, empty_tensor, broadcast);
    return empty_tensor;
}
Tensor exp(Tensor& tensor1) {
    if (tensor1.requires_grad) {
        TensorVector vec;
        vec.emplace_back(tensor1);
        Tensor empty_tensor = (new autograd::Exp())->apply(vec);
        return empty_tensor;
    }
    Tensor empty_tensor = empty_like(tensor1);
    PowerExpKernel().execute(tensor1, empty_tensor);
    return empty_tensor;
}

}  // namespace ops

}  // namespace sail