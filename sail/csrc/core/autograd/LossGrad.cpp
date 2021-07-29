

#pragma once

#include "LossGrad.h"
#include <iostream>
#include <vector>
#include "Tensor.h"
#include "factories.h"
#include "function.h"
#include "kernels/Kernel.h"
#include "ops/ops.h"

namespace sail {

namespace autograd {

using TensorVector = std::vector<Tensor>;

Tensor SoftmaxCrossEntropyLoss::forward(TensorVector inputs) {
    return ops::softmax_cross_entropy(inputs[0], inputs[1]);
}
TensorVector SoftmaxCrossEntropyLoss::backward(Tensor& grad) {
    Tensor y = ops::softmax(Function::arg_storage[0]);
    Tensor z = empty(0, y.get_dtype(), y.get_shape());
    sail::internal::softmax_backward_partial_stub(y, Function::arg_storage[1],
                                                  z);
    return {z};
}

Tensor MeanSquaredErrorLoss::forward(TensorVector inputs) {
    return ops::mean_squared_error(inputs[0], inputs[1]);
}
TensorVector MeanSquaredErrorLoss::backward(Tensor& grad) {
    Tensor diff = Function::arg_storage[0] - Function::arg_storage[1];
    grad = ops::broadcast_to(grad, diff.get_shape());

    void* data =
        _malloc_align(1, grad.get_info().alignment, grad.get_info().dtype_size);
    dispatch_all_numeric_types(grad.get_dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        T inner_data = 2.0 / (T)diff.numel();
        T* d = static_cast<T*>(data);
        d[0] = inner_data;
    });

    Tensor v_ = from_data(data, grad.get_dtype(), TensorShape({1}));

    grad = grad * diff * v_;
    return {grad, -grad};
}

}  // namespace autograd
}  // namespace sail