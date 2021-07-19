#include <iostream>

#include "CompareOps.h"
#include "Tensor.h"
#include "autograd/autograd.h"
#include "factories.h"
#include "kernels/Kernel.h"
#include "ops/tools.h"

#define MAX(a, b) (((a.ndim) > (b.ndim)) ? (a) : (b))
#define MIN(a, b) (((a.ndim) < (b.ndim)) ? (a) : (b))

namespace sail {

namespace ops {
using TensorVector = std::vector<Tensor>;

Tensor relu(const Tensor& tensor1) {
    Tensor empty_tensor;
    if (tensor1.requires_grad) {
        TensorVector vec;
        vec.emplace_back(tensor1);
        empty_tensor = (new autograd::ReLU())->apply(vec);
        return empty_tensor;
    }

    empty_tensor = empty_like(tensor1);
    sail::internal::clip_min_stub(tensor1, 0.0, empty_tensor);
    return empty_tensor;
}

Tensor sigmoid(const Tensor& tensor1) {
    Tensor empty_tensor;
    if (tensor1.requires_grad) {
        TensorVector vec;
        vec.emplace_back(tensor1);
        empty_tensor = (new autograd::Sigmoid())->apply(vec);
        return empty_tensor;
    }

    empty_tensor = empty_like(tensor1);
    sail::internal::sigmoid_stub(tensor1, empty_tensor);
    return empty_tensor;
}

Tensor softmax(Tensor& tensor1, const int axis) {
    Tensor empty_tensor;
    if (tensor1.requires_grad) {
        TensorVector vec;
        vec.emplace_back(tensor1);
        empty_tensor = (new autograd::Softmax(axis))->apply(vec);
        return empty_tensor;
    }

    empty_tensor = empty_like(tensor1);
    sail::internal::softmax_stub(tensor1, axis, empty_tensor);
    return empty_tensor;
}

Tensor log_softmax(Tensor& input, const int axis) {
    if (input.requires_grad) {
        TensorVector vec;
        vec.emplace_back(input);
        return (new autograd::Softmax(axis))->apply(vec);
    }

    Tensor max = ops::max(input, axis, true);
    input = input - max;
    input = ops::exp(input);
    Tensor s = ops::sum(input, axis, true);
    input = input - ops::log(s);
    return input;
}

}  // namespace ops

}  // namespace sail
