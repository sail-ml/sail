#pragma once

#include <iostream>

#include "../../Tensor.h"
#include "../../autograd/autograd.h"
#include "../../factories.h"
#include "../../kernels/kernel.h"
#include "../ops.h"
#include "cross_entropy.h"

namespace sail {
namespace ops {
using TensorVector = std::vector<Tensor>;

Tensor softmax_cross_entropy(Tensor& logits, Tensor& targets) {
    Tensor result;
    if (logits.requires_grad) {
        TensorVector vec;
        vec.emplace_back(logits);
        vec.emplace_back(targets);
        result = (new autograd::SoftmaxCrossEntropyLoss())
                     ->apply(vec);  //{std::make_shared<Tensor>(tensor1)});
        return result;
    }
    Tensor softmax = ops::softmax(logits);
    Tensor log_y = ops::log(softmax);

    Tensor one_hot_tensor = one_hot(targets, logits.get_shape().shape[1]);
    Tensor casted_one_hot_tensor = ops::cast(one_hot_tensor, log_y.get_dtype());

    result = ops::sum(-(log_y * casted_one_hot_tensor));
    return result;
}

}  // namespace ops
}  // namespace sail
