#pragma once

#include <iostream>

#include <chrono>
#include "Tensor.h"
#include "autograd/autograd.h"
#include "cross_entropy.h"
#include "factories.h"
#include "kernels/kernel.h"
#include "ops/ops.h"
using namespace std::chrono;

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

    // Tensor log_y = ops::log_softmax(logits);

    Dtype dt = log_y.get_dtype();
    TensorShape shape = log_y.get_shape();

    float coeff = -1.0 / (shape.shape[0]);
    Tensor t_coeff = from_data(&coeff, dt, TensorShape({1}));

    Tensor temp_res = zeros(TensorShape({1}),
                            dt);  // ops::sum(log_y * casted_one_hot_tensor);
    SoftmaxMulSumKernel().execute(log_y, targets, temp_res);
    result = temp_res * t_coeff;

    return result;
}

}  // namespace ops
// namespace ops
}  // namespace sail
