#include <iostream>

#include "LossOps.h"
#include "Tensor.h"
#include "autograd/autograd.h"
#include "factories.h"
#include "kernels/Kernel.h"
#include "ops/ops.h"

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

    Dtype dt = log_y.get_dtype();
    TensorShape shape = log_y.get_shape();

    float coeff = -1.0 / (shape.shape[0]);
    Tensor t_coeff = from_data(&coeff, dt, TensorShape({1}));

    Tensor temp_res = zeros(TensorShape({1}), dt);
    sail::internal::softmax_mul_sum_stub(log_y, targets, temp_res);
    result = temp_res * t_coeff;

    return result;
}
Tensor mean_squared_error(Tensor& logits, Tensor& targets) {
    Tensor result;
    if (logits.requires_grad) {
        TensorVector vec;
        vec.emplace_back(logits);
        vec.emplace_back(targets);
        result = (new autograd::MeanSquaredErrorLoss())
                     ->apply(vec);  //{std::make_shared<Tensor>(tensor1)});
        return result;
    }

    Tensor out = empty(0, logits.get_dtype(), logits.get_shape());
    sail::internal::mse_stub(logits, targets, out);
    return ops::sum(out);
}
}  // namespace ops
}  // namespace sail
