#include <iostream>

#include <chrono>
#include "Tensor.h"
#include "autograd/autograd.h"
#include "factories.h"
#include "kernels/kernel.h"
#include "mean_squared_error.h"
#include "ops/ops.h"
using namespace std::chrono;

namespace sail {
namespace ops {
using TensorVector = std::vector<Tensor>;

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
    MeanSquaredErrorKernel().execute(logits, targets, out);
    return ops::sum(out);
}

}  // namespace ops
// namespace ops
}  // namespace sail
