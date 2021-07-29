#include "ExponentialGrad.h"
#include <iostream>
#include <vector>
#include "Tensor.h"
#include "exception.h"
#include "factories.h"
#include "function.h"
#include "ops/ops.h"

namespace sail {

namespace autograd {

using TensorVector = std::vector<Tensor>;

Tensor Exp::forward(TensorVector inputs) { return ops::exp(inputs[0]); }
TensorVector Exp::backward(Tensor& grad) {
    Tensor t = ops::exp(Function::arg_storage[0]);
    Tensor b = t * grad;
    return {b};
}

Tensor Log::forward(TensorVector inputs) {
    stored_log = ops::log(inputs[0]);
    return stored_log;
}
TensorVector Log::backward(Tensor& grad) {
    return {grad / Function::arg_storage[0]};
}

Tensor Pow::forward(TensorVector inputs) {
    return ops::power(inputs[0], inputs[1]);
}
TensorVector Pow::backward(Tensor& grad) {
    Tensor x0 = Function::arg_storage[0];
    Tensor x1 = Function::arg_storage[1];

    auto g0 = x1 * (ops::power(x0, x1 - 1)) * grad;
    auto g1 = ops::log(x0) * ops::power(x0, x1) * grad;

    auto g01 = ops::broadcast_to(g0, TensorShape(x0.get_shape().shape));
    auto g11 = ops::broadcast_to(g1, TensorShape(x1.get_shape().shape));

    std::vector<Tensor> o = {clone(g01), clone(g11)};

    return o;
}

}  // namespace autograd
}  // namespace sail