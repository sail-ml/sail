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
TensorVector Pow::backward(Tensor& grad) { throw SailCError("Not yet"); }

}  // namespace autograd
}  // namespace sail