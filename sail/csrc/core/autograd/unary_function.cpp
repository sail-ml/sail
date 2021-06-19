#include "unary_function.h"
#include <iostream>
#include <vector>
#include "Tensor.h"
#include "error.h"
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

Tensor ClipMinOnly::forward(TensorVector inputs) {
    return ops::clip(inputs[0], min);
}
TensorVector ClipMinOnly::backward(Tensor& grad) {
    Tensor cond_check =
        from_data((void*)(&min), grad.get_dtype(), TensorShape({1}));
    Tensor cond = ops::elementwise_lte(cond_check, Function::arg_storage[0]);
    return {grad * cond};
}

Tensor Clip::forward(TensorVector inputs) {
    return ops::clip(inputs[0], min, max);
}
TensorVector Clip::backward(Tensor& grad) {
    Tensor cond_check =
        from_data((void*)(&min), grad.get_dtype(), TensorShape({1}));

    Tensor cond = ops::elementwise_lte(cond_check, Function::arg_storage[0]);

    cond_check = from_data((void*)(&max), grad.get_dtype(), TensorShape({1}));
    cond = cond * ops::elementwise_lte(Function::arg_storage[0], cond_check);

    Tensor out = grad * cond;

    return {out};
}

}  // namespace autograd
}  // namespace sail