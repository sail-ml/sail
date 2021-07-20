#include "UnaryGrad.h"
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

Tensor ClipMinOnly::forward(TensorVector inputs) {
    return ops::clip_min(inputs[0], min);
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