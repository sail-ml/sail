
#include "MutationGrad.h"
#include <iostream>
#include <vector>
#include "Tensor.h"
#include "exception.h"
#include "factories.h"
#include "function.h"
#include "kernels/Kernel.h"
#include "ops/ops.h"
#include "tensor_shape.h"

namespace sail {

namespace autograd {

using TensorVector = std::vector<Tensor>;

Tensor Reshape::forward(TensorVector inputs) {
    original_shape = inputs[0].get_shape();
    Tensor return_ = inputs[0].reshape(new_shape);
    return return_;
}
TensorVector Reshape::backward(Tensor& grad) {
    auto grad2 = grad.reshape(original_shape);
    return {grad2};
}

}  // namespace autograd
}  // namespace sail