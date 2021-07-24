#include <iostream>

#include "Tensor.h"
#include "UnaryOps.h"
#include "autograd/autograd.h"
#include "factories.h"
#include "kernels/Kernel.h"
#include "ops/tools.h"

#define MAX(a, b) (((a.ndim) > (b.ndim)) ? (a) : (b))
#define MIN(a, b) (((a.ndim) < (b.ndim)) ? (a) : (b))

namespace sail {

namespace ops {
using TensorVector = std::vector<Tensor>;

Tensor negate(const Tensor& tensor1) {
    if (tensor1.requires_grad) {
        TensorVector vec;
        vec.emplace_back(tensor1);
        Tensor empty_tensor = (new autograd::Negate())->apply(vec);

        return empty_tensor;
    }
    Tensor empty_tensor = empty_like(tensor1);
    sail::internal::negate_stub(tensor1, empty_tensor);
    return empty_tensor;
}

}  // namespace ops

}  // namespace sail
