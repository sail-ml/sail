#include <iostream>

#include "Tensor.h"
#include "autograd/autograd.h"
#include "dtypes.h"
#include "kernels/Kernel.h"
#include "ops/ops.h"
#include "slice.h"
#include "tensor_shape.h"

#include <chrono>
using namespace std::chrono;
namespace sail {

namespace ops {
using TensorVector = std::vector<Tensor>;

Tensor conv2d(Tensor& input, Tensor& kernel, std::vector<long> stride,
              std::string padding_mode) {
    if (input.requires_grad || kernel.requires_grad) {
        TensorVector vec;
        vec.emplace_back(input);
        vec.emplace_back(kernel);
        Tensor empty_tensor =
            (new autograd::Conv2D(stride, padding_mode))->apply(vec);

        return empty_tensor;
    }

    return internal::conv2d_stub(input, kernel, stride, padding_mode)[0];
}

}  // namespace ops

}  // namespace sail
