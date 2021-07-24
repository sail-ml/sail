// allow-comments
#include <iostream>

#include "Tensor.h"
#include "dtypes.h"
#include "kernels/utils.h"
#include "ops/ops.h"
#include "tensor_shape.h"

namespace sail {

namespace ops {

Tensor max_pool_2d(Tensor& input, TensorShape kernel_size,
                   std::vector<long> strides, std::vector<long> padding) {
    // long b = input.get_shape()[0];
    // long c = input.get_shape()[1];
    // long h = input.get_shape()[2];
    // long w = input.get_shape()[3];

    // long new_height =
    //     (h + strides[0] * padding[0] - kernel_size[0]) / strides[0] + 1;
    // long new_width =
    //     (w + strides[1] * padding[1] - kernel_size[1]) / strides[1] + 1;

    // auto full_kernel_size = TensorShape({c, c, kernel_size[0], kernel_size[1]});

    return input;
}
}  // namespace ops

}  // namespace sail
