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
    THROW_ERROR(SailCError, "Maxpooling requires Intel OneAPI");
}
}  // namespace ops

}  // namespace sail
