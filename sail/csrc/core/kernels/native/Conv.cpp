// allow-no-header

#include "kernels/Conv.h"
#include "Tensor.h"
#include "dtypes.h"
#include "factories.h"
#include "kernels/dispatch.h"
#include "kernels/native/loops.h"
#include "ops/ops.h"

namespace sail {

namespace internal {

namespace {

Tensor im2col_kernel(Tensor &input, std::vector<long> kernel_size,
                     std::vector<long> stride, std::vector<long> pads) {
    Tensor padded_input = ops::pad(input, {pads});

    return empty_like(input);
}

}  // namespace

REGISTER_ONLY_NATIVE_DISPATCH(im2col_stub, &im2col_kernel);

}  // namespace internal

}  // namespace sail