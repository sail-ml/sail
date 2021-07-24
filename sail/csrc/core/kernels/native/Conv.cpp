// allow-no-header

#include "kernels/Conv.h"
#include "Tensor.h"
#include "dtypes.h"
#include "factories.h"
#include "kernels/dispatch.h"
#include "kernels/native/loops.h"
#include "kernels/utils.h"
#include "ops/ops.h"

namespace sail {

namespace internal {

namespace {

std::vector<Tensor> conv2d_kernel(Tensor& input, Tensor& kernel,
                                  std::vector<long> strides,
                                  std::string padding_mode) {
    long b = input.get_shape()[0];

    long k_cout = kernel.get_shape()[0];
    long k_cin = kernel.get_shape()[1];
    long k_h = kernel.get_shape()[2];
    long k_w = kernel.get_shape()[3];

    auto nh_nw = calculate_nh_nw(input.get_shape(), kernel.get_shape(), strides,
                                 padding_mode);

    auto new_height = nh_nw[0];
    auto new_width = nh_nw[1];

    auto cols2 = im2col(input, kernel.get_shape(), strides, padding_mode);
    auto cols = sail::ops::reshape(
        cols2,
        sail::TensorShape({new_height * new_width * b, k_cin * k_w * k_h}));

    Tensor flat_kernel =
        kernel.reshape(TensorShape({k_cout, k_cin * k_w * k_h}));

    Tensor res = ops::matmul(cols, flat_kernel, "N", "T");
    res._inplace_reshape(TensorShape({b, k_cout, new_height, new_width}));

    std::vector<Tensor> first_vec = {res, cols, flat_kernel};
    return first_vec;
}

}  // namespace

REGISTER_ONLY_NATIVE_DISPATCH(conv2d_stub, &conv2d_kernel);

}  // namespace internal

}  // namespace sail