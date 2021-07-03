#include <iostream>

#include "Tensor.h"
#include "dtypes.h"
#include "kernels/Kernel.h"
#include "ops/ops.h"
#include "slice.h"
#include "tensor_shape.h"

namespace sail {

namespace ops {

Tensor conv2d(Tensor& input, Tensor& kernel, std::vector<long> stride,
              std::string padding_mode = "same") {
    Tensor im2col_input = input;

    long d = 1;

    long pad_y = 0;
    long pad_x = 0;

    long c = input.get_shape()[0];
    long h = input.get_shape()[1];
    long w = input.get_shape()[2];

    long k_cin = kernel.get_shape()[0];
    long k_cout = kernel.get_shape()[1];
    long k_h = kernel.get_shape()[2];
    long k_w = kernel.get_shape()[3];

    SAIL_CHECK_LINE(k_cin == c);

    if (padding_mode == "same") {
        pad_y =
            (long)(((1 - (float)d - (float)stride[0] + (float)k_h * (float)d) /
                    2) +
                   (float)h * ((-1 + (float)stride[0]) / 2));
        pad_x =
            (long)(((1 - (float)d - (float)stride[1] + (float)k_w * (float)d) /
                    2) +
                   (float)w * ((-1 + (float)stride[1]) / 2));
        im2col_input =
            ops::pad(im2col_input, {{0, 0}, {pad_x, pad_y}, {pad_x, pad_y}});
    }

    long new_height = (h + 2 * pad_y - k_h) / stride[0] + 1;
    long new_width = (w + 2 * pad_x - k_w) / stride[1] + 1;

    // print out new_height and new_width
    Tensor cols =
        empty(0, input.get_dtype(),
              TensorShape({new_height * new_width, k_cin * k_w * k_h}));

    long z = 0;
    int input_i = 0;
    int input_j = 0;
    for (int i = 0; i < new_height; i += 1) {
        input_j = 0;
        for (int j = 0; j < new_width; j += 1) {
            Slice s =
                Slice({{}, {input_i, k_h + input_i}, {input_j, input_j + k_w}});
            s.print();
            Tensor t = im2col_input.slice(s);
            t = t.reshape(TensorShape({1, k_cin * k_w * k_h}));
            cols.slice(Slice({z, z + 1})).assign(t);
            z += 1;
            input_j += stride[1];
        }
        input_i += stride[0];
        // z += stride[0] - 1;
    }

    Tensor flat_kernel =
        kernel.reshape(TensorShape({k_cout, k_cin * k_w * k_h}));

    Tensor res = ops::matmul(cols, flat_kernel, "N", "T");
    res._inplace_reshape(TensorShape({k_cout, new_height, new_width}));

    return res;
    // return sail::internal::im2col_stub(x, w.get_shape().shape, stride, pad);
}
}  // namespace ops

}  // namespace sail
