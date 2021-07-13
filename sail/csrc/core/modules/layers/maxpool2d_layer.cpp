#include "maxpool2d_layer.h"
#include <math.h> /* pow */
#include "Tensor.h"
#include "autograd/autograd.h"
#include "dtypes.h"
#include "exception.h"
#include "factories.h"
#include "initializers/kaiming.h"
#include "ops/ops.h"
#include "tensor_shape.h"
// #include "module.h"
#ifdef MKLDNN
#include "onednn/pooling.h"
#endif
namespace sail {
namespace modules {
using TensorVector = std::vector<Tensor>;

Tensor MaxPool2D::forward(Tensor& input) {
#ifdef MKLDNN
    std::vector<long> padding;
    long new_width, new_height;
    if (padding_mode == "same") {
        long pad_y = (long)(((1 - (float)1 - (float)strides[0] +
                              (float)kernel_size[0] * (float)1) /
                             2) +
                            (float)input.get_shape().shape[2] *
                                ((-1 + (float)strides[0]) / 2));
        long pad_x = (long)(((1 - (float)1 - (float)strides[1] +
                              (float)(float)kernel_size[1] * (float)1) /
                             2) +
                            (float)input.get_shape().shape[3] *
                                ((-1 + (float)strides[1]) / 2));
        padding.push_back(pad_y);
        padding.push_back(pad_x);
        new_width = input.get_shape()[3];
        new_height = input.get_shape()[2];
    } else {
        padding.push_back(0);
        padding.push_back(0);

        long k_h = kernel_size[0];
        long k_w = kernel_size[1];
        long h = input.get_shape()[2];
        long w = input.get_shape()[3];

        new_height = (h + 2 * 0 - 1 * (k_h - 1) - 1) / strides[0] + 1;
        new_width = (w + 2 * 0 - 1 * (k_w - 1) - 1) / strides[1] + 1;
    }

    long batch_size = input.get_shape().shape[0];

    // if (_batch_size != batch_size) {
    TensorShape output_shape =
        TensorShape({batch_size, input.get_shape()[1], new_height, new_width});

    params.reset(new onednn::OneDNNMaxPoolingParams(
        input, TensorShape(kernel_size), output_shape, strides, padding));
    layer.reset(new onednn::OneDNNMaxPooling(params));

    auto desc = layer->initialize();
    Tensor Tdest = empty(0, Dtype::sFloat32, output_shape);

    TensorVector vec;
    vec.emplace_back(input);

    autograd::Function* fcn = (new autograd::MaxPool2D(params, desc));
    fcn->apply_no_forward(vec);

    fcn->set_fcn(Tdest);

    layer->add_src_dest_data(input.get_data(), Tdest.get_data());
    layer->forward();

    return Tdest;

#endif
    return ops::max_pool_2d(input, TensorShape(kernel_size), strides, padding);
}

}  // namespace modules
}  // namespace sail
