#include "conv2d_layer.h"
#include <cmath>
#include "Tensor.h"
#include "autograd/autograd.h"
#include "dtypes.h"
#include "exception.h"
#include "factories.h"
#include "initializers/kaiming.h"
#include "kernels/utils.h"
#include "ops/ops.h"
#include "tensor_shape.h"

#ifdef MKLDNN
#include "onednn/conv2d_forward.h"
#endif
namespace sail {
namespace modules {
using TensorVector = std::vector<Tensor>;

Conv2D::Conv2D(long _input_channels, long _output_channels,
               std::vector<long> _kernel_size, std::vector<long> _strides,
               std::string _padding_mode, bool _bias) {
    input_channels = _input_channels;
    output_channels = _output_channels;

    strides = _strides;
    weights = empty(0, default_dtype,
                    TensorShape({output_channels, input_channels,
                                 _kernel_size[0], _kernel_size[1]}));
    weights.requires_grad = true;
    weights = initializers::kaiming_uniform(weights);
    register_param(weights);
    padding_mode = _padding_mode;
    if (_bias) {
        biases = zeros(TensorShape({output_channels}), default_dtype);
        biases.requires_grad = true;
        register_param(biases);
    }
    use_bias = _bias;
}

void Conv2D::set_weights(Tensor& new_weights) {
    new_weights.requires_grad = true;
    weights = new_weights;
}
void Conv2D::set_biases(Tensor& new_biases) {
    new_biases.requires_grad = true;
    biases = new_biases;
}

Tensor Conv2D::forward(Tensor& input) {
#ifdef MKLDNN
    std::vector<long> padding_r;
    std::vector<long> padding_l;

    auto nh_nw = calculate_nh_nw(input.get_shape(), weights.get_shape(),
                                 strides, padding_mode);
    auto new_height = nh_nw[0];
    auto new_width = nh_nw[1];

    if (padding_mode == "same") {
        auto padding = calc_2d_same_padding(weights.get_shape());
        padding_l = padding[0];
        padding_r = padding[1];
    } else {
        padding_l = {0, 0};
        padding_r = {0, 0};
    }

    long _batch_size = input.get_shape().shape[0];

    batch_size = _batch_size;
    TensorShape output_shape = TensorShape(
        {batch_size, weights.get_shape()[0], new_height, new_width});

    Tensor Tdest = empty(0, Dtype::sFloat32, output_shape);

    if (use_bias) {
        auto L = onednn::Conv2DForwardFactory(input, weights, biases, Tdest,
                                              strides, padding_l, padding_r);

        L.forward();
    } else {
        auto L = onednn::Conv2DForwardFactory(input, weights, Tdest, strides,
                                              padding_l, padding_r);

        L.forward();
    }

    TensorVector vec;
    vec.emplace_back(input);
    vec.emplace_back(weights);
    if (use_bias) {
        vec.emplace_back(biases);
    }

    autograd::Function* fcn =
        (new autograd::Conv2DMKLDNN(padding_l, padding_r, strides));
    fcn->apply_no_forward(vec);

    fcn->set_fcn(Tdest);

    return Tdest;

#endif
    return ops::conv2d(input, weights, strides, padding_mode);
}

}  // namespace modules
}  // namespace sail
