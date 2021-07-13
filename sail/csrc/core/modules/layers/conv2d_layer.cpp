#include "conv2d_layer.h"
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
#include "onednn/conv2d.h"
#endif
namespace sail {
namespace modules {
using TensorVector = std::vector<Tensor>;

Conv2D::Conv2D(long _input_channels, long _output_channels,
               std::vector<long> _kernel_size, std::vector<long> _strides,
               std::string _padding_mode, bool _bias = true) {
    strides = _strides;
    weights = empty(0, default_dtype,
                    TensorShape({_output_channels, _input_channels,
                                 _kernel_size[0], _kernel_size[1]}));
    weights.requires_grad = true;
    weights = initializers::kaiming_uniform(weights);
    register_param(weights);
    padding_mode = _padding_mode;
    if (_bias) {
        biases = zeros(TensorShape({_output_channels}), default_dtype);
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
    std::vector<long> padding;
    long new_width, new_height;
    if (padding_mode == "same") {
        long pad_y = (long)(((1 - (float)1 - (float)strides[0] +
                              (float)weights.get_shape().shape[2] * (float)1) /
                             2) +
                            (float)input.get_shape().shape[2] *
                                ((-1 + (float)strides[0]) / 2));
        long pad_x = (long)(((1 - (float)1 - (float)strides[1] +
                              (float)weights.get_shape().shape[3] * (float)1) /
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

        long k_h = weights.get_shape()[2];
        long k_w = weights.get_shape()[3];

        new_height =
            (input.get_shape()[0] + 2 * 0 - 1 * (k_h - 1)) / strides[0] + 1;
        new_width =
            (input.get_shape()[1] + 2 * 0 - 1 * (k_w - 1)) / strides[1] + 1;

        // calc nh nw
    }

    long _batch_size = input.get_shape().shape[0];

    // if (_batch_size != batch_size) {
    batch_size = _batch_size;
    TensorShape output_shape = TensorShape(
        {batch_size, weights.get_shape()[0], new_height, new_width});

    params.reset(new onednn::OneDNNConv2DParams(input, weights, output_shape,
                                                strides, padding));
    layer.reset(new onednn::OneDNNConv2D(params));
    // layer2.reset(new onednn::OneDNNConv2DBackward(params));
    // }
    auto x = layer->initialize(use_bias);
    Tensor Tdest = empty(0, Dtype::sFloat32, output_shape);

    TensorVector vec;
    vec.emplace_back(input);
    vec.emplace_back(weights);
    if (use_bias) {
        vec.emplace_back(biases);
    }

    autograd::Function* fcn =
        (new autograd::Conv2DMKLDNN(padding[0], padding[1], strides, x));
    fcn->apply_no_forward(vec);

    fcn->set_fcn(Tdest);

    void* biases_ = nullptr;
    if (use_bias) {
        biases_ = biases.get_data();
    }
    layer->add_base_data(weights.get_data(), biases_);
    layer->add_src_dest_data(input.get_data(), Tdest.get_data());
    layer->forward();

    return Tdest;

#endif
    return ops::conv2d(input, weights, strides);
}

}  // namespace modules
}  // namespace sail
