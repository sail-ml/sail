#pragma once
#include "Tensor.h"
#include "modules/module.h"
#include "tensor_shape.h"

#ifdef MKLDNN
#include "onednn/conv2d.h"
#endif

namespace sail {
namespace modules {

class Conv2D : public Module {
   public:
    Tensor weights;
    Tensor biases;

    std::string padding_mode;

    long input_channels;
    long output_channels;
    long batch_size = 0;
    bool use_bias;

    std::vector<long> strides;

#ifdef MKLDNN
    std::shared_ptr<onednn::OneDNNConv2DParams> params = nullptr;
    std::shared_ptr<onednn::OneDNNConv2DBackward> layer2 = nullptr;
    std::shared_ptr<onednn::OneDNNConv2D> layer = nullptr;
    // TensorShape output_shape;
#endif

    Conv2D(long _input_channels, long _output_channels,
           std::vector<long> _kernel_size, std::vector<long> _strides,
           std::string _padding_mode, bool _bias = true);

    Conv2D(long _input_channels, long _output_channels, long _kernel_size,
           long _strides, std::string _padding_mode, bool _bias = true)
        : Conv2D(_input_channels, _output_channels,
                 {_kernel_size, _kernel_size}, {_strides, _strides},
                 _padding_mode, _bias){};

    void set_weights(Tensor& new_weights);
    void set_biases(Tensor& new_biases);

    Tensor forward(Tensor& input);
};

}  // namespace modules
}  // namespace sail
