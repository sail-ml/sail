#pragma once
#include "Tensor.h"
#include "modules/module.h"
#include "tensor_shape.h"

#ifdef MKLDNN
#include "onednn/linear.h"
#endif

namespace sail {
namespace modules {

class Linear : public Module {
   public:
    Tensor weights;
    Tensor biases;

    long input_features;
    long output_features;
    long batch_size = 0;
    bool use_bias;

#ifdef MKLDNN
    std::shared_ptr<onednn::OneDNNLinearParams> params = nullptr;
    std::shared_ptr<onednn::OneDNNLinear> layer = nullptr;
    TensorShape output_shape;
#endif

    Linear(long _input_features, long _output_features, bool _bias = true);

    void set_weights(Tensor& new_weights);
    void set_biases(Tensor& new_biases);

    Tensor forward(Tensor& input);
};

}  // namespace modules
}  // namespace sail
