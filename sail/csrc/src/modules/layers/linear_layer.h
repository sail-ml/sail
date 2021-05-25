#pragma once
#include "../../Tensor.h"
#include "layer.h"

namespace sail {
namespace modules {

class Linear : public Layer {
    Tensor weights;
    Tensor biases;

    long input_features;
    long output_features;
    bool use_bias;

   public:
    Linear(long _input_features, long _output_features, bool _bias = false);

    forward(Tensor& input);
};

}  // namespace modules
}  // namespace sail
