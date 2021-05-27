#pragma once
#include "../../Tensor.h"
#include "../module.h"

namespace sail {
namespace modules {

class Linear : public Module {
   public:
    Tensor weights;
    Tensor biases;

    long input_features;
    long output_features;
    bool use_bias;

    Linear(long _input_features, long _output_features, bool _bias = false);

    Tensor forward(Tensor& input);
};

}  // namespace modules
}  // namespace sail
