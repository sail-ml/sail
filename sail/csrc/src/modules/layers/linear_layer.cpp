#include "linear_layer.h"
#include <math.h> /* pow */
#include "../../Tensor.h"
#include "../../dtypes.h"
#include "../../factories.h"
#include "../../ops/ops.h"
#include "../../tensor_shape.h"

namespace sail {
namespace modules {

Linear::Linear(long _input_features, long _output_features, bool _bias = false)
    : input_features(_input_features),
      output_features(_output_features),
      use_bias(_bias) {
    double variance = 1.0 / pow((double)output_features, 0.5);
    weights = random::uniform(TensorShape({input_features, output_features}),
                              Dtype::sFloat64, -variance, variance);
    if (use_bias) {
        biases = zeros(TensorShape({output_features}), Dtype::sFloat64);
    }
}

Tensor Linear::forward(Tensor& input) {
    Tensor mm_res = ops::matmul(input, weights);
    if (use_bias) {
        mm_res = mm_res + biases;
    }
    return mm_res;
}

}  // namespace modules
}  // namespace sail
