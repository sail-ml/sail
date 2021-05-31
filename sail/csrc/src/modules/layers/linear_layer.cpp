#include "linear_layer.h"
#include <math.h> /* pow */
#include "../../Tensor.h"
#include "../../dtypes.h"
#include "../../factories.h"
#include "../../ops/ops.h"
#include "../../tensor_shape.h"
// #include "../module.h"

namespace sail {
namespace modules {

Linear::Linear(long _input_features, long _output_features, bool _bias = false)
    : input_features(_input_features),
      output_features(_output_features),
      use_bias(_bias) {
    double variance = 1.0 / ((double)output_features);
    weights = random::uniform(TensorShape({input_features, output_features}),
                              default_dtype, -variance, variance);
    if (use_bias) {
        biases = zeros(TensorShape({output_features}), default_dtype);
    }
}

Tensor Linear::forward(Tensor& input) {
    // Tensor mm_res = ops::matmul(input, weights);
    // if (use_bias) {
    //     mm_res = mm_res + biases;
    // }
    // return mm_res;

    if (use_bias) {
        return ops::addmm(input, weights, biases);
    } else {
        return ops::matmul(input, weights);
    }
}

}  // namespace modules
}  // namespace sail
