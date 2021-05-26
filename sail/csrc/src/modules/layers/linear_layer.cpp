#include "linear_layer.h"
#include "../../Tensor.h"
#include "../../factories.h"
#include "../../dtypes.h"
#include "../../tensor_shape.h"

namespace sail {
namespace modules {

Linear::Linear(long _input_features, long _output_features, bool _bias = false)
    : input_features(_input_features),
      output_features(_output_features),
      use_bias(_bias) {
          double variance = 1.0 / (output_features ** 0.5);
          weights = random::uniform(TensorShape({input_features, output_features}), Dtype::sFloat64, -variance, variance);
          if (use_bias) {
              bias = zeros(TensorShape({output_features}), Dtype::sFloat64);
          }
      }
}  // namespace modules
}  // namespace sail
