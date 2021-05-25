#include "linear_layer.h"
#include "../../Tensor.h"

namespace sail {
namespace modules {

Linear::Linear(long _input_features, long _output_features, bool _bias = false)
    : input_features(_input_features),
      output_features(_output_features),
      use_bias(_bias) {}
}  // namespace modules
}  // namespace sail
