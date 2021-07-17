#include "linear_layer.h"
#include <math.h> /* pow */
#include "Tensor.h"
#include "autograd/autograd.h"
#include "dtypes.h"
#include "exception.h"
#include "factories.h"
#include "ops/ops.h"
#include "tensor_shape.h"
// #include "module.h"

#ifdef MKLDNN
#include "onednn/linear.h"
#include "onednn/onednn_utils.h"
#endif

namespace sail {
namespace modules {
using TensorVector = std::vector<Tensor>;

Linear::Linear(long _input_features, long _output_features, bool _bias = true)
    : input_features(_input_features),
      output_features(_output_features),
      use_bias(_bias) {
    double variance = 1.0 / (double)output_features;
    // double variance = 1.0 / pow(((double)output_features), 2.0);
    weights = random::uniform(TensorShape({input_features, output_features}),
                              default_dtype, -variance, variance);
    weights.requires_grad = true;
    register_param(weights);
    if (use_bias) {
        biases = zeros(TensorShape({output_features}), default_dtype);
        biases.requires_grad = true;
        register_param(biases);
    }
}

void Linear::set_weights(Tensor& new_weights) {
    new_weights.requires_grad = true;
    weights = new_weights;
}
void Linear::set_biases(Tensor& new_biases) {
    new_biases.requires_grad = true;
    biases = new_biases;
}

// Linear::~Linear() {
// #ifdef MKLDNN
//     delete layer;
//     delete params;
// #endif
// }

Tensor Linear::forward(Tensor& input) {
    if (input.get_dtype() != Dtype::sFloat32) {
        throw SailCError("Linear only support Float32 data");
    }
    if (use_bias) {
#ifdef MKLDNN
        batch_size = input.get_shape().shape[0];

        output_shape = TensorShape({batch_size, output_features});

        Tensor Tdest = empty(0, Dtype::sFloat32, output_shape);
        auto L = onednn::LinearFactory(input, weights, biases, Tdest);

        L.forward();

        TensorVector vec;
        vec.emplace_back(input);
        vec.emplace_back(weights);
        vec.emplace_back(biases);

        autograd::Function* fcn = (new autograd::AddMM());
        fcn->apply_no_forward(vec);

        fcn->set_fcn(Tdest);

        return Tdest;

#endif

        return ops::addmm(input, weights, biases);

    } else {
        Tensor res = ops::matmul(input, weights);
        return res;
    }
}

}  // namespace modules
}  // namespace sail
