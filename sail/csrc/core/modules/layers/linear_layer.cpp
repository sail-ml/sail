#include "linear_layer.h"
#include <math.h> /* pow */
#include "../../Tensor.h"
#include "../../autograd/autograd.h"
#include "../../dtypes.h"
#include "../../error.h"
#include "../../factories.h"
#include "../../ops/ops.h"
#include "../../tensor_shape.h"
// #include "../module.h"

#ifdef MKLDNN
#include "../../onednn/linear.h"
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
        long _batch_size = input.get_shape().shape[0];
        if (_batch_size != batch_size) {
            batch_size = _batch_size;
            output_shape = TensorShape({batch_size, output_features});

            // if (params != nullptr) {
            //     delete params;
            //     delete layer;
            // }
            params.reset(new onednn::OneDNNLinearParams(input, input_features,
                                                        output_features));
            layer.reset(new onednn::OneDNNLinear(params));

            layer->initialize();
            // layer->add_base_data(weights.get_data(), biases.get_data());
        }

        Tensor Tdest = empty(0, Dtype::sFloat32, output_shape);

        TensorVector vec;
        vec.emplace_back(input);
        vec.emplace_back(weights);
        vec.emplace_back(biases);

        autograd::Function* fcn = (new autograd::AddMM());
        fcn->apply_no_forward(vec);

        fcn->set_fcn(Tdest);

        layer->add_base_data(weights.get_data(), biases.get_data());
        layer->add_src_dest_data(input.get_data(), Tdest.get_data());

        layer->forward();
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
