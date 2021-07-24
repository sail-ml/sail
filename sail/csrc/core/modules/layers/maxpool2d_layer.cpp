#include "maxpool2d_layer.h"
#include <cmath>
#include "Tensor.h"
#include "autograd/autograd.h"
#include "dtypes.h"
#include "exception.h"
#include "factories.h"
#include "initializers/kaiming.h"
#include "kernels/utils.h"
#include "ops/ops.h"
#include "tensor_shape.h"

#ifdef MKLDNN
#include "onednn/pooling.h"
#endif
namespace sail {
namespace modules {
using TensorVector = std::vector<Tensor>;

Tensor MaxPool2D::forward(Tensor& input) {
#ifdef MKLDNN
    std::vector<long> padding;

    std::vector<long> padding_r;
    std::vector<long> padding_l;

    auto nh_nw =
        calculate_nh_nw(input.get_shape(), kernel_size, strides, padding_mode);
    auto new_height = nh_nw[0];
    auto new_width = nh_nw[1];

    if (padding_mode == "same") {
        THROW_ERROR(SailCError, "Same padding not supported");

    } else {
        padding_l = {0, 0};
        padding_r = {0, 0};
    }

    long batch_size = input.get_shape().shape[0];

    TensorShape output_shape =
        TensorShape({batch_size, input.get_shape()[1], new_height, new_width});

    params.reset(new onednn::OneDNNMaxPoolingParams(
        input, TensorShape(kernel_size), output_shape, strides, padding_l,
        padding_r));
    layer.reset(new onednn::OneDNNMaxPooling(params));

    auto desc = layer->initialize();
    Tensor Tdest = empty(0, Dtype::sFloat32, output_shape);

    TensorVector vec;
    vec.emplace_back(input);

    autograd::Function* fcn = (new autograd::MaxPool2D(params, desc));
    fcn->apply_no_forward(vec);

    fcn->set_fcn(Tdest);

    layer->add_src_dest_data(input.get_data(), Tdest.get_data());
    layer->forward();

    return Tdest;

#endif
    return ops::max_pool_2d(input, TensorShape(kernel_size), strides, padding);
}

}  // namespace modules
}  // namespace sail
