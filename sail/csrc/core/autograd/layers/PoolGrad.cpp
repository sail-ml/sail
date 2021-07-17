

#pragma once

#include "PoolGrad.h"
#include <iostream>
#include <vector>
#include "Tensor.h"
#include "autograd/function.h"
#include "factories.h"
#include "kernels/Kernel.h"
#include "ops/ops.h"

#ifdef MKLDNN
#include "onednn/pooling.h"
#endif

namespace sail {

namespace autograd {

using TensorVector = std::vector<Tensor>;

#ifdef MKLDNN
Tensor MaxPool2D::forward(TensorVector inputs) {
    THROW_ERROR_DETAILED(SailCError, "Should not call this function");
}
TensorVector MaxPool2D::backward(Tensor& grad) {
    if (grad.is_view()) {
        grad = clone(grad);
    }
    Tensor input = Function::arg_storage[0];

    onednn::OneDNNMaxPoolingBackward layer =
        onednn::OneDNNMaxPoolingBackward(params);
    layer.store_forward_desc(desc);
    layer.initialize();

    Tensor out = empty_like(input);

    layer.add_grad_data(grad.get_data());
    layer.add_src_grad_loc(out.get_data());

    layer.forward();

    return {out};
}

#endif

}  // namespace autograd
}  // namespace sail