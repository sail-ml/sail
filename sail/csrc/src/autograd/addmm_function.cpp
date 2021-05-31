#pragma once

#include "addmm_function.h"
#include <iostream>
#include <vector>
#include "../Tensor.h"
#include "../factories.h"
#include "../ops/ops.h"
#include "function.h"

namespace sail {

namespace autograd {

using TensorVector = std::vector<Tensor>;

std::string AddMM::getName() { return "AddMMOp"; }
Tensor AddMM::forward(TensorVector inputs) {
    return ops::matmul(inputs[0], inputs[1]);
}
TensorVector AddMM::backward(Tensor& grad) {
    Tensor a = Function::arg_storage[0];
    Tensor b = Function::arg_storage[1];
    Tensor c_grad = grad;

    bool a_is_vector = a.get_ndim() == 1;
    bool b_is_vector = b.get_ndim() == 1;

    Tensor ga, gb, u, v;

    if (b_is_vector) {
        u = grad;
        v = b;
        if (!a_is_vector) {
            u = ops::expand_dims(u, -1);
            if (v.get_ndim() > 1) {
                v = ops::expand_dims(v, -2);
            }
        }
        ga = u * v;
    } else if (a_is_vector) {
        Tensor bt = ops::rollaxis(b, -2);
        ga = ops::tensordot(bt, grad, grad.get_ndim());
    } else {
        ga = ops::matmul(grad, b.transpose({1, 0}));
        ga = clone(ops::broadcast_to(ga, a.get_shape()));
    }

    if (a_is_vector) {
        u = a;
        v = grad;
        if (!b_is_vector) {
            u = ops::expand_dims(u, -1);
            if (v.get_ndim() > 1) {
                v = ops::expand_dims(v, -2);
            }
        }
        gb = u * v;
    } else if (b_is_vector) {
        Tensor at = ops::rollaxis(a, -2);
        gb = ops::tensordot(at, grad, grad.get_ndim());
    } else {
        gb = ops::matmul(a.transpose({1, 0}), grad);
        gb = clone(ops::broadcast_to(gb, b.get_shape()));
    }

    return {ga, gb, c_grad};
}

}  // namespace autograd
}  // namespace sail