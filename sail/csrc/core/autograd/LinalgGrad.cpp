#pragma once

#include "LinalgGrad.h"
#include <iostream>
#include <vector>
#include "Tensor.h"
#include "factories.h"
#include "function.h"
#include "ops/ops.h"

namespace sail {

namespace autograd {

using TensorVector = std::vector<Tensor>;

Tensor AddMM::forward(TensorVector inputs) {
    return ops::addmm(inputs[0], inputs[1], inputs[2]);
}
TensorVector AddMM::backward(Tensor& grad) {
    Tensor a = Function::arg_storage[0];
    Tensor b = Function::arg_storage[1];

    bool a_is_vector = a.get_ndim() == 1;
    bool b_is_vector = b.get_ndim() == 1;

    Tensor ga, gb, u, v;
    if (a.requires_grad) {
        // DISABLE_GRAD_IND(a)

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
            ga = ops::matmul(grad, b, NO_TRANS, TRANS);
            // ga = ops::matmul(grad, b.transpose({1, 0}), NO_TRANS, NO_TRANS);
            // ga = ops::broadcast_to(ga, a.get_shape());
            // std::cout << ga << std::endl;
        }
        // ENABLE_GRAD_IND(a)
        // ga.requires_grad = true;
    }

    if (b.requires_grad) {
        // DISABLE_GRAD_IND(b)

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
            // gb = ops::matmul(a.transpose({1, 0}), grad, NO_TRANS, NO_TRANS);
            gb = ops::matmul(a, grad, TRANS, NO_TRANS);
            // gb = ops::broadcast_to(gb, b.get_shape());
        }
        // ENABLE_GRAD_IND(b)
        // gb.requires_grad = true;
    }

    return {ga, gb, grad};
}

Tensor Matmul::forward(TensorVector inputs) {
    return ops::matmul(inputs[0], inputs[1], trans_a, trans_b);
}
TensorVector Matmul::backward(Tensor& grad) {
    Tensor a = Function::arg_storage[0];
    Tensor b = Function::arg_storage[1];

    DISABLE_GRAD_IND(a)
    DISABLE_GRAD_IND(b)

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

    ENABLE_GRAD_IND(a)
    ENABLE_GRAD_IND(b)

    return {ga, gb};
}

}  // namespace autograd
}  // namespace sail