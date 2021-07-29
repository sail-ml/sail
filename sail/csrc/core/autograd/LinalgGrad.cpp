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

Tensor inner_backward(Tensor u, Tensor v, std::string trans,
                      bool is_other_vector, bool transpose = false) {
    Tensor nu = u;
    Tensor nv = v;
    if (!is_other_vector) {
        if (trans == TRANS) {
            auto swap = nu;
            nu = nv;
            nv = swap;
        }
        nu = clone(ops::expand_dims(nu, -1));
        if (nv.get_ndim() > 1) {
            nv = clone(ops::expand_dims(nv, -2));
        }
    }
    auto out = clone(nu * nv);
    if (transpose) {
        if (out.get_ndim() == 3) {
            out = clone(out.transpose({1, 0, 2}));
        } else {
            out = clone(out.transpose());
        }
    }
    return out;
}

Tensor AddMM::forward(TensorVector inputs) {
    return ops::addmm(inputs[0], inputs[1], inputs[2]);
}
TensorVector AddMM::backward(Tensor& grad) {
    auto trans_a = "N";
    auto trans_b = "N";
    Tensor a = Function::arg_storage[0];
    Tensor b = Function::arg_storage[1];

    bool a_is_vector = a.get_ndim() == 1;
    bool b_is_vector = b.get_ndim() == 1;

    Tensor ga, gb, u, v;

    if (b_is_vector) {
        ga = inner_backward(grad, b, trans_a, a_is_vector);
    } else if (a_is_vector) {
        long roll = trans_b == TRANS ? -1 : -2;
        Tensor bt = ops::rollaxis(b, roll);
        ga = ops::matmul(bt, grad);
    } else {
        auto use_trans = trans_b;  // NOLINT
        if (trans_b == TRANS) {
            use_trans = NO_TRANS;
        } else {
            use_trans = TRANS;
        }
        ga = ops::matmul(grad, b, NO_TRANS, use_trans);
        ga = clone(ops::broadcast_to(ga, a.get_shape()));
    }

    if (a_is_vector) {
        gb = inner_backward(a, grad, trans_b, b_is_vector, true);
    } else if (b_is_vector) {
        long roll = trans_a == TRANS ? -2 : -1;
        Tensor at = ops::rollaxis(a, roll);
        gb = ops::matmul(at, grad);
    } else {
        auto use_trans = trans_a;  // NOLINT
        if (trans_a == TRANS) {
            use_trans = NO_TRANS;
        } else {
            use_trans = TRANS;
        }
        gb = ops::matmul(a, grad, use_trans, NO_TRANS);
        gb = clone(ops::broadcast_to(gb, b.get_shape()));
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
        ga = inner_backward(grad, b, trans_a, a_is_vector);
    } else if (a_is_vector) {
        long roll = trans_b == TRANS ? -1 : -2;
        Tensor bt = ops::rollaxis(b, roll);
        ga = ops::matmul(bt, grad);
    } else {
        auto use_trans = trans_b;
        if (trans_b == TRANS) {
            use_trans = NO_TRANS;
        } else {
            use_trans = TRANS;
        }
        ga = ops::matmul(grad, b, NO_TRANS, use_trans);
        ga = clone(ops::broadcast_to(ga, a.get_shape()));
    }

    if (a_is_vector) {
        gb = inner_backward(a, grad, trans_b, b_is_vector, true);
    } else if (b_is_vector) {
        long roll = trans_a == TRANS ? -2 : -1;
        Tensor at = ops::rollaxis(a, roll);
        gb = ops::matmul(at, grad);
    } else {
        auto use_trans = trans_a;
        if (trans_a == TRANS) {
            use_trans = NO_TRANS;
        } else {
            use_trans = TRANS;
        }
        gb = ops::matmul(a, grad, use_trans, NO_TRANS);
        gb = clone(ops::broadcast_to(gb, b.get_shape()));
    }

    ENABLE_GRAD_IND(a)
    ENABLE_GRAD_IND(b)

    return {ga, gb};
}

}  // namespace autograd
}  // namespace sail