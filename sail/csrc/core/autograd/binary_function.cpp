#pragma once

#include "binary_function.h"
#include <iostream>
#include <vector>
#include "Tensor.h"
#include "factories.h"
#include "function.h"
#include "ops/ops.h"

namespace sail {

namespace autograd {

using TensorVector = std::vector<Tensor>;

/** begin block
 * cName = [Add, Subtract, Divide, Multiply]
 * op = [+, -, /, *]
 */

std::string Add::getName() { return "AddOp"; }
Tensor Add::forward(TensorVector inputs) {
    return ops::add(inputs[0], inputs[1]);
}
TensorVector Add::backward(Tensor& grad) {
    TensorVector o;
    o.emplace_back(grad);
    o.emplace_back(grad);
    return o;
}

std::string Subtract::getName() { return "SubtractOp"; }
Tensor Subtract::forward(TensorVector inputs) {
    return ops::subtract(inputs[0], inputs[1]);
}
TensorVector Subtract::backward(Tensor& grad) {
    TensorVector o;
    o.emplace_back(grad);
    o.emplace_back(-grad);
    return o;
}

std::string Divide::getName() { return "DivideOp"; }
Tensor Divide::forward(TensorVector inputs) {
    return ops::divide(inputs[0], inputs[1]);
}
TensorVector Divide::backward(Tensor& grad) {
    Tensor a = Function::arg_storage[0];
    Tensor b = Function::arg_storage[1];

    Tensor gx0 = grad / b;

    Tensor gx1 = -gx0 * a / b;  // * a;  //((a / b) / b);

    TensorVector o = {gx0, gx1};
    return o;
}

std::string Multiply::getName() { return "MultiplyOp"; }
Tensor Multiply::forward(TensorVector inputs) {
    return ops::multiply(inputs[0], inputs[1]);
}
TensorVector Multiply::backward(Tensor& grad) {
    Tensor a = Function::arg_storage[0];
    Tensor b = Function::arg_storage[1];
    TensorVector o = {b, a};
    return o;
}

std::string Matmul::getName() { return "MatmulOp"; }
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

std::string Pow::getName() { return "PowOp"; }
Tensor Pow::forward(TensorVector inputs) {
    return ops::power(inputs[0], inputs[1]);
}
TensorVector Pow::backward(Tensor& grad) { throw SailCError("Not yet"); }

}  // namespace autograd
}  // namespace sail