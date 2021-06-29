#include <iostream>

#include "ExponentialOps.h"
#include "Tensor.h"
#include "autograd/autograd.h"
#include "factories.h"
#include "kernels/Kernel.h"
#include "ops/tools.h"

#define MAX(a, b) (((a.ndim) > (b.ndim)) ? (a) : (b))
#define MIN(a, b) (((a.ndim) < (b.ndim)) ? (a) : (b))

namespace sail {

namespace ops {
using TensorVector = std::vector<Tensor>;

Tensor exp(const Tensor& tensor1) {
    if (tensor1.requires_grad) {
        TensorVector vec;
        vec.emplace_back(tensor1);
        return (new autograd::Exp())->apply(vec);
    }
    Tensor empty_tensor = empty_like(tensor1);
    sail::internal::exp_stub(tensor1, empty_tensor);
    return empty_tensor;
}

Tensor log(const Tensor& tensor1) {
    if (tensor1.requires_grad) {
        TensorVector vec;
        vec.emplace_back(tensor1);
        return (new autograd::Log())->apply(vec);
    }
    Tensor empty_tensor = empty_like(tensor1);
    sail::internal::log_stub(tensor1, empty_tensor);
    return empty_tensor;
}

Tensor power(const Tensor& tensor1, const Tensor& tensor2) {
    if (tensor1.requires_grad) {
        TensorVector vec;
        vec.emplace_back(tensor1);
        vec.emplace_back(tensor2);
        Tensor empty_tensor = (new autograd::Pow())->apply(vec);
        return empty_tensor;
    }
    Tensor empty_tensor;  //= empty_like(tensor1);
    Tensor t1, t2;

    Dtype dt = promote_dtype(tensor1.get_dtype(), tensor2.get_dtype());
    bool broadcast = must_broadcast(tensor1, tensor2);
    if (broadcast) {
        std::vector<long> new_ =
            merge_shapes(tensor1.get_shape().shape, tensor2.get_shape().shape);
        TensorShape s = TensorShape(new_);
        empty_tensor = empty(s.ndim(), dt, s);
        empty_tensor.requires_grad = t1.requires_grad;
        t1 = ops::broadcast_to(tensor1, s);
        t2 = ops::broadcast_to(tensor2, s);
    } else {
        t1 = tensor1;
        t2 = tensor2;
        if (!t1.is_view()) {
            empty_tensor = empty_like(t1, dt);
        } else if (!t2.is_view()) {
            empty_tensor = empty_like(t2, dt);
        } else {
            empty_tensor = empty_like(t1, dt);
            TensorShape ns = TensorShape(t1.get_shape().shape);
            empty_tensor.set_shape(ns);
        }
    }

    if (t1.is_view() || t2.is_view()) {
        broadcast = true;
    }

    t1 = t1.cast(dt);
    t2 = t2.cast(dt);
    sail::internal::power_stub(t1, t2, empty_tensor, broadcast);
    return empty_tensor;
}

}  // namespace ops

}  // namespace sail
