#pragma once
#include "sgd.h"
#include <omp.h>
#include "Tensor.h"
#include "dtypes.h"
#include "exception.h"
#include "kernels/Kernel.h"
#include "modules/module.h"
#include "ops/ops.h"
#include "optimizer.h"
namespace sail {
namespace optimizers {

using TensorVector = std::vector<Tensor>;

void SGD::update() {
    Optimizer::steps += 1;
    int n = Optimizer::params.size();

    for (int i = 0; i < n; i++) {
        Tensor t = Optimizer::params[i];
        Tensor grad = t.get_grad();
        SAIL_CHECK(t.is_view() == false)
        sail::internal::sgd_stub(t, grad, learning_rate);
        t.clear_grad();
        t.clear_function();
    }
}

}  // namespace optimizers
}  // namespace sail
