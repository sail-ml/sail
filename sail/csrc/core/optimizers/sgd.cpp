#pragma once
#include "sgd.h"
#include <omp.h>
#include "Tensor.h"
#include "dtypes.h"
#include "exception.h"
#include "kernels/Kernel.h"
#include "modules/module.h"
#include "ops/ops.h"
#include "optimizers.h"
namespace sail {
namespace optimizers {

using TensorVector = std::vector<Tensor>;

#define DISABLE_GRAD(a, b)                     \
    {                                          \
        a.was_requires_grad = a.requires_grad; \
        a.requires_grad = false;               \
        b.was_requires_grad = b.requires_grad; \
        b.requires_grad = false;               \
    }
#define ENABLE_GRAD(a, b)                      \
    {                                          \
        a.requires_grad = a.was_requires_grad; \
        b.requires_grad = b.was_requires_grad; \
    }

void SGD::update() {
    Optimizer::steps += 1;
    int n = Optimizer::params.size();

    for (int i = 0; i < n; i++) {
        Tensor t = Optimizer::params[i];
        Tensor grad = t.get_grad();
        // std::cout << grad << std::endl;
        SAIL_CHECK(t.is_view() == false)
        // DISABLE_GRAD(t, grad);
        sail::internal::sgd_stub(t, grad, learning_rate);
        // ENABLE_GRAD(t, grad);
        // t += grad;
        t.clear_grad();
        t.clear_function();
    }
}

}  // namespace optimizers
}  // namespace sail
