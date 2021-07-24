#include "gtest/gtest.h"
#include "core/factories.h"
#include "core/ops/ops.h"
#include "core/Tensor.h"
#include "core/dtypes.h"
#include "core/kernels/Kernel.h"
#include "core/tensor_shape.h"

#include <iostream>

#define EPS 0.000001

using namespace sail;
TEST(ForceUnary, Negate) {
    auto x = random::uniform(TensorShape({10, 10}), -1, 1);

    auto y = -x;
    sail::internal::negate_stub.USE = sail::internal::negate_stub.DEFAULT;
    auto z = -x;

    auto y_d = (float*)y.get_data();
    auto z_d = (float*)z.get_data();

    for (int i = 0; i < x.numel(); i++) {
        auto error = y_d[i] - z_d[i];
        ASSERT_LE(error, EPS);
    }

}
TEST(ForceUnary, log) {
    auto x = random::uniform(TensorShape({10, 10}), 1, 2);

    auto y = ops::log(x);
    sail::internal::log_stub.USE = sail::internal::log_stub.DEFAULT;
    auto z = ops::log(x);

    auto y_d = (float*)y.get_data();
    auto z_d = (float*)z.get_data();

    for (int i = 0; i < x.numel(); i++) {
        auto error = y_d[i] - z_d[i];
        ASSERT_LE(error, EPS);
    }

}
