#include "gtest/gtest.h"
#include "core/factories.h"
#include "core/ops/ops.h"
#include "core/Tensor.h"
#include "core/dtypes.h"
#include "core/modules/modules.h"
#include "core/kernels/Kernel.h"
#include "core/tensor_shape.h"

#include <iostream>

#define EPS 0.00001

using namespace sail;
TEST(ForceCompare, clip_min) {
    auto x1 = random::uniform(TensorShape({10, 10}), -1, 1);

    auto y = ops::clip_min(x1, 0);
    sail::internal::clip_min_stub.USE = sail::internal::clip_min_stub.DEFAULT;
    auto z = ops::clip_min(x1, 0);

    auto y_d = (float*)y.get_data();
    auto z_d = (float*)z.get_data();

    for (int i = 0; i < y.numel(); i++) {
        auto error = y_d[i] - z_d[i];
        ASSERT_LE(error, EPS);
    }

}
TEST(ForceCompare, clip_max) {
    auto x1 = random::uniform(TensorShape({10, 10}), -1, 1);

    auto y = ops::clip_max(x1, 0);
    sail::internal::clip_max_stub.USE = sail::internal::clip_max_stub.DEFAULT;
    auto z = ops::clip_max(x1, 0);

    auto y_d = (float*)y.get_data();
    auto z_d = (float*)z.get_data();

    for (int i = 0; i < y.numel(); i++) {
        auto error = y_d[i] - z_d[i];
        ASSERT_LE(error, EPS);
    }

}
TEST(ForceCompare, clip) {
    auto x1 = random::uniform(TensorShape({10, 10}), -1, 1);

    auto y = ops::clip(x1, -0.5, 0.5);
    sail::internal::clip_stub.USE = sail::internal::clip_stub.DEFAULT;
    auto z = ops::clip(x1, -.5, 0.5);

    auto y_d = (float*)y.get_data();
    auto z_d = (float*)z.get_data();

    for (int i = 0; i < y.numel(); i++) {
        auto error = y_d[i] - z_d[i];
        ASSERT_LE(error, EPS);
    }

}
