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
TEST(ForceBinary, Add) {
    auto x1 = random::uniform(TensorShape({10, 10}), -1, 1);
    auto x2 = random::uniform(TensorShape({10, 10}), -1, 1);

    auto y = x1 + x2;
    sail::internal::add_stub.USE = sail::internal::add_stub.DEFAULT;
    auto z = x1 + x2;

    auto y_d = (float*)y.get_data();
    auto z_d = (float*)z.get_data();

    for (int i = 0; i < y.numel(); i++) {
        auto error = y_d[i] - z_d[i];
        ASSERT_LE(error, EPS);
    }

}
TEST(ForceBinary, Subtract) {
    auto x1 = random::uniform(TensorShape({10, 10}), -1, 1);
    auto x2 = random::uniform(TensorShape({10, 10}), -1, 1);

    auto y = x1 - x2;
    sail::internal::subtract_stub.USE = sail::internal::subtract_stub.DEFAULT;
    auto z = x1 - x2;

    auto y_d = (float*)y.get_data();
    auto z_d = (float*)z.get_data();

    for (int i = 0; i < y.numel(); i++) {
        auto error = y_d[i] - z_d[i];
        ASSERT_LE(error, EPS);
    }

}
TEST(ForceBinary, Multiply) {
    auto x1 = random::uniform(TensorShape({10, 10}), -1, 1);
    auto x2 = random::uniform(TensorShape({10, 10}), -1, 1);

    auto y = x1 * x2;
    sail::internal::multiply_stub.USE = sail::internal::multiply_stub.DEFAULT;
    auto z = x1 * x2;

    auto y_d = (float*)y.get_data();
    auto z_d = (float*)z.get_data();

    for (int i = 0; i < y.numel(); i++) {
        auto error = y_d[i] - z_d[i];
        ASSERT_LE(error, EPS);
    }

}
TEST(ForceBinary, Divide) {
    auto x1 = random::uniform(TensorShape({10, 10}), 1, 2);
    auto x2 = random::uniform(TensorShape({10, 10}), 1,2);

    auto y = x1 / x2;
    sail::internal::divide_stub.USE = sail::internal::divide_stub.DEFAULT;
    auto z = x1 / x2;

    auto y_d = (float*)y.get_data();
    auto z_d = (float*)z.get_data();

    for (int i = 0; i < y.numel(); i++) {
        auto error = y_d[i] - z_d[i];
        ASSERT_LE(error, EPS);
    }

}
TEST(ForceBinary, Conv) {
    auto x1 = random::uniform(TensorShape({1, 1, 6, 6}), 1, 2);
    auto x2 = random::uniform(TensorShape({1, 1, 3, 3}), 1,2);

    x1.requires_grad = true;
    auto y = sail::ops::conv2d(x1, x2, {1, 1}, "same");


    auto l = sail::modules::Conv2D(1, 1, 3, 1, "same", false);
    l.set_weights(x2);
    auto z = l.forward(x1);

    auto y_d = (float*)y.get_data();
    auto z_d = (float*)z.get_data();

    for (int i = 0; i < y.numel(); i++) {
        auto error = y_d[i] - z_d[i];
        ASSERT_LE(error, EPS);
    }
}
TEST(ForceBinary, Conv2) {
    auto x1 = random::uniform(TensorShape({1, 1, 6, 6}), 1, 2);
    auto x2 = random::uniform(TensorShape({1, 1, 3, 3}), 1,2);

    x2.requires_grad = true;
    auto y = sail::ops::conv2d(x1, x2, {1, 1}, "same");


    auto l = sail::modules::Conv2D(1, 1, 3, 1, "same", false);
    l.set_weights(x2);
    auto z = l.forward(x1);

    auto y_d = (float*)y.get_data();
    auto z_d = (float*)z.get_data();

    for (int i = 0; i < y.numel(); i++) {
        auto error = y_d[i] - z_d[i];
        ASSERT_LE(error, EPS);
    }
}
