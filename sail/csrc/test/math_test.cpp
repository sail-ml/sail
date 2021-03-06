#include "core/exception.h"
#include "core/ops/ops.h"
#include "core/tensor_shape.h"
#include "gtest/gtest.h"

#include <iostream>

TEST(MathTest, NegateTest) {
    sail::Tensor x = sail::random::uniform(sail::TensorShape({32, 32}), 0, 1);
    sail::Tensor x2 = -x;

    for (int i = 0; i < 32 * 32; i++) {
        ASSERT_EQ(-((float*)(x.get_data()))[i], ((float*)(x2.get_data()))[i]);
    }
}
