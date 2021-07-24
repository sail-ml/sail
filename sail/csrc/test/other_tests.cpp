#include "gtest/gtest.h"
#include "core/exception.h"
#include "core/tensor_shape.h"
#include "core/Tensor.h"
#include "core/utils.h"
#include "core/ops/ops.h"

#include <iostream>

using namespace sail;
TEST(Misc, GetVectorString) {
    std::vector<long> vec = {(long)1, long(2)};
    ASSERT_EQ(getVectorString(vec), "(1, 2)");
    std::vector<long> vec2 = {};
    ASSERT_EQ(getVectorString(vec2), "()");

}

TEST(Misc, Copy) {
    auto x = sail::random::uniform(TensorShape({10, 2}), 0, 1);
    auto y = ops::copy(x);

    auto x_d = (float*)x.get_data();
    auto y_d = (float*)y.get_data();
    for (int i = 0; i < 20; i++) {
        ASSERT_EQ(x_d[i], y_d[i]);
    }
}
