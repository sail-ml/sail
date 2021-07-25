#include "core/Tensor.h"
#include "core/dtypes.h"
#include "core/factories.h"
#include "core/kernels/Kernel.h"
#include "core/modules/modules.h"
#include "core/ops/ops.h"
#include "core/ops/tools.h"
#include "core/tensor_shape.h"
#include "gtest/gtest.h"

#include <iostream>

#define EPS 0.00001

using namespace sail;
TEST(OpsTools, MustBroadcast) {
    auto x1 = random::uniform(TensorShape({10, 10}), -1, 1);
    auto x2 = random::uniform(TensorShape({10, 1}), -1, 1);

    ASSERT_TRUE(must_broadcast(x1, x2));

    x1 = random::uniform(TensorShape({10, 10}), -1, 1);
    x2 = random::uniform(TensorShape({1, 1}), -1, 1);

    ASSERT_TRUE(must_broadcast(x1, x2));

    x1 = random::uniform(TensorShape({10, 10}), -1, 1);
    x2 = random::uniform(TensorShape({5}), -1, 1);

    ASSERT_THROW(must_broadcast(x1, x2), SailCError);

    x1 = random::uniform(TensorShape({10, 10}), -1, 1);
    x2 = random::uniform(TensorShape({10, 10}), -1, 1);

    ASSERT_FALSE(must_broadcast(x1, x2));
}
TEST(OpsTools, MergeShapes) {
    auto x1 = random::uniform(TensorShape({10, 10}), -1, 1);
    auto x2 = random::uniform(TensorShape({10, 1}), -1, 1);
    std::vector<long> check = {(long)10, (long)10};

    ASSERT_EQ(merge_shapes(x1.get_shape().shape, x2.get_shape().shape), check);

    x1 = random::uniform(TensorShape({10, 10}), -1, 1);
    x2 = random::uniform(TensorShape({1, 1}), -1, 1);

    ASSERT_EQ(merge_shapes(x1.get_shape().shape, x2.get_shape().shape), check);

    x1 = random::uniform(TensorShape({1, 10, 10}), -1, 1);
    x2 = random::uniform(TensorShape({10, 1, 1}), -1, 1);
    check.push_back(long(10));

    ASSERT_EQ(merge_shapes(x1.get_shape().shape, x2.get_shape().shape), check);

    x1 = random::uniform(TensorShape({10, 10, 10}), -1, 1);
    x2 = random::uniform(TensorShape({10, 10, 10}), -1, 1);

    ASSERT_EQ(merge_shapes(x1.get_shape().shape, x2.get_shape().shape), check);
}
