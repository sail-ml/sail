#include "core/Tensor.h"
#include "core/exception.h"
#include "core/numeric.h"
#include "core/ops/ops.h"
#include "core/tensor_iterator.h"
#include "core/tensor_shape.h"
#include "gtest/gtest.h"

#include <iostream>

using namespace sail;
TEST(IterTest, Create) {
    sail::Tensor x =
        sail::random::uniform(sail::TensorShape({4, 5, 3, 2}), 0, 1);
    auto iter = TensorIterator(x.get_shape());
}

TEST(IterTest, Create2) {
    sail::Tensor x =
        sail::random::uniform(sail::TensorShape({1, 5, 1, 1}), 0, 1);
    auto y = ops::broadcast_to(x, TensorShape({10, 5, 4, 3}));
    auto iter = TensorIterator(y.get_shape());
    iter.ndim();

    ASSERT_EQ(iter.numel(), 10 * 5 * 4 * 3);
}

TEST(IterTest, next) {
    sail::Tensor x =
        sail::random::uniform(sail::TensorShape({4, 5, 3, 2}), 0, 1);
    auto iter = TensorIterator(x.get_shape());
    iter.advance_d_ptr(5);
    iter.backup_d_ptr();
    iter.next();
}
TEST(IterTest, next2) {
    sail::Tensor x =
        sail::random::uniform(sail::TensorShape({4, 5, 3, 2}), 0, 1);
    auto iter = TensorIterator(x.get_shape());

    int inner_loop_size = iter.inner_loop_size();
    int outer_steps = iter.out_loop_size();

    int z = 0;
    for (int i = 0; i < outer_steps; i++) {
        for (int j = 0; j < inner_loop_size; j += 1) {
            iter.advance_d_ptr(1);
        }
        iter.backup_d_ptr();
        iter.next();
    }
}
