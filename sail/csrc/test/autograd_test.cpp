#include "core/autograd/autograd.h"
#include "core/exception.h"
#include "core/ops/ops.h"
#include "core/tensor_shape.h"
#include "gtest/gtest.h"

#include <iostream>

TEST(AutogradTest, ClipMinOnlyBackwards) {
    sail::Tensor x = sail::random::uniform(sail::TensorShape({32, 32}), 0, 1);
    x.requires_grad = true;

    auto y = sail::ops::clip_min(x, 0.5);
    auto z = y.sum();

    z.backward();
}

TEST(AutogradTest, EmptyFunction) {
    auto x = sail::autograd::Function();

    std::vector<sail::Tensor> t = {};
    auto emp = sail::random::uniform(sail::TensorShape({1}), 0, 1);

    ASSERT_THROW(x.forward(t), SailCError);
    ASSERT_THROW(x.backward(emp), SailCError);
}
