#include "gtest/gtest.h"
#include "core/error.h"
#include "core/tensor_shape.h"
#include "core/ops/ops.h"

#include <iostream>




TEST(TensorTest, BodyCount) {
    
    sail::Tensor x = sail::random::uniform(sail::TensorShape({4, 5, 3, 2}), 0, 1);
    sail::Tensor y = x;
    sail::Tensor z = y;

    ASSERT_EQ(x.get_body_ref_count(), 3);
    ASSERT_EQ(y.get_body_ref_count(), 3);
    ASSERT_EQ(z.get_body_ref_count(), 3);
}

TEST(TensorTest, SetShape) {
    
    sail::Tensor x = sail::random::uniform(sail::TensorShape({1, 2, 3}), 0, 1);
    sail::TensorShape sh = sail::TensorShape({3, 2, 1});
    x.set_shape(sh);

    ASSERT_EQ(x.get_shape().shape, sh.shape);
}