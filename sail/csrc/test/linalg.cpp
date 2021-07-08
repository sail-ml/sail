#include "gtest/gtest.h"
#include "core/exception.h"
#include "core/tensor_shape.h"
#include "core/ops/ops.h"

#include <iostream>




TEST(LinalgTest, TensorDotMissing) {
    
    sail::Tensor x = sail::random::uniform(sail::TensorShape({4, 5, 3, 2}), 0, 1);
    sail::Tensor x2 = sail::random::uniform(sail::TensorShape({3, 2, 6}), 0, 1);
    sail::Tensor x3 = sail::ops::tensordot(x, x2, 2);

    sail::TensorShape expected = sail::TensorShape({4, 5, 6});
    ASSERT_EQ(x3.get_shape().shape, expected.shape);

}
