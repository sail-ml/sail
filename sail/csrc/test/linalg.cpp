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

TEST(LinalgTest, TensordotWrongSize) {
    
    sail::Tensor x = sail::random::uniform(sail::TensorShape({4, 5, 3, 2}), 0, 1);
    sail::Tensor x2 = sail::random::uniform(sail::TensorShape({3, 2, 6}), 0, 1);
    ASSERT_THROW(sail::ops::tensordot(x, x2, {2, 3}, {1}), SailCError);

}

TEST(LinalgTest, MatmulScalar) {
    
    sail::Tensor x = sail::random::uniform(sail::TensorShape({1}), 0, 1);
    sail::Tensor x2 = sail::random::uniform(sail::TensorShape({3, 2}), 0, 1);
    ASSERT_THROW(sail::ops::matmul(x, x2), SailCError);

}

TEST(LinalgTest, VectorMatrix) {
    
    sail::Tensor x = sail::random::uniform(sail::TensorShape({3}), 0, 1);
    sail::Tensor x2 = sail::random::uniform(sail::TensorShape({3, 2}), 0, 1);
    sail::ops::matmul(x, x2);


}
