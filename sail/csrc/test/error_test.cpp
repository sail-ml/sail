// #include "gtest/gtest.h"
// #include "core/exception.h"
// #include "core/tensor_shape.h"
// #include "core/ops/ops.h"

// #include <iostream>

// TEST(ErrorTest, CatchDimensionError) {
    
//     sail::TensorShape x = sail::TensorShape({10});
//     ASSERT_THROW(x.insert_one(-10), DimensionError);

// }

// TEST(ErrorTest, MatmulErrorCatch1) {
    
//     sail::Tensor x = sail::random::uniform(sail::TensorShape({10, 20}), 0, 1);
//     sail::Tensor x2 = sail::random::uniform(sail::TensorShape({20, 10}), 0, 1);
//     ASSERT_THROW(sail::ops::matmul(x, x2, "T", "N"), SailCError);

// }

// TEST(ErrorTest, MatmulErrorCatch2) {
    
//     sail::Tensor x = sail::random::uniform(sail::TensorShape({10, 20}), 0, 1);
//     sail::Tensor x2 = sail::random::uniform(sail::TensorShape({20, 10}), 0, 1);
//     ASSERT_THROW(sail::ops::matmul(x, x2, "N", "T"), SailCError);

// }

// TEST(ErrorTest, MatmulErrorCatch3) {
    
//     sail::Tensor x = sail::random::uniform(sail::TensorShape({10, 25}), 0, 1);
//     sail::Tensor x2 = sail::random::uniform(sail::TensorShape({20, 13}), 0, 1);
//     ASSERT_THROW(sail::ops::matmul(x, x2, "T", "T"), SailCError);

// }
