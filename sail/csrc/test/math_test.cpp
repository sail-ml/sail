#include "gtest/gtest.h"
#include "core/error.h"
#include "core/tensor_shape.h"
#include "core/ops/ops.h"

#include <iostream>


TEST(MathTest, NegateTest) {
    
    // sail::Tensor x = sail::random::uniform(sail::TensorShape({32, 32}), 0, 1);
    // sail::Tensor x2 = -x;

    // for (int i = 0; i < 32 * 32; i++) {
    //     ASSERT_EQ(-((float*)(x.get_data()))[i], ((float*)(x2.get_data()))[i]);
    // }
    sail::Tensor x = sail::random::uniform(sail::TensorShape({32, 32}), 0, 1);
    sail::Tensor x2 = -x;
    std::cout << x2 << std::endl;
}
