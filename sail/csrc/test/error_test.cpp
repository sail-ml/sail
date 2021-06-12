#include "gtest/gtest.h"
#include "core/error.h"
#include "core/tensor_shape.h"

#include <iostream>

TEST(ErrorTest, CatchDimensionError) {
    
    sail::TensorShape x = sail::TensorShape({10});
    ASSERT_THROW(x.insert_one(-10), DimensionError);

}
