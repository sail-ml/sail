#include "gtest/gtest.h"
#include "core/exception.h"
#include "core/tensor_shape.h"
#include "core/Tensor.h"
#include "core/numeric.h"
#include "core/ops/ops.h"

#include <iostream>

TEST(TensorTest, ForceIncref) {
    
    sail::Tensor x = sail::random::uniform(sail::TensorShape({4, 5, 3, 2}), 0, 1);
    
    x.get_body()->force_incref();
    ASSERT_EQ(x.get_body()->get_ref_count(), 3);
    x.get_body()->force_decref();

}
