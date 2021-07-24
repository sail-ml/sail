#include "gtest/gtest.h"
#include "core/factories.h"
#include "core/ops/ops.h"
#include "core/ops/tools.h"
#include "core/Tensor.h"
#include "core/dtypes.h"
#include "core/modules/modules.h"
#include "core/kernels/Kernel.h"
#include "core/tensor_shape.h"
#include "core/initializers/utils.h"

#include <iostream>

#define EPS 0.00001

using namespace sail;
TEST(Init, gain) {
    ASSERT_EQ(initializers::get_gain("linear"), 1.0);
    ASSERT_EQ(initializers::get_gain("conv2d"), 1.0);
    ASSERT_EQ(initializers::get_gain("sigmoid"), 1.0);
    ASSERT_EQ(initializers::get_gain("tanh"), 5.0/3);
}
