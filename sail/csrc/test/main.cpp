#include "gtest/gtest.h"
#include "core/Tensor.h"
#include "core/modules/modules.h"
#include "core/factories.h"
#include "core/ops/ops.h"
#include "core/tensor_shape.h"
#include "core/slice.h"

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
     return 0;
}