#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <tuple>
#include <unistd.h>
#include <utility>
#include "gtest/gtest.h"
#include "core/Tensor.h"
#include "core/modules/modules.h"
#include "core/factories.h"
#include "core/ops/ops.h"
#include "core/tensor_shape.h"
#include "core/slice.h"
#include "core/kernels/Kernel.h"
#include "core/onednn/pooling.h"

using namespace sail;
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();

     return 0;
}