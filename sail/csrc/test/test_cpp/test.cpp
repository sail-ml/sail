#include "../../src/Tensor.h"
#include "../../src/dtypes.h"
#include "../../src/ops/ops.h"

#include <iostream>

#include <gtest/gtest.h>

#define MAX_VAL 320000


// Demonstrate some basic assertions.
TEST(HelloTest, BasicAssertions) {
  // Expect two strings not to be equal.
  EXPECT_STRNE("hello", "world");
  // Expect equality.
  EXPECT_EQ(7 * 6, 42);
}