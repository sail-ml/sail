#include "../../src/Tensor.h"
#include "../../src/dtypes.h"
#include "../../src/ops/ops.h"

#include <iostream>
#include <ostream>

#include <gtest/gtest.h>

#define GTEST_COUT std::cerr << "[          ] [ INFO ] "

#define MAX_VAL 320000

// Demonstrate some basic assertions.
TEST(SailTest, FreeTest) {
  // Expect two strings not to be equal.
    double x[MAX_VAL];
    double y[MAX_VAL];
    int ndim = 1;
    Dtype dt = Dtype::sFloat64;
    TensorSize st = {8};
    TensorSize sh = {MAX_VAL};

    for (int i = 0; i < MAX_VAL; i++) {
        x[i] = 2.21;
        y[i] = 3.21;
    }

    void* xt = static_cast<void*>(x);
    void* yt = static_cast<void*>(y);

    sail::Tensor t1 = sail::Tensor(ndim, xt, dt, st, sh);
    sail::Tensor t2 = sail::Tensor(ndim, yt, dt, st, sh);

    t1.free();
    t2.free();

    ASSERT_EQ(t1.data, NULL);
    ASSERT_EQ(t2.data, NULL);
}

TEST(SailTest, CastFloat32ToInt32) {
    float x[MAX_VAL];
    int ndim = 1;
    Dtype dt = Dtype::sFloat32;
    TensorSize st = {8};
    TensorSize sh = {MAX_VAL};

    float val = 2.21;

    for (int i = 0; i < MAX_VAL; i++) {
        x[i] = 2.21;
    }

    void* xt = static_cast<void*>(x);

    sail::Tensor t1 = sail::Tensor(ndim, xt, dt, st, sh);
    sail::Tensor t2 = t1.cast(Dtype::sInt32);

    int32_t* t2_data = (int32_t*)(t2.data);
    float* t1_data = (float*)(t1.data);

    for (int i = 0; i < MAX_VAL; i++) {
        ASSERT_EQ(t2_data[i], 2);
        ASSERT_EQ(t1_data[i], val);
    }

    t1.free();
    t2.free();

    ASSERT_EQ(t1.data, NULL);
    ASSERT_EQ(t2.data, NULL);
}


TEST(SailTest, CastInt32ToFloat64) {
    int32_t x[MAX_VAL];
    int ndim = 1;
    Dtype dt = Dtype::sInt32;
    TensorSize st = {4};
    TensorSize sh = {MAX_VAL};

    int32_t val = 2;
    double check_val = 2;

    for (int i = 0; i < MAX_VAL; i++) {
        x[i] = 2;
    }

    void* xt = static_cast<void*>(x);

    sail::Tensor t1 = sail::Tensor(ndim, xt, dt, st, sh);
    sail::Tensor t2 = t1.cast(Dtype::sFloat64);

    double* t2_data = (double*)(t2.data);
    int32_t* t1_data = (int32_t*)(t1.data);

    for (int i = 0; i < MAX_VAL; i++) {
        ASSERT_EQ(t2_data[i], check_val);
        ASSERT_EQ(t1_data[i], val);
    }

    t1.free();
    t2.free();

    ASSERT_EQ(t1.data, NULL);
    ASSERT_EQ(t2.data, NULL);
}