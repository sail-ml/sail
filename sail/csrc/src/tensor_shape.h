#pragma once
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace sail {
using LongVec = std::vector<long>;
class TensorShape {
   public:
    LongVec shape;
    LongVec strides;
    LongVec shape_m1;
    LongVec coordinates;
    LongVec back_strides;
    long d_ptr = 0;

    explicit TensorShape(){};

    TensorShape(LongVec shape_, LongVec size_);
    TensorShape(LongVec shape_);

    void insert_one(const int dim);
    void remove_one(const int dim);
    void recompute();

    int next();
    void reset();

    long* get_shape_ptr();

    long numel();
    long getTotalSize(int mod);
    int ndim();

    std::string get_string();
};
}  // namespace sail