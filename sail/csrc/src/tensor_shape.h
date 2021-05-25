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
    long at = 0;
    bool contiguous = true;

    explicit TensorShape(){};

    TensorShape(LongVec shape_, LongVec size_);
    TensorShape(LongVec shape_);

    TensorShape reverse();

    // template <class T>
    TensorShape reorder(const LongVec& order);

    TensorShape move_axis(long axis, long position);

    void insert_one(const int dim);
    void remove_one(const int dim);
    void recompute_strides();
    void recompute(bool strides_too = false);
    std::vector<long> generate_all_indexes();

    int next();
    void reset();

    long int* get_shape_ptr();

    long numel() const;
    long getTotalSize(int mod);
    int ndim();

    std::string get_string();
};
}  // namespace sail