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
    int enforced = -1;
    bool is_single = false;

    explicit TensorShape() = default;

    TensorShape(LongVec shape_, LongVec size_);
    TensorShape(LongVec shape_);

    TensorShape reverse();

    TensorShape reorder(const LongVec& order);

    TensorShape roll_axis(long axis, long position);
    TensorShape move_axis(long axis, long position);

    void ignore_innermost();

    void insert_one(const int dim);
    void remove_one(const int dim);
    void recompute_strides();
    void recompute(bool strides_too = false);
    void enforce_axis(int axis);

    bool operator==(const TensorShape& other) const;
    long operator[](const int index) const;

    int next();
    int next(int n);
    void reset();

    long int* get_shape_ptr();

    long numel() const;
    long ndim() const;

    std::string get_string() const;
};
}  // namespace sail