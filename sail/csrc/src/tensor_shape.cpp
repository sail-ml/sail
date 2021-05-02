#include "tensor_shape.h"

#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

namespace sail {
TensorShape::TensorShape(LongVec shape_, LongVec strides_) {
    shape = shape_;
    strides = strides_;

    for (int i; i < shape_.size(); i++) {
        shape_m1.push_back(shape_[i] - 1);
        back_strides.push_back(strides[i] * shape_m1[i]);
    }
}
TensorShape::TensorShape(LongVec shape_) {
    shape = shape_;

    for (int i; i < shape_.size(); i++) {
        shape_m1.push_back(shape_[i] - 1);
    }
}

void TensorShape::insert_one(const int dim) {
    shape.insert(shape.begin() + dim, 1);
    strides.insert(strides.begin() + dim, 1);
    shape_m1.insert(shape_m1.begin() + dim, 0);
    back_strides.insert(back_strides.begin() + dim, 0);
}
void TensorShape::remove_one(const int dim) {
    shape.erase(shape.begin() + dim);
    strides.erase(strides.begin() + dim);
    shape_m1.erase(shape_m1.begin() + dim);
    back_strides.erase(back_strides.begin() + dim);
}

long TensorShape::numel() {
    long s = 1;
    for (long a : shape) {
        s *= a;
    }
    return s;
}
long TensorShape::getTotalSize(int mod) {
    long s = 1;
    for (long a : shape) {
        s *= (a * mod);
    }
    return s;
}

std::string TensorShape::get_string() {
    std::stringstream result;
    std::copy(shape.begin(), shape.end(),
              std::ostream_iterator<int>(result, ", "));
    std::string x = result.str();
    x.pop_back();
    x.pop_back();
    // std::string  shape_string("(");
    return std::string("(") + x + std::string(")");
}

long* TensorShape::get_shape_ptr() { return (long*)shape.data(); }
int TensorShape::ndim() { return shape.size(); }
}  // namespace sail