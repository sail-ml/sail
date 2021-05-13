#include "tensor_shape.h"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

namespace sail {
TensorShape::TensorShape(LongVec& shape_, LongVec& strides_) {
    shape = shape_;
    strides = strides_;
    // if (shape.size() != 0) {
    //     strides.erase(strides.begin());
    // }
    // strides.push_back(1);
    std::reverse(strides.begin(), strides.end());

    std::vector<long> co(shape.size(), 0);
    coordinates = co;

    for (int i; i < shape_.size(); i++) {
        // if (i > 0) {
        //     strides[i] = strides[i] * strides[i - 1];
        // }
        shape_m1.push_back(shape_[i] - 1);
        back_strides.push_back(strides[i] * shape_m1[i]);
    }
    std::reverse(strides.begin(), strides.end());
}
TensorShape::TensorShape(LongVec& shape_) {
    shape = shape_;
    strides = shape_;
    if (shape.size() != 0) {
        strides.erase(strides.begin());
    }
    strides.push_back(1);
    std::reverse(strides.begin(), strides.end());

    if (shape.size() > 0) {
        std::vector<long> co(shape.size(), 0);
        coordinates = co;

        for (int i; i < shape_.size(); i++) {
            if (i > 0) {
                strides[i] = strides[i] * strides[i - 1];
            }
            shape_m1.push_back(shape_[i] - 1);
            back_strides.push_back(strides[i] * shape_m1[i]);
        }
        std::reverse(strides.begin(), strides.end());
    }
}
int TensorShape::next() {
    int i;
    // if (contiguous) {
    //     d_ptr += 1;
    if (shape.size() == 0 || shape.size() == 1 && shape[0] == 1) {
        return d_ptr;
    }
    if (shape.size() == 1) {
        d_ptr += strides[0];
        coordinates[0]++;
        // } else if (shape.size() == 2) {
        //     if (coordinates[1] < shape_m1[1]) {
        //         coordinates[1]++;
        //         d_ptr += strides[1];
        //     } else {
        //         coordinates[1] = 0;
        //         coordinates[0]++;
        //         d_ptr += strides[0] - back_strides[1];
        //     }
    } else {
        for (i = shape.size() - 1; i >= 0; i--) {
            if (coordinates[i] < shape_m1[i]) {
                coordinates[i] += 1;
                d_ptr += strides[i];
                at = i;
                break;
            } else {
                coordinates[i] = 0;
                d_ptr -= back_strides[i];
                at = i;
            }
        }
    }
    return d_ptr;
}

void TensorShape::recompute() {
    LongVec new_s_m1, n_b_s;
    for (int i; i < shape.size(); i++) {
        new_s_m1.push_back(shape[i] - 1);
        n_b_s.push_back(strides[i] * new_s_m1[i]);
    }
    shape_m1 = new_s_m1;
    back_strides = n_b_s;
    std::vector<long> co(shape_m1.size(), 0);
    coordinates = co;
}

void TensorShape::reset() {
    std::vector<long> coordinates(shape.size(), 0);
    d_ptr = 0;
    at = -1;
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

long TensorShape::numel() const {
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

std::vector<long> TensorShape::generate_all_indexes() {
    std::vector<long> out;
    for (int i = 0; i < numel(); i++) {
        out.push_back(d_ptr);
        next();
    }
    reset();
    return out;
}

long* TensorShape::get_shape_ptr() { return (long*)shape.data(); }
int TensorShape::ndim() { return shape.size(); }
}  // namespace sail