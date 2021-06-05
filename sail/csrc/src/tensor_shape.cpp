#include "tensor_shape.h"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "error.h"

namespace sail {

bool is_one(LongVec& shape, int dim) { return shape[dim] == 1; }

TensorShape::TensorShape(LongVec shape_, LongVec strides_) {
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
TensorShape::TensorShape(LongVec shape_) {
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
    //     return d_ptr;
    // }
    if (enforced == -1) {
        if (shape.size() == 0 || (shape.size() == 1 && shape[0] == 1)) {
            return d_ptr;
        }
        if (shape.size() == 1) {
            d_ptr += strides[0];
            coordinates[0]++;
        } else if (shape.size() == 2) {
            if (coordinates[1] < shape_m1[1]) {
                coordinates[1]++;
                d_ptr += strides[1];
            } else {
                coordinates[1] = 0;
                coordinates[0]++;
                d_ptr += strides[0] - back_strides[1];
            }
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
    for (i = shape.size() - 1; i >= 0; i--) {
        if (i == enforced) {
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

void TensorShape::recompute_strides() {
    strides = shape;
    if (shape.size() != 0) {
        strides.erase(strides.begin());
    }
    strides.push_back(1);
    std::reverse(strides.begin(), strides.end());

    if (shape.size() > 0) {
        for (int i; i < shape.size(); i++) {
            if (i > 0) {
                strides[i] = strides[i] * strides[i - 1];
            }
            shape_m1.push_back(shape[i] - 1);
            back_strides.push_back(strides[i] * shape_m1[i]);
        }
        std::reverse(strides.begin(), strides.end());
    }
}
void TensorShape::recompute(bool strides_too = false) {
    if (strides_too) {
        recompute_strides();
    } else {
        LongVec new_s_m1, n_b_s;
        for (int i; i < shape.size(); i++) {
            new_s_m1.push_back(shape[i] - 1);
            n_b_s.push_back(strides[i] * new_s_m1[i]);
        }
        shape_m1 = new_s_m1;
        back_strides = n_b_s;
    }
    std::vector<long> co(shape_m1.size(), 0);
    coordinates = co;
}

TensorShape TensorShape::reverse() {
    std::reverse(shape.begin(), shape.end());
    recompute(true);
    return *this;
}

// template <class T>
TensorShape TensorShape::reorder(const LongVec& order) {
    LongVec Order = order;

    LongVec new_shape;
    LongVec new_strides;

    for (long i : order) {
        new_shape.push_back(shape[i]);
        new_strides.push_back(strides[i]);
    }

    strides = new_strides;
    shape = new_shape;

    recompute();
    return *this;
}

void TensorShape::reset() {
    std::vector<long> coordinates(shape.size(), 0);
    d_ptr = 0;
    at = -1;
}

void TensorShape::enforce_axis(int axis) {
    if (axis < 0) {
        axis = ndim() + axis;
    }
    enforced = axis;
}

void TensorShape::insert_one(const int dim) {
    if (dim == -1) {
        shape.push_back(1);
    } else {
        if (dim > shape.size()) {
            throw DimensionError(
                "Dimension value is too large for expand_dims");
        }
        shape.insert(shape.begin() + dim, 1);
    }
    recompute(true);
}
void TensorShape::remove_one(const int dim) {
    if (dim == -1) {
        int new_dim = shape.size() - 1;
        if (is_one(shape, new_dim)) {
            shape.erase(shape.begin() + new_dim);
        }
    } else {
        if (dim > shape.size()) {
            throw DimensionError("Dimension value is too large for squeeze");
        }
        if (is_one(shape, dim)) {
            shape.erase(shape.begin() + dim);
        }
    }
    recompute(true);
}
void TensorShape::remove(const int dim) {
    if (dim == -1) {
        int new_dim = shape.size() - 1;
        shape.erase(shape.begin() + new_dim);
    } else {
        if (dim > shape.size()) {
            throw DimensionError("Dimension value is too large for squeeze");
        }
        shape.erase(shape.begin() + dim);
    }
    recompute(true);
}

long TensorShape::numel() const {
    long s = 1;
    for (long a : shape) {
        s *= a;
    }
    return s;
}
long TensorShape::numel_avoid(int dim) const {
    long s = 1;
    int c = 0;
    for (long a : shape) {
        if (c != dim) {
            s *= a;
        }
        c += 1;
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

TensorShape TensorShape::move_axis(long axis, long position) {
    if (axis < 0) {
        axis = ndim() + axis;
    }
    if (position < 0) {
        position = ndim() + position;
    }

    if (position < 0 || position >= ndim()) {
        throw SailCError("Invalid position");
    }
    if (axis < 0 || axis >= ndim()) {
        throw SailCError("Invalid axis");
    }

    std::vector<long> axes(ndim());
    std::iota(axes.begin(), axes.end(), 0);

    if (axis < position) {
        position -= 1;
    }

    axes.erase(axes.begin() + axis);
    axes.insert(axes.begin() + position, axis);

    this->reorder(axes);

    // long val_shape = shape[axis];
    // shape.erase(shape.begin() + axis);
    // long val_stride = strides[axis];
    // strides.erase(strides.begin() + axis);

    // shape.insert(shape.begin() + position, val_shape);
    // strides.insert(strides.begin() + position, val_stride);
    // recompute();
    return *this;
}

long int* TensorShape::get_shape_ptr() { return (long*)shape.data(); }
int TensorShape::ndim() { return shape.size(); }
}  // namespace sail