#include "tensor_iterator.h"
#include <omp.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include "Tensor.h"
#include "tensor_shape.h"

namespace sail {

TensorIterator::TensorIterator(TensorShape t_shape) {
    std::vector<long> old_shape = t_shape.shape;
    std::vector<long> old_strides = t_shape.strides;

    long old_ndim = t_shape.ndim();

    long last = 0;
    int i2 = 0;

    for (int i = 0; i < old_ndim; i++) {
        if (i == 0) {
            shape.emplace_back(old_shape[0]);
            strides.emplace_back(old_strides[0]);
            shape_m1.emplace_back(old_shape[0] - 1);
            strides_back.emplace_back(shape_m1[0] * strides[0]);
        } else {
            if ((old_strides[i] == 0 && last == 0) ||
                (old_strides[i] != 0 && last != 0)) {
                shape[i2] *= old_shape[i];
                strides[i2] = old_strides[i];
            } else {
                shape_m1[i2] = shape[i2] - 1;
                strides_back[i2] = (shape_m1[i2] * strides[i2]);

                _inner_loop_size *= shape[i2];

                shape.emplace_back(old_shape[i]);
                strides.emplace_back(old_strides[i]);

                shape_m1.emplace_back(0);
                strides_back.emplace_back(0);
                i2 += 1;
            }
        }
        last = old_strides[i];
    }

    _ndim = shape.size();
    _numel =
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());

    std::vector<long> co(i2, 0);
    coordinates = co;
    lasts.push_back(strides.back());
}

long TensorIterator::numel() const { return _numel; }

long TensorIterator::ndim() const { return _ndim; }

long TensorIterator::inner_loop_size() const { return shape.back(); }

long TensorIterator::out_loop_size() const { return _numel / shape.back(); }

void TensorIterator::advance_d_ptr(int times) { d_ptr += lasts[0] * times; }
void TensorIterator::backup_d_ptr() { d_ptr -= lasts[0]; }

long TensorIterator::next() {
    int i = 0;
    for (i = _ndim - 2; i >= 0; i--) {
        if (coordinates[i] < shape_m1[i]) {
            coordinates[i] += 1;
            d_ptr += strides[i];
            break;
        } else {
            coordinates[i] = 0;
            d_ptr -= strides_back[i];
        }
    }
    d_ptr -= strides_back.back();
    return d_ptr;
}

MultiTensorIterator::MultiTensorIterator(TensorShape t_shape) {
    std::vector<long> old_shape = t_shape.shape;
    std::vector<long> old_strides = t_shape.strides;
    TensorIterator::shape = old_shape;
    TensorIterator::_ndim = old_shape.size();
    std::vector<long> temp_strides_back(_ndim);

    shape = old_shape;
    strides = Vec2D(old_strides);
    lasts.emplace_back(old_strides.back());

    std::vector<long> co(TensorIterator::_ndim, 0);
    coordinates = Vec2D(co);

    for (int i = 0; i < _ndim; i++) {
        TensorIterator::_numel *= old_shape[i];
        shape_m1.emplace_back(old_shape[i] - 1);
        temp_strides_back[i] = shape_m1[i] * old_strides[i];
    }

    strides_back = Vec2D(temp_strides_back);
    d_ptrs.push_back(0);
}

MultiTensorIterator MultiTensorIterator::add_input(TensorShape& t_shape) {
    tensor_count += 1;

    std::vector<long> old_strides = t_shape.strides;
    std::vector<long> temp_strides_back(_ndim);

    strides.push_back(old_strides);
    lasts.emplace_back(old_strides.back());

    std::vector<long> co(TensorIterator::_ndim, 0);
    coordinates.push_back(co);

    for (int i = 0; i < TensorIterator::_ndim; i++) {
        temp_strides_back[i] = shape_m1[i] * old_strides[i];
    }

    strides_back.push_back(temp_strides_back);
    d_ptrs.push_back(0);
    return *this;
}

}  // namespace sail