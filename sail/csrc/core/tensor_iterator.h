#pragma once
#include "tensor_shape.h"

#define OUTER_LOOP_2TC(ndim, coords, shape_m1, strides_back, d_ptrs) \
    {                                                                \
        for (int i = TensorIterator::_ndim - 2; i >= 0; i--) {       \
            if (coordinates.at(0, i) < shape_m1[i]) {                \
                coordinates.at(0, i) += 1;                           \
                d_ptrs[0] += strides.at(0, i);                       \
                break;                                               \
            } else {                                                 \
                coordinates.at(0, i) = 0;                            \
                d_ptrs[0] -= strides_back.at(0, i);                  \
            }                                                        \
        }                                                            \
        d_ptrs[0] -= strides_back.at_back(0);                        \
        for (int i = TensorIterator::_ndim - 2; i >= 0; i--) {       \
            if (coordinates.at(1, i) < shape_m1[i]) {                \
                coordinates.at(1, i) += 1;                           \
                d_ptrs[1] += strides.at(1, i);                       \
                break;                                               \
            } else {                                                 \
                coordinates.at(1, i) = 0;                            \
                d_ptrs[1] -= strides_back.at(1, i);                  \
            }                                                        \
        }                                                            \
        d_ptrs[1] -= strides_back.at_back(1);                        \
    }

namespace sail {

template <class T>
class Vec2D {
   public:
    std::vector<T> vec;
    size_t rows = 0;
    size_t cols = 0;
    size_t last_loc = 0;

    explicit Vec2D() = default;
    Vec2D(std::vector<T> vect) {
        vec = vect;
        cols = vec.size();
        last_loc = cols + (cols - 1);
        rows += 1;
    }

    Vec2D emplace_back(std::vector<T> other) {
        vec.insert(vec.end(), other.begin(), other.end());
        rows += 1;
        return *this;
    }
    Vec2D push_back(std::vector<T> other) {
        vec.insert(vec.end(), other.begin(), other.end());
        rows += 1;
        return *this;
    }

    const T& at(int row, int col) const { return vec[row * cols + col]; }
    T& at(int row, int col) { return vec[row * cols + col]; }

    const T& at_back(int row) const { return vec[row * cols + (cols - 1)]; }
    T& at_back(int row) { return vec[row * cols + (cols - 1)]; }
};

class TensorIterator {
   public:
    std::vector<long> shape;
    std::vector<long> strides;
    std::vector<long> shape_m1;
    std::vector<long> strides_back;
    std::vector<long> coordinates;
    std::vector<long> lasts;

    explicit TensorIterator() = default;
    TensorIterator(TensorShape& t_shape);

    long numel() const;
    long ndim() const;
    long inner_loop_size() const;
    long out_loop_size() const;
    virtual void advance_d_ptr(int j = 1);
    virtual void backup_d_ptr();
    long next();

    long _numel = 1;
    long _ndim = 0;
    long _out_loop_size = 1;
    long _inner_loop_size = 1;
    long d_ptr = 0;
};

class MultiTensorIterator : public TensorIterator {
   public:
    std::vector<long> shape;
    std::vector<long> shape_m1;
    Vec2D<long> strides;
    Vec2D<long> strides_back;
    Vec2D<long> coordinates;
    // std::vector<std::vector<long>> strides_back;
    // std::vector<std::vector<long>> coordinates;

    MultiTensorIterator(TensorShape t_shape);

    MultiTensorIterator add_input(TensorShape& t_shape);

    inline bool contiguous_at(int index) {
        int v = 0;  // strides.at(index, 0);
        for (int i = 1; i < shape.size(); i++) {
            if (strides.at(index, i) > v) {
                return false;
            } else {
                v = strides.at(index, i);
            }
        }
        return true;
    }

    inline void advance_d_ptr(int b) override {
        if (tensor_count == 2) {
            d_ptrs[0] += lasts[0] * b;
            d_ptrs[1] += lasts[1] * b;
            return;
        }
        for (int a = 0; a < tensor_count; a++) {
            d_ptrs[a] += lasts[a] * b;
        }
    }
    inline void backup_d_ptr() override {
        if (tensor_count == 2) {
            d_ptrs[0] -= lasts[0];
            d_ptrs[1] -= lasts[1];
            return;
        }
        for (int a = 0; a < tensor_count; a++) {
            d_ptrs[a] -= lasts[a];
        }
    }

    inline std::vector<long> next() {
        if (tensor_count == 2) {
            OUTER_LOOP_2TC(TensorIterator::_ndim, coordinates, shape_m1,
                           strides_back, d_ptrs);
            return d_ptrs;
        }
        for (int a = 0; a < tensor_count; a++) {
            for (int i = TensorIterator::_ndim - 2; i >= 0; i--) {
                if (coordinates.at(a, i) < shape_m1[i]) {
                    coordinates.at(a, i) += 1;
                    d_ptrs[a] += strides.at(a, i);  //[a][i];
                    break;
                } else {
                    coordinates.at(a, i) = 0;
                    d_ptrs[a] -= strides_back.at(a, i);  //[a][i];
                }
            }
            d_ptrs[a] -= strides_back.at_back(a);  //[a].back();
        }
        return d_ptrs;
    }

    inline std::vector<long> get_strides() {
        std::vector<long> return_(tensor_count);
        for (int a = 0; a < tensor_count; a++) {
            return_[a] = strides.at_back(a);
        }
        return return_;
    }
    long tensor_count = 1;

    std::vector<long> d_ptrs;
    //    private:
};

}  // namespace sail