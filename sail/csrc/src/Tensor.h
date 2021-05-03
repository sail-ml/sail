#pragma once

#include <iostream>
#include <memory>
#include <vector>

#include "dtypes.h"
#include "error.h"
#include "tensor_shape.h"
#include "types.h"

namespace sail {
namespace autograd {
class Function;
}  // namespace autograd

class Tensor {
   public:
    explicit Tensor(){};

    void* data;
    int ndim;
    int arr_numel;
    bool requires_grad;
    Dtype dtype;
    TensorSize shape;
    TensorSize strides;
    alignemnt_information info;
    TensorShape shape_details;

    autograd::Function* fcn;

    //     explicit Tensor(TensorStorage storage);

    Tensor(int& ndims, void*& data, Dtype& dt, TensorShape shape_data);
    Tensor(int& ndims, void*& data, Dtype& dt, TensorShape shape_data,
           bool requires_grad);

    static Tensor move(int& ndims, void*& data, Dtype& dt,
                       TensorShape shape_data);
    //     Tensor(int ndims, void* data, Dtype dt, TensorSize strides,
    //            TensorSize shape);

    Tensor cast(const Dtype dt);
    Tensor reshape(const TensorShape new_shape);
    Tensor expand_dims(const int dim);
    Tensor squeeze(const int dim);
    long getTotalSize();

    void free();

    long int* get_shape_ptr();
    bool is_scalar();
    int get_np_type_num();

    int numel() const;
    int get_ndim();

    void backward();

    Tensor operator+(Tensor& t);
    Tensor operator-(Tensor& t);
    Tensor operator*(Tensor& t);
    Tensor operator/(Tensor& t);
    Tensor operator[](const int t);

    Tensor sum();

    void register_op(autograd::Function* new_func);

    //     Tensor cast(const Dtype dt);
    //     void inplace_cast(const Dtype dt);

    //     void reshape(const TensorSize s);

    //     Tensor expand_dims(const int dim);

    //     static Tensor createEmptyScalar(Dtype dt);

    //     void* data();

    //     bool is_scalar();

    //     Dtype dtype();

    //     void free();

    //     long int* get_shape_ptr();
    //     int get_np_type_num();

    //     TensorStorage storage;
};

inline int _numel(TensorSize _shape) {
    auto size = 1;
    for (long value : _shape) {
        size = size * value;
    }
    return size;
}

}  // namespace sail
