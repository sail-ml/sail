#pragma once

#include <iostream>
#include <vector>

#include "dtypes.h"
#include "error.h"
#include "types.h"

namespace sail {

class Tensor {
   public:
    explicit Tensor(){};

       void* data;
       int ndim;
       int arr_numel;
       Dtype dtype;
       TensorSize shape;
       TensorSize strides;
       alignemnt_information info;


//     explicit Tensor(TensorStorage storage);

    Tensor(int& ndims, void*& data, Dtype& dt, TensorSize& strides,
           TensorSize& shape); 
    static Tensor move(int& ndims, void*& data, Dtype& dt, TensorSize& strides,
           TensorSize& shape); 
//     Tensor(int ndims, void* data, Dtype dt, TensorSize strides,
//            TensorSize shape);



       Tensor cast(const Dtype dt);
       Tensor reshape(const TensorSize new_shape);
       Tensor expand_dims(const int dim);

       void free();

       long int* get_shape_ptr();
       bool is_scalar();
       int get_np_type_num();

       int numel() const;


    Tensor operator+(Tensor& t);
    Tensor operator-(Tensor& t);
    Tensor operator*(Tensor& t);
    Tensor operator/(Tensor& t);
    Tensor operator[](const int t);

    Tensor sum();

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
