#pragma once

#include <iostream>
#include <memory>
#include <vector>

#include "dtypes.h"
#include "error.h"
#include "tensor_shape.h"
#include "types.h"

#define MAKE_PTR(value) value;  // std::shared_ptr<void>(value, std::free)

namespace sail {
namespace autograd {
class Function;
}  // namespace autograd

class Tensor {
   public:
    explicit Tensor(){};

    // smart desruction stuff
    // long data_references;
    // Tensor* parent_ref = nullptr;
    // bool is_reference = false;

    // void* data;
    int ndim;
    int arr_numel;
    bool requires_grad;
    bool has_grad = false;
    std::shared_ptr<Tensor> grad;
    // Tensor* grad;
    Dtype dtype;
    TensorSize shape;
    TensorSize strides;
    alignemnt_information info;
    TensorShape shape_details;

    autograd::Function* fcn = nullptr;

    bool broadcasted = false;
    bool view = false;
    bool owner = true;
    TensorShape view_base_shape;
    TensorShape old_shape;

    // std::shared_ptr<void> data;
    void* data = nullptr;

    Tensor::~Tensor() {
        // std::cout << "Trying to destruct" << std::endl;
        // std::cout << this << std::endl;
        // std::cout << owner << std::endl;
        if (owner) {
            // std::cout << "freeing" << std::endl;
            std::free(data);
            delete fcn;
            owner = false;
        }
    }

    Tensor(const Tensor& t) {
        // std::cout << "copy" << std::endl;
        ndim = t.ndim;
        arr_numel = t.arr_numel;
        requires_grad = t.requires_grad;
        has_grad = t.has_grad;
        grad = t.grad;
        dtype = t.dtype;
        info = t.info;
        fcn = t.fcn;
        broadcasted = t.broadcasted;
        view = t.view;
        shape_details = t.shape_details;

        data = t.data;
        grad = t.grad;
        fcn = t.fcn;
        owner = false;
    }

    Tensor(Tensor&& t) noexcept {
        // std::cout << "move called" << std::endl;
        ndim = t.ndim;
        arr_numel = t.arr_numel;
        requires_grad = t.requires_grad;
        has_grad = t.has_grad;
        dtype = t.dtype;
        info = t.info;
        broadcasted = t.broadcasted;
        view = t.view;
        shape_details = t.shape_details;

        grad = t.grad;
        // t.grad = nullptr;
        data = t.data;  //
        t.data = nullptr;
        fcn = t.fcn;
        t.fcn = nullptr;

        owner = true;
        t.owner = false;
    }

    Tensor& operator=(const Tensor& t) {
        // std::cout << "copy assignment" << std::endl;
        ndim = t.ndim;
        arr_numel = t.arr_numel;
        requires_grad = t.requires_grad;
        has_grad = t.has_grad;
        grad = t.grad;
        dtype = t.dtype;
        info = t.info;
        fcn = t.fcn;
        broadcasted = t.broadcasted;
        view = t.view;
        shape_details = t.shape_details;

        data = t.data;
        grad = t.grad;
        fcn = t.fcn;
        owner = false;

        return *this;
    }

    Tensor& operator=(Tensor&& t) noexcept {
        // std::cout << "move assignment" << std::endl;
        ndim = t.ndim;
        arr_numel = t.arr_numel;
        requires_grad = t.requires_grad;
        has_grad = t.has_grad;
        dtype = t.dtype;
        info = t.info;
        broadcasted = t.broadcasted;
        view = t.view;
        shape_details = t.shape_details;

        grad = t.grad;
        // t.grad = nullptr;
        data = t.data;  //
        t.data = nullptr;
        fcn = t.fcn;
        t.fcn = nullptr;

        this->owner = true;
        t.owner = false;
        // std::cout << this << std::endl;
        // std::cout << &t << std::endl;

        return *this;
    }

    Tensor(int& ndims, void*& data, Dtype& dt, TensorShape shape_data);
    Tensor(int& ndims, void*& data, Dtype& dt, TensorShape shape_data,
           bool requires_grad);

    static Tensor move(int& ndims, void*& data, Dtype& dt,
                       TensorShape shape_data);

    Tensor cast(const Dtype dt);
    Tensor reshape(const TensorShape new_shape);
    Tensor expand_dims(const int dim);
    Tensor squeeze(const int dim);
    long getTotalSize();

    void free();
    inline void* get_data() { return data; }
    inline void* get_data() const { return data; }
    inline void* get_shared_ptr() const { return data; }

    void set_data(void* new_data);
    void set_data(const std::shared_ptr<void>& new_data);

    long int* get_shape_ptr();
    bool is_scalar();
    int get_np_type_num();

    int numel() const;
    int get_ndim();

    void backward();
    void backward(Tensor grad);

    Tensor operator+(const Tensor& t);
    Tensor operator-(const Tensor& t);
    Tensor operator-();
    Tensor operator*(const Tensor& t);
    Tensor operator/(const Tensor& t);
    Tensor operator[](const int t) const;

    Tensor sum();

    void register_op(autograd::Function* new_func);

    //    private:
};

inline int _numel(TensorSize _shape) {
    auto size = 1;
    for (long value : _shape) {
        size = size * value;
    }
    return size;
}

}  // namespace sail
