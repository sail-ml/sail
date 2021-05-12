#pragma once

#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
#include "TensorBody.h"

#include "dtypes.h"
#include "error.h"
#include "tensor_shape.h"
#include "types.h"

#define MAKE_PTR(value) std::shared_ptr<TensorBody>(new TensorBody(value));

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

    TensorBody::pointer body;

    int ndim;
    int arr_numel;
    bool requires_grad;
    bool has_grad = false;
    // std::shared_ptr<Tensor> grad;
    // std::unique_ptr<Tensor> grad;
    Tensor* grad = nullptr;
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
    // void* data = nullptr;
    bool freed = false;

    bool is_grad = false;

    // Tensor::~Tensor() = default;

    // Tensor(Tensor& old, bool _requires_grad)
    //     : body(intrusive_ptr<TensorBody>::reclaim(old.body.get())),
    //       requires_grad(_requires_grad){};
    // Tensor(intrusive_ptr<TensorBody>& data, bool _requires_grad)
    //     : body(std::move(data)), requires_grad(_requires_grad){};
    Tensor(Tensor& old, bool _requires_grad)
        : body(old.body.get(), false), requires_grad(_requires_grad){};
    Tensor(TensorBody::pointer data, bool _requires_grad)
        : body(std::move(data)), requires_grad(_requires_grad){};

    Tensor(const Tensor&) = default;
    Tensor(Tensor&&) = default;

    Tensor& operator=(const Tensor& x) & {
        body = x.body;
        requires_grad = x.requires_grad;
        return *this;
    }
    Tensor& operator=(Tensor&& x) & {
        body = std::move(x.body);
        requires_grad = std::move(x.requires_grad);
        return *this;
    }

    // // RG
    // Tensor& operator=(const Tensor& x) & {
    //     std::cout << "tensor copy assign" << std::endl;
    //     body = x.body;
    //     return *this;
    // }
    // Tensor& operator=(Tensor&& x) & {
    //     std::cout << "tensor move assign" << std::endl;
    //     body = std::move(x.body);
    //     return *this;
    // }

    long numel() const { return body.get()->get_shape().numel(); }

    Dtype get_dtype() const { return body.get()->get_dtype(); }

    TensorShape get_shape() const { return body.get()->get_shape(); }

    void* get_data() const { return body.get()->get_data(); }
    alignemnt_information get_info() const { return body.get()->get_info(); }
    bool is_view() const { return body.get()->is_view(); }

    Tensor cast(const Dtype dt);
    Tensor reshape(const TensorShape new_shape);
    Tensor expand_dims(const int dim);
    Tensor squeeze(const int dim);
    long getTotalSize();

    void free();

    long int* get_shape_ptr();
    bool is_scalar();
    int get_np_type_num();

    int get_ndim();

    void backward();
    void backward(Tensor& grad);

    Tensor operator+(const Tensor& t);
    Tensor operator-(const Tensor& t);
    Tensor operator-();
    Tensor operator*(const Tensor& t);
    Tensor operator/(const Tensor& t);
    Tensor operator[](const int t) const;

    friend std::ostream& operator<<(std::ostream& os, const Tensor& dt);

    Tensor sum();

    void register_op(autograd::Function* new_func);

    //    private:
};

std::ostream& operator<<(std::ostream& os, Tensor& tensor);

inline int _numel(TensorSize _shape) {
    auto size = 1;
    for (long value : _shape) {
        size = size * value;
    }
    return size;
}

}  // namespace sail
