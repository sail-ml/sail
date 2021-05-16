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

    TensorBody::pointer body;

    bool requires_grad;
    bool has_grad = false;
    // std::shared_ptr<Tensor> grad;
    // std::unique_ptr<Tensor> grad;
    Tensor* grad = nullptr;
    autograd::Function* fcn = nullptr;

    bool is_grad = false;

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

    long numel() const { return body.get()->get_shape().numel(); }

    Dtype get_dtype() const { return body.get()->get_dtype(); }

    TensorShape get_shape() const { return body.get()->get_shape(); }

    void* get_data() const { return body.get()->get_data(); }
    alignemnt_information get_info() const { return body.get()->get_info(); }
    bool is_view() const { return body.get()->is_view(); }

    Tensor cast(const Dtype dt);
    Tensor reshape(const TensorShape& new_shape) const;
    Tensor expand_dims(const int dim);
    Tensor squeeze(const int dim);
    long getTotalSize();

    int get_body_ref_count() { return body.get()->get_ref_count(); }

    void free();

    long int* get_shape_ptr();
    bool is_scalar();
    int get_np_type_num();

    void set_shape(TensorShape& s) { body.get()->set_shape(s); }

    int get_ndim() const { return get_shape().ndim(); }

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
