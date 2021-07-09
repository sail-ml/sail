#pragma once

#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
#include "TensorBody.h"

#include "constants.h"
#include "dtypes.h"
#include "exception.h"
#include "numeric.h"
#include "slice.h"
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

    bool requires_grad = false;
    bool was_requires_grad = false;
    // std::shared_ptr<Tensor> grad;
    // std::unique_ptr<Tensor> grad;
    // Tensor* grad = nullptr;
    std::shared_ptr<autograd::Function> fcn = nullptr;

    bool is_grad = false;

    void swap(Tensor& t) {
        bool t_rq = requires_grad;
        bool t_is_grad = is_grad;
        std::shared_ptr<autograd::Function> t_fcn = fcn;
        TensorBody::pointer t_body = body;

        body = t.body;
        fcn = t.fcn;
        requires_grad = t.requires_grad;
        is_grad = t.is_grad;

        t.body = t_body;
        t.fcn = t_fcn;
        t.requires_grad = t_rq;
        t.is_grad = t_is_grad;
        return t;
    }

    Tensor(Tensor& old, bool _requires_grad)
        : body(old.body.get(), false), requires_grad(_requires_grad){};
    Tensor(TensorBody::pointer data, bool _requires_grad)
        : body(std::move(data)), requires_grad(_requires_grad){};

    Tensor(const Tensor&) = default;
    Tensor(Tensor&&) = default;

    Tensor& operator=(const Tensor& x) & {
        body = x.body;
        requires_grad = x.requires_grad;
        fcn = x.fcn;
        return *this;
    }
    Tensor& operator=(Tensor&& x) & {
        body = std::move(x.body);
        requires_grad = std::move(x.requires_grad);
        fcn = std::move(x.fcn);
        return *this;
    }

    void clear_grad() { body.get()->clear_grad(); }
    void clear_function() { fcn = nullptr; }

    long numel() const { return body.get()->get_shape().numel(); }
    long len() const { return body.get()->get_shape().shape[0]; }

    Dtype get_dtype() const { return body.get()->get_dtype(); }

    TensorShape get_shape() const { return body.get()->get_shape(); }

    void* get_data() const { return body.get()->get_data(); }
    alignemnt_information get_info() const { return body.get()->get_info(); }
    bool is_view() const { return body.get()->is_view(); }

    Tensor cast(const Dtype dt);
    Tensor reshape(const TensorShape& new_shape) const;
    Tensor _inplace_reshape(const TensorShape& new_shape) const;
    Tensor expand_dims(const int dim);
    Tensor _expand_dims_inplace(const int dim);
    Tensor squeeze(const int dim);
    long getTotalSize();

    template <typename T>
    T get() {
        T result;
        if (is_scalar()) {
            dispatch_all_types(get_dtype(), [&](auto pt) {
                using TT = typename decltype(pt)::type;
                result = static_cast<TT*>(get_data())[0];
            });
            return result;
        } else {
            THROW_ERROR_DETAILED(SailCError,
                                 "Cannot get value from non scalar tensor");
        }
        return result;
    }

    int get_body_ref_count() { return body.get()->get_ref_count(); }

    void free();
    void Tensor::swap_body(Tensor& t);

    TensorBody::pointer get_body() { return body; }

    long int* get_shape_ptr();
    bool is_scalar() const;
    inline bool has_grad() { return body.get()->has_grad(); }
    int get_np_type_num();

    void set_shape(const TensorShape& s) { body.get()->set_shape(s); }
    void set_view() { body.get()->set_is_view(true); }

    int get_ndim() const { return get_shape().ndim(); }
    Tensor get_grad() const { return body.get()->get_grad(); }
    void set_grad(Tensor& g) { body.get()->set_grad(g); }

    void backward();
    void backward(Tensor& grad);

    Tensor slice(long start, long stop, long axis = 0);
    Tensor slice(Slice slice);

    Tensor assign(const Tensor& other);
    Tensor fill(const Numeric& other);

    Tensor operator+(const Tensor& t);
    Tensor operator+(const Numeric n);

    Tensor operator+=(const Tensor& t);
    Tensor operator+=(const Numeric n);

    Tensor operator-(const Tensor& t);
    Tensor operator-(const Numeric n);
    Tensor operator-();

    Tensor operator*(const Tensor& t);
    Tensor operator*(const Numeric n);

    Tensor operator/(const Tensor& t);
    Tensor operator/(const Numeric n);

    Tensor operator[](const int t) const;

    Tensor operator==(const Tensor& other);
    Tensor operator>=(const Tensor& other);
    Tensor operator<=(const Tensor& other);
    Tensor operator>(const Tensor& other);
    Tensor operator<(const Tensor& other);

    Tensor transpose();
    Tensor transpose(const LongVec& axes);

    friend std::ostream& operator<<(std::ostream& os, const Tensor& dt);

    Tensor sum();

    void register_op(autograd::Function* new_func);

    //    private:
};

std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
Tensor operator+(Numeric n, Tensor& te);
Tensor operator/(Numeric n, Tensor& te);
Tensor operator-(Numeric n, Tensor& te);
Tensor operator*(Numeric n, Tensor& te);

inline int _numel(TensorSize _shape) {
    auto size = 1;
    for (long value : _shape) {
        size = size * value;
    }
    return size;
}

}  // namespace sail
