// allow-impl-in-header

#pragma once

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
    explicit Tensor() = default;

    TensorBody::pointer body;

    bool requires_grad = false;
    bool was_requires_grad = false;
    std::shared_ptr<autograd::Function> fcn = nullptr;

    bool is_grad = false;

    void swap(Tensor& t);

    Tensor(Tensor& old, bool _requires_grad)
        : body(old.body.get(), false), requires_grad(_requires_grad){};
    Tensor(TensorBody::pointer data, bool _requires_grad)
        : body(std::move(data)), requires_grad(_requires_grad){};

    Tensor(const Tensor&) = default;
    Tensor(Tensor&&) = default;

    Tensor& operator=(const Tensor& x) & {  // NOLINT
        body = x.body;
        requires_grad = x.requires_grad;
        fcn = x.fcn;
        return *this;
    }
    Tensor& operator=(Tensor&& x) & {  // NOLINT
        body = std::move(x.body);
        requires_grad = x.requires_grad;
        fcn = std::move(x.fcn);
        return *this;
    }

    void clear_grad();
    void clear_function();

    long numel() const;
    long len() const;

    Dtype get_dtype() const;

    TensorShape get_shape() const;

    void* get_data() const;
    alignemnt_information get_info() const;
    bool is_view() const;

    Tensor cast(const Dtype dt) const;
    Tensor reshape(const TensorShape& new_shape) const;
    Tensor _inplace_reshape(const TensorShape& new_shape);
    Tensor expand_dims(const int dim) const;
    Tensor _expand_dims_inplace(const int dim);
    Tensor squeeze(const int dim) const;
    long getTotalSize();

    template <typename T>
    T get() {
        T result;
        if (is_scalar()) {
            dispatch_all_numeric_types(get_dtype(), [&](auto pt) {
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

    int get_body_ref_count();
    void free();
    void swap_body(Tensor& t);

    TensorBody::pointer get_body() const;

    long int* get_shape_ptr();
    bool is_scalar() const;
    bool has_grad();
    int get_np_type_num();

    void set_shape(const TensorShape& s);
    void set_view();
    void set_data(void* data);

    long get_ndim() const;
    long ndim() const;
    Tensor get_grad() const;
    void set_grad(Tensor& g);

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
};

std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
Tensor operator+(Numeric n, Tensor& te);
Tensor operator/(Numeric n, Tensor& te);
Tensor operator-(Numeric n, Tensor& te);
Tensor operator*(Numeric n, Tensor& te);

int _numel(TensorSize _shape);

}  // namespace sail
