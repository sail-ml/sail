#include "Tensor.h"

#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
#include "TensorBody.h"

#include "autograd/autograd.h"
#include "dtypes.h"
#include "exception.h"
#include "factories.h"
#include "kernels/Kernel.h"
#include "numeric.h"
#include "ops/ops.h"
#include "slice.h"
#include "tensor_iterator.h"
#include "tensor_shape.h"
#include "types.h"
#include "utils.h"

#define MIN_NUMEL_PAR 4096

using Tensor = sail::Tensor;
using RefTensorVector = std::vector<Tensor*>;
using TensorVector = std::vector<Tensor>;

namespace sail {

std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    sail::internal::repr_stub(tensor, os);
    return os;
}

long Tensor::getTotalSize() {
    long size = GetDtypeSize(get_dtype());
    for (long value : get_shape().shape) {
        size = size * value;
    }
    return size;
}

Tensor Tensor::reshape(const TensorShape& new_shape) const {
    return ops::reshape(*this, new_shape);
}
Tensor Tensor::_inplace_reshape(const TensorShape& new_shape) {
    int s = new_shape.numel();
    if (s != numel()) {
        THROW_ERROR_DETAILED(DimensionError, "Cannot reshape tensor of shape ",
                             get_shape().get_string(), " to ",
                             new_shape.get_string());
    }
    set_shape(new_shape);
    return *this;
}

Tensor Tensor::transpose() const { return ops::transpose(*this); }
Tensor Tensor::transpose(const LongVec& axes) const {
    return ops::transpose(*this, axes);
}

Tensor Tensor::expand_dims(const int dim) const {
    int new_dim = dim;

    TensorShape s = body->get_shape();
    s.insert_one(new_dim);
    TensorShape x = TensorShape(s.shape);
    TensorBody::pointer new_body = TensorBody::pointer(
        new TensorBody(body->get_data(), body->get_dtype(), x,
                       /*is_view*/ true));
    Tensor new_tensor = Tensor(new_body, requires_grad);
    return new_tensor;
}
Tensor Tensor::_expand_dims_inplace(const int dim) {
    TensorShape s = body->get_shape();
    s.insert_one(dim);
    set_shape(s);
    return *this;
}

Tensor Tensor::squeeze(const int dim) const {
    int new_dim = dim;
    TensorShape s = body->get_shape();
    s.remove_one(new_dim);
    TensorShape x = TensorShape(s.shape);
    TensorBody::pointer new_body = TensorBody::pointer(
        new TensorBody(body->get_data(), body->get_dtype(), x,
                       /*is_view*/ true));
    Tensor new_tensor = Tensor(new_body, requires_grad);
    return new_tensor;
}

void Tensor::set_shape(const TensorShape& s) { body.get()->set_shape(s); }
void Tensor::set_view(bool val) { body.get()->set_is_view(val); }

long Tensor::get_ndim() const { return get_shape().ndim(); }
long Tensor::ndim() const { return get_shape().ndim(); }
Tensor Tensor::get_grad() const { return body.get()->get_grad(); }
void Tensor::set_grad(Tensor& g) { body.get()->set_grad(g); }

void Tensor::clear_grad() { body.get()->clear_grad(); }
void Tensor::clear_function() { fcn = nullptr; }

long Tensor::numel() const { return body.get()->get_shape().numel(); }
long Tensor::len() const { return body.get()->get_shape().shape[0]; }

int Tensor::get_body_ref_count() { return body.get()->get_ref_count(); }
TensorBody::pointer Tensor::get_body() const { return body; }
bool Tensor::has_grad() { return body.get()->has_grad(); }

Dtype Tensor::get_dtype() const { return body.get()->get_dtype(); }

TensorShape Tensor::get_shape() const { return body.get()->get_shape(); }

void* Tensor::get_data() const { return body.get()->get_data(); }
alignemnt_information Tensor::get_info() const {
    return body.get()->get_info();
}
bool Tensor::is_view() const { return body.get()->is_view(); }

bool Tensor::is_scalar() const {
    if (get_shape().shape.size() == 0) {
        return true;
    }
    return false;
}

bool Tensor::is_single() const {
    if ((get_shape().shape.size() == 0) ||
        (get_shape().shape.size() == 1 && get_shape().shape[0] == 1)) {
        return true;
    }
    return false;
}

void Tensor::register_op(autograd::Function* new_func) {
    fcn = std::shared_ptr<autograd::Function>(new_func);
}

long int* Tensor::get_shape_ptr() { return body->get_shape_ptr(); }

int Tensor::get_np_type_num() { return get_np_type_numFromDtype(get_dtype()); }

Tensor Tensor::cast(const Dtype dt) const {
    if (dt != get_dtype()) {
        Tensor casted = ops::cast(*this, dt);
        return casted;
    }
    return *this;
}

Tensor Tensor::assign(const Tensor& other) {
    ops::copy(*this, other);
    return *this;
}

Tensor Tensor::slice(long start, long stop, long step) {
    return slice(Slice({start, stop, step}));
}

Tensor Tensor::slice(Slice slice) {
    char* new_ptr;
    TensorSize new_shape;
    TensorSize new_strides;
    alignemnt_information info = get_info();
    TensorShape shape_details = get_shape();
    long offset = 0;
    int i = 0;
    char* data = (char*)get_data();

    for (std::vector<long> s : slice.slices) {
        if (s.size() == 2) {
            offset += (shape_details.strides[i] * info.dtype_size) * (s[0]);
            new_shape.push_back(s[1] - s[0]);
            new_strides.push_back(shape_details.strides[i]);
        } else if (s.size() == 3) {
            offset += (shape_details.strides[i] * info.dtype_size) * (s[0]);
            new_shape.push_back((s[1] - s[0]) / s[2]);
            new_strides.push_back(shape_details.strides[i] * s[2]);
        } else {
            new_shape.push_back(shape_details.shape[i]);
            new_strides.push_back(shape_details.strides[i]);
        }
        i += 1;
    }

    new_ptr = data + offset;
    for (int i = slice.size(); i < shape_details.ndim(); i++) {
        new_shape.push_back(shape_details.shape[i]);
        new_strides.push_back(shape_details.strides[i]);
    }

    Tensor e = make_view((void*)new_ptr, get_dtype(),
                         TensorShape(new_shape, new_strides));

    return e;
}

Tensor Tensor::operator[](const int index) const {
    char* new_ptr;
    TensorSize new_shape;
    TensorSize new_strides;
    alignemnt_information info = get_info();
    TensorShape shape_details = get_shape();
    char* data = (char*)get_data();

    if (is_scalar()) {
        THROW_ERROR_DETAILED(SailCError, "Cannot index a single value.");
    }
    long offset = 0;

    offset += (shape_details.strides[0] * info.dtype_size) * (index);
    new_ptr = data + offset;
    for (int i = 1; i < shape_details.ndim(); i++) {
        new_shape.push_back(shape_details.shape[i]);
        new_strides.push_back(shape_details.strides[i]);
    }

    Tensor e = make_view((void*)new_ptr, get_dtype(),
                         TensorShape(new_shape, new_strides));

    return e;
}

Tensor Tensor::operator+(const Tensor& other) { return ops::add(*this, other); }
Tensor Tensor::operator+(const Numeric other) {
    Tensor t = Tensor(other.get(), requires_grad);
    return *this + t;
}

Tensor Tensor::operator+=(const Tensor& other) {
    return ops::iadd(*this, other);
}
Tensor Tensor::operator+=(const Numeric other) {
    Tensor t = Tensor(other.get(), requires_grad);
    return ops::iadd(*this, t);
}

Tensor Tensor::operator-(const Tensor& other) {
    return ops::subtract(*this, other);
}
Tensor Tensor::operator-(const Numeric other) {
    Tensor t = Tensor(other.get(), requires_grad);
    return *this - t;
}

Tensor Tensor::operator*(const Tensor& other) {
    return ops::multiply(*this, other);
}
Tensor Tensor::operator*(const Numeric other) {
    Tensor t = Tensor(other.get(), requires_grad);
    return *this * t;
}

Tensor Tensor::operator/(const Tensor& other) {
    return ops::divide(*this, other);
}
Tensor Tensor::operator/(const Numeric other) {
    Tensor t = Tensor(other.get(), requires_grad);
    return *this / t;
}

Tensor Tensor::operator-() { return ops::negate(*this); }

Tensor Tensor::operator!=(const Tensor& other) {
    return ops::elementwise_ne(*this, other);
}
Tensor Tensor::operator==(const Tensor& other) {
    return ops::elementwise_equal(*this, other);
}
Tensor Tensor::operator>=(const Tensor& other) {
    return ops::elementwise_gte(*this, other);
}
Tensor Tensor::operator<=(const Tensor& other) {
    return ops::elementwise_lte(*this, other);
}
Tensor Tensor::operator>(const Tensor& other) {
    return ops::elementwise_gt(*this, other);
}
Tensor Tensor::operator<(const Tensor& other) {
    return ops::elementwise_lt(*this, other);
}

Tensor Tensor::operator!=(const Numeric& other) {
    Tensor t = Tensor(other.get(), requires_grad);
    return ops::elementwise_ne(*this, t);
}
Tensor Tensor::operator==(const Numeric& other) {
    Tensor t = Tensor(other.get(), requires_grad);
    return ops::elementwise_equal(*this, t);
}
Tensor Tensor::operator>=(const Numeric& other) {
    Tensor t = Tensor(other.get(), requires_grad);
    return ops::elementwise_gte(*this, t);
}
Tensor Tensor::operator<=(const Numeric& other) {
    Tensor t = Tensor(other.get(), requires_grad);
    return ops::elementwise_lte(*this, t);
}
Tensor Tensor::operator>(const Numeric& other) {
    Tensor t = Tensor(other.get(), requires_grad);
    return ops::elementwise_gt(*this, t);
}
Tensor Tensor::operator<(const Numeric& other) {
    Tensor t = Tensor(other.get(), requires_grad);
    return ops::elementwise_lt(*this, t);
}

Tensor Tensor::sum(int axis, bool keepdims) {
    return ops::sum(*this, axis, keepdims);
}

void Tensor::backward() {
    Tensor t = one_scalar(get_dtype());
    backward(t);
}
void Tensor::backward(Tensor& _grad) {
    if (requires_grad) {
        if (has_grad()) {
            Tensor ng = _grad + get_grad();
            set_grad(ng);
        } else {
            set_grad(_grad);
        }
        if (fcn != nullptr) {
            TensorVector grad_arglist = fcn->arg_storage;
            std::vector<Tensor> new_grads = fcn->backward(_grad);

            for (int i = 0; i < new_grads.size(); i++) {
                if (grad_arglist[i].requires_grad) {
                    Tensor grad_tensor = new_grads[i];
                    grad_arglist[i].backward(grad_tensor);
                }
            }
        }
    }
}

Tensor operator+(Numeric n, Tensor& te) {
    Tensor t = Tensor(n.get(), te.requires_grad);
    return t + te;
}
Tensor operator-(Numeric n, Tensor& te) {
    Tensor t = Tensor(n.get(), te.requires_grad);
    return t - te;
}
Tensor operator/(Numeric n, Tensor& te) {
    Tensor t = Tensor(n.get(), te.requires_grad);
    return t / te;
}
Tensor operator*(Numeric n, Tensor& te) {
    Tensor t = Tensor(n.get(), te.requires_grad);
    return t * te;
}

Tensor operator!=(Numeric n, Tensor& te) {
    Tensor t = Tensor(n.get(), te.requires_grad);
    return ops::elementwise_ne(t, te);
}
Tensor operator==(Numeric n, Tensor& te) {
    Tensor t = Tensor(n.get(), te.requires_grad);
    return ops::elementwise_equal(t, te);
}
Tensor operator>=(Numeric n, Tensor& te) {
    Tensor t = Tensor(n.get(), te.requires_grad);
    return ops::elementwise_gte(t, te);
}
Tensor operator<=(Numeric n, Tensor& te) {
    Tensor t = Tensor(n.get(), te.requires_grad);
    return ops::elementwise_lte(t, te);
}
Tensor operator>(Numeric n, Tensor& te) {
    Tensor t = Tensor(n.get(), te.requires_grad);
    return ops::elementwise_gt(t, te);
}
Tensor operator<(Numeric n, Tensor& te) {
    Tensor t = Tensor(n.get(), te.requires_grad);
    return ops::elementwise_lt(t, te);
}

}  // namespace sail
