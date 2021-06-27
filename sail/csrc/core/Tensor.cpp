#include "Tensor.h"

#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
#include "TensorBody.h"

#include "autograd/autograd.h"
// #include "cuda/cuda_math.h"
#include "dtypes.h"
#include "factories.h"
#include "kernels/Kernel.h"
#include "ops/ops.h"
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
Tensor Tensor::_inplace_reshape(const TensorShape& new_shape) const {
    int s = new_shape.numel();
    if (s != numel()) {
        THROW_ERROR_DETAILED(DimensionError, "Cannot reshape tensor of shape ",
                             get_shape().get_string(), " to ",
                             new_shape.get_string());
    }
    set_shape(new_shape);
    return *this;
}

Tensor Tensor::transpose() { return ops::transpose(*this); }
Tensor Tensor::transpose(const LongVec& axes) {
    return ops::transpose(*this, axes);
}

Tensor Tensor::expand_dims(const int dim) {
    int new_dim = dim;
    // if (dim < 0) {
    //     new_dim = dim + get_ndim() + 1;
    // }
    // if (new_dim < 0 || new_dim > get_ndim()) {
    //     throw DimensionError("dim must be in the range of [-ndim, ndim]");
    // }
    TensorShape s = body->get_shape();
    s.insert_one(new_dim);
    TensorShape x = TensorShape(s.shape);  // this is a super hacky fix
    TensorBody::pointer new_body = TensorBody::pointer(
        new TensorBody(body->get_data(), body->get_dtype(), x,
                       /*is_view*/ true));
    Tensor new_tensor = Tensor(new_body, requires_grad);
    return new_tensor;
}
Tensor Tensor::_expand_dims_inplace(const int dim) {
    int new_dim = dim;
    TensorShape s = body->get_shape();
    s.insert_one(dim);
    set_shape(s);
    return *this;
}

Tensor Tensor::squeeze(const int dim) {
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

bool Tensor::is_scalar() const {
    if (numel() == 1) {
        return true;
    }
    return false;
}

void Tensor::register_op(autograd::Function* new_func) {
    fcn = std::shared_ptr<autograd::Function>(new_func);
}

long int* Tensor::get_shape_ptr() { return body->get_shape_ptr(); }

int Tensor::get_np_type_num() { return get_np_type_numFromDtype(get_dtype()); }

// todo - move to op
Tensor Tensor::cast(const Dtype dt) {
    if (dt != get_dtype()) {
        Tensor casted = ops::cast(*this, dt);
        return casted;
    }
    return *this;
}

Tensor Tensor::operator[](const int index) const {
    void* new_ptr;
    TensorSize new_shape;
    TensorSize new_strides;
    alignemnt_information info = get_info();
    TensorShape shape_details = get_shape();
    long dim = shape_details.shape[0];
    long offset = 0;

    offset += (shape_details.strides[0] * info.dtype_size) * (index);
    new_ptr = get_data() + offset;
    for (int i = 1; i < shape_details.ndim(); i++) {
        new_shape.push_back(shape_details.shape[i]);
        new_strides.push_back(shape_details.strides[i]);
    }

    // std::cout << getVectorString(new_shape) << std::endl;

    Tensor e =
        make_view(new_ptr, get_dtype(), TensorShape(new_shape, new_strides));

    return e;
}

void Tensor::swap_body(Tensor& t) {
    TensorBody::pointer temp = body;
    body = t.body;
    t.body = temp;
}

Tensor Tensor::operator+(const Tensor& other) { return ops::add(*this, other); }
Tensor Tensor::operator+=(const Tensor& other) {
    return ops::iadd(*this, other);
}
Tensor Tensor::operator-(const Tensor& other) {
    return ops::subtract(*this, other);
}
Tensor Tensor::operator*(const Tensor& other) {
    return ops::multiply(*this, other);
}
Tensor Tensor::operator/(const Tensor& other) {
    return ops::divide(*this, other);
}

Tensor Tensor::operator-() { return ops::negate(*this); }

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

Tensor Tensor::sum() { return ops::sum(*this); }

void Tensor::backward() {
    // if (requires_grad) {
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
        if (fcn != nullptr) {  ////// THIS NEEDS TO CHANGE
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

}  // namespace sail
