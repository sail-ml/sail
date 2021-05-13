#include "Tensor.h"

#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
#include "TensorBody.h"

#include "autograd/autograd.h"
#include "cuda/cuda_math.h"
#include "dtypes.h"
#include "factories.h"
#include "kernels/kernel.h"
#include "ops/ops.h"
#include "tensor_shape.h"
#include "types.h"
#include "utils.h"

#define MIN_NUMEL_PAR 4096

using Tensor = sail::Tensor;
using RefTensorVector = std::vector<Tensor*>;

namespace sail {

std::ostream& operator<<(std::ostream& os, Tensor& tensor) {
    ReprKernel().execute(tensor, os);
    return os;
}

long Tensor::getTotalSize() {
    long size = GetDtypeSize(get_dtype());
    for (long value : get_shape().shape) {
        size = size * value;
    }
    return size;
}

Tensor Tensor::reshape(const TensorShape new_shape) {
    int s = new_shape.numel();
    if (s != numel()) {
        throw DimensionError{"Cannot reshape tensor of shape ",
                             get_shape().get_string(), " to ",
                             new_shape.get_string()};
    }

    body->set_shape(new_shape);
    return *this;
}

Tensor Tensor::expand_dims(const int dim) {
    TensorShape s = get_shape();
    s.insert_one(dim);
    // TensorSize s = shape;
    // s.insert(s.begin() + dim, 1);
    reshape(s);
    return *this;
}

Tensor Tensor::squeeze(const int dim) {
    get_shape().remove_one(dim);
    // ops::squeeze(*this, dim);
    return *this;
}

bool Tensor::is_scalar() {
    if (numel() == 1) {
        return true;
    }
    return false;
}

void Tensor::register_op(autograd::Function* new_func) { fcn = new_func; }

void Tensor::free() {
    // std::cout << "FREEING TENSOR" << std::endl;
    // if (data != NULL) {
    //     std::free(data);
    //     data = NULL;
    // }
}

long int* Tensor::get_shape_ptr() { return body->get_shape_ptr(); }

int Tensor::get_np_type_num() { return get_np_type_numFromDtype(get_dtype()); }

// todo - move to op
Tensor Tensor::cast(const Dtype dt) {
    Tensor casted = ops::cast(*this, dt);
    return casted;
}
// // operators

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

    int new_ndim = (get_ndim()) - 1;
    Tensor e = make_view(new_ndim, new_ptr, get_dtype(),
                         TensorShape(new_shape, new_strides));

    return e;
}

Tensor Tensor::operator+(const Tensor& other) { return ops::add(*this, other); }
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
// UNARY OPS

Tensor Tensor::sum() { return ops::sum(*this); }

void Tensor::backward() {
    // double data = 1.0;
    Tensor t = one_scalar(get_dtype());
    t = ops::broadcast_to(t, TensorShape(get_shape().shape));
    backward(t);
}
void Tensor::backward(Tensor& _grad) {
    // _grad.owner = false;
    std::cout << "_grad.is_view()" << std::endl;
    std::cout << _grad.is_view() << std::endl;
    // std::cout << g->owner << std::endl;

    // for (Tensor i : fcn->)
    if (requires_grad) {
        Tensor* g = new Tensor(std::move(_grad));
        std::cout << "calculating" << std::endl;
        if (has_grad) {
            // _grad = (*grad) + _grad;
            // grad = &_grad;
            // grad = g;
            grad = g;
            // grad.get()->owner = true;
        } else {
            has_grad = true;
            // grad = &_grad;
            // std::cout << "grad.owner" << std::endl;
            // std::cout << _grad.owner << std::endl;
            // grad = g;  // std::make_shared<Tensor>(_grad);
            grad = g;
            // std::cout << _grad.owner << std::endl;

            // std::cout << grad.get()->owner << std::endl;
            // grad = &_grad;

            // memcpy(grad, &_grad, sizeof(Tensor));
        }
        // _grad.owner = false;
        //     // std::cout << grad.get()->is_grad << std::endl;
        if (fcn != nullptr) {  ////// THIS NEEDS TO CHANGE

            RefTensorVector grad_arglist = fcn->arg_storage;
            std::vector<Tensor> new_grads = fcn->backward(*g);

            for (int i = 0; i < new_grads.size(); i++) {
                if (grad_arglist[i]->requires_grad) {
                    std::cout << i << ", " << new_grads[i] << std::endl;
                    Tensor g = new_grads[i];
                    grad_arglist[i]->backward(g);
                    // grad_arglist[i]->backward(new_grads[i]);
                }
            }
            //         }
        }
    }
}

}  // namespace sail
