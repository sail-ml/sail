#include "TensorBody.h"
#include "Tensor.h"
#include "dtypes.h"
#include "factories.h"
#include "tensor_shape.h"
#include "utils.h"

namespace sail {

TensorBody::TensorBody(void* _data, Dtype _dtype, TensorShape _shape,
                       bool _view)
    : data(_data),
      dtype(_dtype),
      shape(new TensorShape(_shape)),
      view(_view),
      info(getAlignment(_dtype)),
      refcount_(0){};

TensorBody::TensorBody(Dtype _dtype, TensorShape _shape, bool _view) {
    dtype = _dtype;
    shape = new TensorShape(_shape);
    info = getAlignment(_dtype);
    refcount_ = 0;
    view = _view;
    data = _malloc_align(shape->numel(), info.alignment, info.dtype_size);
};

TensorBody::pointer TensorBody::create_owner() {
    TensorBody::pointer a = new TensorBody(data, dtype, get_shape(), view);
    int temp_ref = get_ref_count();
    refcount_ = a->get_ref_count();
    a->refcount_ = temp_ref;
    return a;
}

TensorBody::~TensorBody() {
    if (data != nullptr) {
        if (!view) {
            std::free(data);  // NOLINT
        }
        delete shape;
        if (_has_grad) {
            delete grad;
        }

        data = nullptr;
        shape = nullptr;
        grad = nullptr;
    } else {
        THROW_ERROR(SailCError,
                    "Cannot free a tensor that does not have any data");
    }
}

void TensorBody::set_grad(Tensor& t) {
    if (t.is_view()) {
        t = clone(t);
        grad = new Tensor(t.get_body(), t.requires_grad);
    } else {
        grad = new Tensor(t.get_body(), t.requires_grad);
    }
    _has_grad = true;
}
Tensor TensorBody::get_grad() { return *grad; }
void TensorBody::clear_grad() {
    delete grad;
    grad = nullptr;
    _has_grad = false;
}

void* TensorBody::get_data() { return data; }
void TensorBody::set_data(void* d) { data = d; }
Dtype TensorBody::get_dtype() { return dtype; }
TensorShape TensorBody::get_shape() { return *shape; }
alignemnt_information TensorBody::get_info() { return info; }
bool TensorBody::is_view() { return view; }
bool TensorBody::has_grad() { return _has_grad; }
void TensorBody::set_is_view(bool x) { view = x; }

int TensorBody::get_ref_count() { return (int)refcount_; }
void TensorBody::force_incref() {
    refcount_.fetch_add(1, std::memory_order_relaxed);
}

void TensorBody::set_shape(const TensorShape& s) {
    delete shape;
    shape = new TensorShape(s.shape, s.strides);
}

long int* TensorBody::get_shape_ptr() { return shape->get_shape_ptr(); }

}  // namespace sail