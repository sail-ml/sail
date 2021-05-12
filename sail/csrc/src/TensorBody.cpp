#include "TensorBody.h"
#include "dtypes.h"
#include "tensor_shape.h"
#include "utils.h"

namespace sail {

TensorBody::TensorBody(void* _data, Dtype _dtype, TensorShape _shape,
                       bool _view = false)
    : data(_data),
      dtype(_dtype),
      shape(_shape),
      view(_view),
      info(getAlignment(_dtype)),
      refcount_(0){};
TensorBody::TensorBody(Dtype _dtype, TensorShape _shape, bool _view = false) {
    dtype = _dtype;
    shape = _shape;
    info = getAlignment(_dtype);
    refcount_ = 0;
    view = _view;
    data = _malloc_align(shape.numel(), info.alignment, info.dtype_size);
};

void* TensorBody::get_data() { return data; }
Dtype TensorBody::get_dtype() { return dtype; }
TensorShape TensorBody::get_shape() { return shape; }
alignemnt_information TensorBody::get_info() { return info; }
bool TensorBody::is_view() { return view; }

}  // namespace sail