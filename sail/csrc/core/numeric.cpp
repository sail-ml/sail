#include "numeric.h"
#include "Tensor.h"
#include "TensorBody.h"
#include "dtypes.h"
#include "factories.h"
#include "tensor_shape.h"
namespace sail {

template <typename T>
TensorBody::pointer from_single_value(T value, Dtype dt) {
    TensorShape shape = TensorShape({1});
    TensorBody::pointer body =
        TensorBody::pointer(new TensorBody(dt, shape), true);
    dispatch_all_numeric_types(dt, [&](auto pt) {
        using Tensor_T = typename decltype(pt)::type;
        Tensor_T val = static_cast<Tensor_T>(value);
        Tensor_T* data = static_cast<Tensor_T*>(body->get_data());
        data[0] = val;
    });
    return body;
}

Numeric::Numeric(int i) {
    Dtype dt = min_type((long)i);
    t = from_single_value<int>(i, dt);
}
Numeric::Numeric(int64_t i) {
    Dtype dt = min_type(i);
    t = from_single_value<int64_t>(i, dt);
}
Numeric::Numeric(double i) {
    Dtype dt = min_type(i);
    t = from_single_value<double>(i, dt);
}
Numeric::Numeric(float i) {
    Dtype dt = min_type((double)i);
    t = from_single_value<float>(i, dt);
}
TensorBody::pointer Numeric::get() const { return t; }  // namespace sail
}  // namespace sail