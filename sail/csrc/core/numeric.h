#pragma once

#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
#include "TensorBody.h"

#include "dtypes.h"
#include "exception.h"
#include "tensor_shape.h"
#include "types.h"
namespace sail {

template <typename T>
TensorBody::pointer from_single_value(T value, Dtype dt);

class Numeric {
   public:
    TensorBody::pointer t;
    Numeric(int i);
    Numeric(int64_t i);
    Numeric(double i);
    Numeric(float i);
    TensorBody::pointer get();
};
}  // namespace sail