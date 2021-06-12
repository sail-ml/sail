#pragma once
#include "Tensor.h"
#include "dtypes.h"
namespace sail {
namespace modules {

inline Dtype default_dtype = Dtype::sFloat32;

using TensorVector = std::vector<Tensor>;

class Module {
   public:
    TensorVector params;

    virtual explicit Module(){};
    virtual void forward(){};

    void register_param(Tensor& t) {
        t.requires_grad = true;
        params.push_back(t);
    }
};
}  // namespace modules
}  // namespace sail
