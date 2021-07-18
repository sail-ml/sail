#pragma once
#include "Tensor.h"
#include "dtypes.h"
#include "kernels/Kernel.h"
namespace sail {
namespace modules {

inline Dtype default_dtype = Dtype::sFloat32;

using TensorVector = std::vector<Tensor>;

class Module {
   public:
    TensorVector params;

    explicit Module() = default;
    virtual void forward(){};

    void register_param(Tensor& t) {
        t.requires_grad = true;
        params.push_back(t);
    }
    void register_params(TensorVector& p) {
        for (const auto& t : p) {
            params.push_back(t);
        }
    }
};
}  // namespace modules
}  // namespace sail
