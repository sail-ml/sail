
#pragma once

#include <iostream>
#include <vector>
#include "Tensor.h"
#include "autograd/function.h"
#include "ops/ops.h"

#ifdef MKLDNN
#include <dnnl.hpp>
using namespace dnnl;
using dnnl::inner_product_forward;
using tag = memory::format_tag;
using dt = memory::data_type;
#endif

namespace sail {

namespace autograd {

using TensorVector = std::vector<Tensor>;

class Conv2D : public Function {
   public:
    Tensor cols;
    Tensor flat_kernel;
    std::vector<long> strides;
    std::string padding_mode;

    long kh, kw;
    long pad_y, pad_x;

    Conv2D(std::vector<long> strides, std::string padding_mode)
        : strides(strides), padding_mode(padding_mode){};
    // RefTensorVector arg_storage;
    Tensor forward(TensorVector inputs);
    TensorVector backward(Tensor& grad);
    ~Conv2D() {}
};

#ifdef MKLDNN
class Conv2DMKLDNN : public Function {
   public:
    Tensor cols;
    Tensor flat_kernel;
    std::vector<long> strides;

    long kh, kw;
    std::vector<long> padding_l;
    std::vector<long> padding_r;
    Conv2DMKLDNN(std::vector<long> pad_y, std::vector<long> pad_x,
                 std::vector<long> strides)
        : padding_l(pad_y), padding_r(pad_x), strides(strides){};
    // RefTensorVector arg_storage;
    Tensor forward(TensorVector inputs);
    TensorVector backward(Tensor& grad);
    ~Conv2DMKLDNN() {}
};

#endif

}  // namespace autograd
}  // namespace sail