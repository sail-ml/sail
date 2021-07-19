
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

    long kh = 0;
    long kw = 0;
    long pad_y = 0;
    long pad_x = 0;

    Conv2D(std::vector<long> strides, std::string padding_mode)
        : strides(std::move(strides)), padding_mode(std::move(padding_mode)){};
    ~Conv2D() override = default;

    Tensor forward(TensorVector inputs) override;
    TensorVector backward(Tensor& grad) override;
};

#ifdef MKLDNN
class Conv2DMKLDNN : public Function {
   public:
    Tensor cols;
    Tensor flat_kernel;
    std::vector<long> strides;

    long kh = 0;
    long kw = 0;
    std::vector<long> padding_l;
    std::vector<long> padding_r;
    Conv2DMKLDNN(std::vector<long> pad_y, std::vector<long> pad_x,
                 std::vector<long> strides)
        : padding_l(std::move(pad_y)),
          padding_r(std::move(pad_x)),
          strides(std::move(strides)){};
    ~Conv2DMKLDNN() override = default;

    Tensor forward(TensorVector inputs) override;
    TensorVector backward(Tensor& grad) override;
};

#endif

}  // namespace autograd
}  // namespace sail