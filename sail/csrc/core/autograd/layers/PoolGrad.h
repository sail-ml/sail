
#pragma once

#include <iostream>
#include <vector>
#include "Tensor.h"
#include "autograd/function.h"
#include "onednn/pooling.h"
#include "ops/ops.h"

#ifdef MKLDNN
#include <dnnl.hpp>
using namespace dnnl;
using dnnl::pooling_v2_forward;
using tag = memory::format_tag;
using dt = memory::data_type;
#endif

namespace sail {

namespace autograd {

using TensorVector = std::vector<Tensor>;
#ifdef MKLDNN

class MaxPool2D : public Function {
   public:
    std::vector<long> strides;
    std::string padding_mode;

    TensorShape kernel_size;

    std::shared_ptr<onednn::OneDNNMaxPoolingParams> params;
    pooling_v2_forward::primitive_desc desc;

    MaxPool2D(std::shared_ptr<onednn::OneDNNMaxPoolingParams> params,
              pooling_v2_forward::primitive_desc desc)
        : params(std::move(params)), desc(std::move(desc)){};
    ~MaxPool2D() override = default;

    Tensor forward(TensorVector inputs) override;
    TensorVector backward(Tensor& grad) override;
};
#endif

}  // namespace autograd
}  // namespace sail