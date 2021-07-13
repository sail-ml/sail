#pragma once
#include "Tensor.h"
#include "modules/module.h"
#include "tensor_shape.h"

#ifdef MKLDNN
#include "onednn/pooling.h"
#endif

namespace sail {
namespace modules {

class MaxPool2D : public Module {
   public:
    std::string padding_mode;

    std::vector<long> strides;
    std::vector<long> kernel_size;

#ifdef MKLDNN
    std::shared_ptr<onednn::OneDNNMaxPoolingParams> params = nullptr;
    std::shared_ptr<onednn::OneDNNMaxPooling> layer = nullptr;
#endif

    MaxPool2D(std::vector<long> kernel_size, std::vector<long> _strides,
              std::string _padding_mode = "valid")
        : kernel_size(kernel_size),
          strides(_strides),
          padding_mode(_padding_mode){};
    MaxPool2D(std::vector<long> kernel_size,
              std::string _padding_mode = "valid")
        : kernel_size(kernel_size),
          strides(kernel_size),
          padding_mode(_padding_mode){};

    MaxPool2D(long _kernel_size, long _strides,
              std::string _padding_mode = "valid")
        : MaxPool2D({_kernel_size, _kernel_size}, {_strides, _strides},
                    _padding_mode){};
    MaxPool2D(long _kernel_size, std::string _padding_mode = "valid")
        : MaxPool2D({_kernel_size, _kernel_size}, {_kernel_size, _kernel_size},
                    _padding_mode){};

    Tensor forward(Tensor& input);
};

}  // namespace modules
}  // namespace sail
