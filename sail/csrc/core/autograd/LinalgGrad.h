
#pragma once

#include <iostream>
#include <vector>
#include "Tensor.h"
#include "function.h"
#include "ops/ops.h"

namespace sail {

namespace autograd {

using TensorVector = std::vector<Tensor>;

class AddMM : public Function {
   public:
    explicit AddMM() = default;
    // TensorVector arg_storage;
    Tensor forward(TensorVector inputs) override;
    TensorVector backward(Tensor& grad) override;
};

class Matmul : public Function {
   public:
    std::string trans_a, trans_b;
    Matmul(std::string _trans_a, std::string _trans_b)
        : trans_a(std::move(_trans_a)), trans_b(std::move(_trans_b)){};
    // TensorVector arg_storage;
    Tensor forward(TensorVector inputs) override;
    TensorVector backward(Tensor& grad) override;
};

}  // namespace autograd
}  // namespace sail