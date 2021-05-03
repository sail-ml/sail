
#pragma once

#include "reduction_function.h"
#include <iostream>
#include <vector>
#include "../Tensor.h"
#include "../ops/broadcast.h"
#include "../ops/reduction.h"
#include "function.h"

namespace sail {

namespace autograd {

using TensorVector = std::vector<Tensor>;

std::string Sum::getName() { return "SumOp"; }
inline Tensor Sum::forward(TensorVector inputs) {
    std::cout << Function::arg_storage[0].shape_details.get_string()
              << std::endl;

    return ops::sum(inputs[0]);
}
inline TensorVector Sum::backward(Tensor grad) {
    Tensor full_size =
        ops::broadcast_to(grad, Function::arg_storage[0].shape_details);
    return {full_size};
}

}  // namespace autograd
}  // namespace sail