#include <iostream>

#include "../Tensor.h"
#include "../factories.h"
#include "../kernels/kernel.h"

namespace sail {
namespace ops {

Tensor sum(const Tensor& tensor1) {
    Tensor empty_tensor = empty_scalar(tensor1.storage.dtype);
    SumTKernel().execute(tensor1, empty_tensor);
    return empty_tensor;
}
Tensor mean(const Tensor& tensor1) {
    Tensor empty_tensor = empty_scalar(tensor1.storage.dtype);
    MeanTKernel().execute(tensor1, empty_tensor);
    return empty_tensor;
}

}
} // end sail
