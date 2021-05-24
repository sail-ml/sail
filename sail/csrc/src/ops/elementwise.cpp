/*
################################################################################
#                  THIS CODE IS AUTOGENERATED FROM A TEMPLATE                  #
#                 TO MAKE CHANGES, EDIT THE ORIGINAL .src FILE                 #
################################################################################
*/

#include <iostream>

#include "../Tensor.h"
#include "../autograd/autograd.h"
#include "../factories.h"
#include "../kernels/kernel.h"
#include "elementwise.h"

#define MAX(a, b) (((a.ndim) > (b.ndim)) ? (a) : (b))
#define MIN(a, b) (((a.ndim) < (b.ndim)) ? (a) : (b))

namespace sail {

namespace ops {
using TensorVector = std::vector<Tensor>;

bool must_broadcast(const Tensor& t1, const Tensor& t2) {
    TensorSize shape1 = t1.get_shape().shape;
    TensorSize shape2 = t2.get_shape().shape;
    // Tensor& larger_shape = (new Tensor());
    // Tensor& smaller_shape = (new Tensor());
    TensorShape larger_shape, smaller_shape;
    bool bc = false;
    if (t1.get_ndim() > t2.get_ndim()) {
        larger_shape = t1.get_shape();
        smaller_shape = t2.get_shape();
    } else {
        larger_shape = t2.get_shape();
        smaller_shape = t1.get_shape();
    }

    int idx_2 = smaller_shape.ndim() - 1;
    for (int i = larger_shape.ndim() - 1; i >= 0; i--) {
        if (idx_2 < 0) {
            return true;
        }
        if (larger_shape.shape[i] != smaller_shape.shape[idx_2]) {
            if (larger_shape.shape[i] != 1 && smaller_shape.shape[idx_2] != 1) {
                throw "shapes cannot be broadcasted together";
            } else {
                bc = true;
            }
        }
        idx_2 -= 1;
    }
    return bc;
}

std::vector<long> merge_shapes(std::vector<long> s1, std::vector<long> s2) {
    std::vector<long> merged;
    std::vector<long> larger_shape = (s1.size() > s2.size()) ? s1 : s2;
    std::vector<long> smaller_shape = (s1.size() < s2.size()) ? s1 : s2;
    int idx_2 = smaller_shape.size() - 1;
    for (int i = larger_shape.size() - 1; i >= 0; i--) {
        if (idx_2 < 0) {
            merged.push_back(larger_shape[i]);
        } else {
            if (larger_shape[i] == smaller_shape[idx_2]) {
                merged.push_back(larger_shape[i]);
            } else {
                merged.push_back((larger_shape[i] > smaller_shape[idx_2])
                                     ? larger_shape[i]
                                     : smaller_shape[idx_2]);
            }
        }
        idx_2 -= 1;
    }
    std::reverse(merged.begin(), merged.end());
    return merged;
}

/** begin block
 * name = [add, subtract, divide, multiply]
 * kName = [Add, Sub, Divide, Multiply]
 * agName = [Add, Subtract, Divide, Multiply]
 */


Tensor add(const Tensor& tensor1, const Tensor& tensor2) {
    // std::cout << &tensor1 << std::endl;
    // TensorShape s = tensor1.get_shape();
    // Tensor empty_tensor = empty(s.ndim(), tensor1.get_dtype(), s);

    if (tensor1.requires_grad || tensor2.requires_grad) {
        // empty_tensor = (new autograd::Add())->apply({&tensor1,
        // &tensor2});
        TensorVector vec;
        vec.emplace_back(tensor1);
        vec.emplace_back(tensor2);
        Tensor empty_tensor = (new autograd::Add())->apply(vec);

        // std::endl;

        return empty_tensor;
    }

    Tensor empty_tensor = empty_like(tensor1);

    bool broadcast = must_broadcast(tensor1, tensor2);
    if (broadcast) {
        std::vector<long> new_ =
            merge_shapes(tensor1.get_shape().shape, tensor2.get_shape().shape);
        TensorShape s = TensorShape(new_);
        empty_tensor.set_shape(s);
    }

    bool t1_scalar = tensor1.is_scalar();
    bool t2_scalar = tensor2.is_scalar();

    if ((t1_scalar && t2_scalar) || (!t1_scalar && !t2_scalar)) {
        AddTTKernel().execute(tensor1, tensor2, empty_tensor, broadcast);
    } else if (t1_scalar && !t2_scalar) {
        // empty_tensor = empty(s.ndim(), tensor2.get_dtype(), s);
        AddTSKernel().execute(tensor2, tensor1, empty_tensor, broadcast);
    } else {
        // empty_tensor = empty(s.ndim(), tensor1.get_dtype(), s);
        AddTSKernel().execute(tensor1, tensor2, empty_tensor, broadcast);
    }

    return empty_tensor;
}



Tensor subtract(const Tensor& tensor1, const Tensor& tensor2) {
    // std::cout << &tensor1 << std::endl;
    // TensorShape s = tensor1.get_shape();
    // Tensor empty_tensor = empty(s.ndim(), tensor1.get_dtype(), s);

    if (tensor1.requires_grad || tensor2.requires_grad) {
        // empty_tensor = (new autograd::Subtract())->apply({&tensor1,
        // &tensor2});
        TensorVector vec;
        vec.emplace_back(tensor1);
        vec.emplace_back(tensor2);
        Tensor empty_tensor = (new autograd::Subtract())->apply(vec);

        // std::endl;

        return empty_tensor;
    }

    Tensor empty_tensor = empty_like(tensor1);

    bool broadcast = must_broadcast(tensor1, tensor2);
    if (broadcast) {
        std::vector<long> new_ =
            merge_shapes(tensor1.get_shape().shape, tensor2.get_shape().shape);
        TensorShape s = TensorShape(new_);
        empty_tensor.set_shape(s);
    }

    bool t1_scalar = tensor1.is_scalar();
    bool t2_scalar = tensor2.is_scalar();

    if ((t1_scalar && t2_scalar) || (!t1_scalar && !t2_scalar)) {
        SubTTKernel().execute(tensor1, tensor2, empty_tensor, broadcast);
    } else if (t1_scalar && !t2_scalar) {
        // empty_tensor = empty(s.ndim(), tensor2.get_dtype(), s);
        SubTSKernel().execute(tensor2, tensor1, empty_tensor, broadcast);
    } else {
        // empty_tensor = empty(s.ndim(), tensor1.get_dtype(), s);
        SubTSKernel().execute(tensor1, tensor2, empty_tensor, broadcast);
    }

    return empty_tensor;
}



Tensor divide(const Tensor& tensor1, const Tensor& tensor2) {
    // std::cout << &tensor1 << std::endl;
    // TensorShape s = tensor1.get_shape();
    // Tensor empty_tensor = empty(s.ndim(), tensor1.get_dtype(), s);

    if (tensor1.requires_grad || tensor2.requires_grad) {
        // empty_tensor = (new autograd::Divide())->apply({&tensor1,
        // &tensor2});
        TensorVector vec;
        vec.emplace_back(tensor1);
        vec.emplace_back(tensor2);
        Tensor empty_tensor = (new autograd::Divide())->apply(vec);

        // std::endl;

        return empty_tensor;
    }

    Tensor empty_tensor = empty_like(tensor1);

    bool broadcast = must_broadcast(tensor1, tensor2);
    if (broadcast) {
        std::vector<long> new_ =
            merge_shapes(tensor1.get_shape().shape, tensor2.get_shape().shape);
        TensorShape s = TensorShape(new_);
        empty_tensor.set_shape(s);
    }

    bool t1_scalar = tensor1.is_scalar();
    bool t2_scalar = tensor2.is_scalar();

    if ((t1_scalar && t2_scalar) || (!t1_scalar && !t2_scalar)) {
        DivideTTKernel().execute(tensor1, tensor2, empty_tensor, broadcast);
    } else if (t1_scalar && !t2_scalar) {
        // empty_tensor = empty(s.ndim(), tensor2.get_dtype(), s);
        DivideTSKernel().execute(tensor2, tensor1, empty_tensor, broadcast);
    } else {
        // empty_tensor = empty(s.ndim(), tensor1.get_dtype(), s);
        DivideTSKernel().execute(tensor1, tensor2, empty_tensor, broadcast);
    }

    return empty_tensor;
}



Tensor multiply(const Tensor& tensor1, const Tensor& tensor2) {
    // std::cout << &tensor1 << std::endl;
    // TensorShape s = tensor1.get_shape();
    // Tensor empty_tensor = empty(s.ndim(), tensor1.get_dtype(), s);

    if (tensor1.requires_grad || tensor2.requires_grad) {
        // empty_tensor = (new autograd::Multiply())->apply({&tensor1,
        // &tensor2});
        TensorVector vec;
        vec.emplace_back(tensor1);
        vec.emplace_back(tensor2);
        Tensor empty_tensor = (new autograd::Multiply())->apply(vec);

        // std::endl;

        return empty_tensor;
    }

    Tensor empty_tensor = empty_like(tensor1);

    bool broadcast = must_broadcast(tensor1, tensor2);
    if (broadcast) {
        std::vector<long> new_ =
            merge_shapes(tensor1.get_shape().shape, tensor2.get_shape().shape);
        TensorShape s = TensorShape(new_);
        empty_tensor.set_shape(s);
    }

    bool t1_scalar = tensor1.is_scalar();
    bool t2_scalar = tensor2.is_scalar();

    if ((t1_scalar && t2_scalar) || (!t1_scalar && !t2_scalar)) {
        MultiplyTTKernel().execute(tensor1, tensor2, empty_tensor, broadcast);
    } else if (t1_scalar && !t2_scalar) {
        // empty_tensor = empty(s.ndim(), tensor2.get_dtype(), s);
        MultiplyTSKernel().execute(tensor2, tensor1, empty_tensor, broadcast);
    } else {
        // empty_tensor = empty(s.ndim(), tensor1.get_dtype(), s);
        MultiplyTSKernel().execute(tensor1, tensor2, empty_tensor, broadcast);
    }

    return empty_tensor;
}

/** end block **/

}  // namespace ops

}  // namespace sail
