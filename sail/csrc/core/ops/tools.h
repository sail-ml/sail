#include <iostream>

#include "Tensor.h"
#include "autograd/autograd.h"
#include "factories.h"

namespace sail {

inline bool must_broadcast(const Tensor& t1, const Tensor& t2) {
    TensorSize shape1 = t1.get_shape().shape;
    TensorSize shape2 = t2.get_shape().shape;

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
                throw SailCError("shapes cannot be broadcasted together");
            } else {
                bc = true;
            }
        }
        idx_2 -= 1;
    }
    return bc;
}

inline std::vector<long> merge_shapes(std::vector<long> s1,
                                      std::vector<long> s2) {
    std::vector<long> merged;
    std::vector<long> larger_shape = (s1.size() > s2.size()) ? s1 : s2;
    std::vector<long> smaller_shape = (s1.size() > s2.size()) ? s2 : s1;
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
}  // namespace sail