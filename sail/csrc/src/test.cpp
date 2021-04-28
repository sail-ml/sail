#include "Tensor.h"
#include "dtypes.h"
#include "ops/ops.h"

#include <iostream>

#define MAX_VAL 3

int main() {
    double x[MAX_VAL];
    double y[MAX_VAL];
    int ndim = 1;
    Dtype dt = Dtype::sFloat64;
    TensorSize st = {8};
    TensorSize sh = {MAX_VAL};

    for (int i = 0; i < MAX_VAL; i++) {
        x[i] = 2.21;
        y[i] = 3.21;
    }

    void* xt = static_cast<void*>(x);
    void* yt = static_cast<void*>(y);

    sail::Tensor t1 = sail::Tensor(ndim, xt, dt, st, sh);
    sail::Tensor t2 = sail::Tensor(ndim, yt, dt, st, sh);

    sail::Tensor t3 = sail::ops::cast(t1, Dtype::sInt32);
    std::cout << *(double*)(t1.data) << std::endl;
    std::cout << *(int32_t*)(t3.data) << std::endl;

    t1[0];

    t1.free();
    t2.free();

    return 0;
}