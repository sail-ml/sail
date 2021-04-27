#include "Tensor.h"
#include "dtypes.h"

#include <iostream>

#define MAX_VAL 320000

int main() {
    double x[MAX_VAL];
    double y[MAX_VAL];
    int ndim = 1;
    Dtype dt = Dtype::sFloat64;
    TensorSize st = {8};
    TensorSize sh = {MAX_VAL};

    for (int i = 0; i < MAX_VAL; i++) {
        x[i] = 0.01;
        y[i] = 0.21;
    }

    void* xt = static_cast<void*>(x);
    void* yt = static_cast<void*>(y);

    sail::Tensor t1 = sail::Tensor(ndim, xt, dt, st, sh);
    sail::Tensor t2 = sail::Tensor(ndim, yt, dt, st, sh);

    // for (int i = 0; i < 2000; i ++) {
    //     std::cout << i << std::endl;
    //     sail::Tensor new_ = t1 + t2;
    //     new_.free();
    // }

    std::cout << *(double*)t1[0].data << std::endl;
    t1[0];

    t1.free();
    t2.free();

    return 0;
}