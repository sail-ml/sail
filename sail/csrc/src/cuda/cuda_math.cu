#include <iostream>
#include <stdio.h>
#include <typeinfo>
#include <cuda_runtime.h>
#include <cuda.h>
// #include <pybind11/pybind11.h>
// #include <pybind11/numpy.h>
__global__ void add(double* ptr1, double* ptr2, double* ptr3, int shape0) {
    for (size_t idx = 0; idx < shape0; idx++) {
        printf("a %f\n", ptr1[idx]);
        ptr3[idx] = ptr1[idx] + ptr2[idx];
    }
}

inline void print_pointer(std::string name, void* pointer) {
    std::cout << name << ", " << pointer << std::endl;
}

double* cupy_add_(double* ptr11, size_t size1, int shape0, double* ptr22, size_t size2, double* ptr33) {
    cudaSetDevice(0);

    // # if __CUDA_ARCH__>=200

    // #endif
    // py::buffer_info buf1i = buf1.request(), buf2i = buf2.request();
    // auto result = py::array_t<double>(size1);

    // double* ptr3 = (double*)malloc(shape0*sizeof(double));
    // double* cuda_ptr1;
    // double* cuda_ptr2;
    // double* cuda_ptr3;

    // print_pointer("ptr11", ptr11);
    // print_pointer("ptr22", ptr22);
    // std::cout << *ptr11 << std::endl;
    
    // double *ptr1 = static_cast<double *>(ptr11);
    // double *ptr2 = static_cast<double *>(ptr22);
    // // double *ptr3 = static_cast<double *>(ptr3_);
    // print_pointer("ptr1", ptr1);
    // print_pointer("ptr2", ptr2);
    // // double *cuda_ptr3 = static_cast<double *>(cuda_ptr3_);

    // cudaMallocManaged(&cuda_ptr1, shape0*sizeof(double));
    // cudaMallocManaged(&cuda_ptr2, shape0*sizeof(double));
    // cudaMallocManaged(&cuda_ptr3, shape0*sizeof(double));
    // cudaMemcpy(cuda_ptr1, ptr11, shape0*sizeof(double), cudaMemcpyDefault );
    // cudaMemcpy(cuda_ptr2, ptr22, shape0*sizeof(double), cudaMemcpyDefault );
    // // cudaMemcpy(cuda_ptr3, ptr1, shape0*sizeof(double), cudaMemcpyDefault );

    cudaDeviceSynchronize();

    // cudaDeviceProp prop;
    // cudaGetDeviceProperties(&prop,0);


    // print_pointer("cuda_ptr1", cuda_ptr1);
    // print_pointer("cuda_ptr3", cuda_ptr3);
    // // print_pointer("cuda_ptr2", cuda_ptr2);
    // print_pointer("cuda_ptr3", cuda_ptr3);
    
    
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 512*1024*1024);
    add<<<1,1>>>(ptr11, ptr22, ptr33, shape0);
    cudaDeviceSynchronize();
    // free(ptr3);
    // print_pointer("ptr3", ptr3);
    // print_pointer("cuda_ptr3", cuda_ptr3);

    // std::cout <<static_cast<void*>(ptr1) << std::endl;
    // std::cout << *cuda_ptr3 << std::endl;
    // cudaMemcpy(ptr3, cuda_ptr3, shape0*sizeof(double), cudaMemcpyDeviceToHost);
    // std::flush();
    // std::cout << ptr3 << std::endl;
    // return cuda_ptr3;
    return ptr33;
    // return cuda_ptr3;
}

