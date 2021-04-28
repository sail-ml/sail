cmake_minimum_required(VERSION 2.8)
project(sail_c LANGUAGES CUDA CXX)
find_package(CUDA REQUIRED)

file(GLOB_RECURSE headers CONFIGURE_DEPENDS "*.h")
file(GLOB_RECURSE files CONFIGURE_DEPENDS "*.cpp")

include(FindBLAS)

cuda_add_library(src STATIC  ${headers} ${files} "cuda/cuda_math.h" "cuda/cuda_math.cu")
target_include_directories(src PRIVATE "../libs/xsimd/include")
if (BLAS_FOUND)
    target_compile_options(src PRIVATE ${BLAS_LINKER_FLAGS})
    target_link_libraries(src ${BLAS_LIBRARIES})
endif ()

