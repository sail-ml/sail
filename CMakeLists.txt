# set(CMAKE_CXX_COMPILER "gcc")
cmake_minimum_required(VERSION 2.8.12)
project(sail_c LANGUAGES CUDA CXX C)
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE 1)

ADD_DEFINITIONS(-DUSE_AVX2) # NEED FOR AVX2 SUPPORT IN KERNELS

add_subdirectory(sail/csrc)

# set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fPIC -Ofast -pthread -march=native -fpermissive -fopenmp -ffast-math -mavx2 -Wno-unused-variable")# -march=haswell -mavx2 -ftree-vectorize -fopt-info-loop-optimized")


