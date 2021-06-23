cmake_minimum_required(VERSION 2.8)
project(sail_c LANGUAGES CXX) # CUDA
# find_package(CUDA REQUIRED)


file(GLOB_RECURSE headers "*.h")
file(GLOB_RECURSE files "*.cpp")


include(FindBLAS)

add_library(core STATIC ${headers} ${files} )#"cuda/cuda_math.h" "cuda/cuda_math.cu")
target_include_directories(core PUBLIC ./)
target_include_directories(core PUBLIC ${SAIL_INCLUDE_DIRS})
if (COVERAGE)
target_link_libraries(core PUBLIC "-lgcov")
endif()
target_link_libraries(core PUBLIC ${SAIL_LIBRARIES})

target_include_directories(core PRIVATE "../libs/xsimd/include")




