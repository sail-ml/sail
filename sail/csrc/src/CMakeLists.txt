cmake_minimum_required(VERSION 2.8)
project(sail_c LANGUAGES CXX) # CUDA
# find_package(CUDA REQUIRED)

file(GLOB_RECURSE headers CONFIGURE_DEPENDS "*.h")
file(GLOB_RECURSE files CONFIGURE_DEPENDS "*.cpp")

include(FindBLAS)

add_library(src STATIC  ${headers} ${files} )#"cuda/cuda_math.h" "cuda/cuda_math.cu")
if (MKL_FOUND) 
target_include_directories(src PUBLIC ${MKL_INCLUDE_DIR})
target_link_libraries(src PUBLIC ${MKL_LIBRARIES})
target_link_libraries(src PUBLIC ${MKL_OPENMP_LIBRARY})
target_link_libraries(src PUBLIC ${MKL_SCALAPACK_LIBRARIES})
target_link_libraries(src PUBLIC ${MKL_CDFT_LIBRARIES})
else()
message(WARNING "MKL not found")
endif()
target_include_directories(src PRIVATE "../libs/xsimd/include")
# if (BLAS_FOUND)
#     target_compile_options(src PRIVATE ${BLAS_LINKER_FLAGS})
#     target_link_libraries(src ${BLAS_LIBRARIES})
# endif ()
