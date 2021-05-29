# set(CMAKE_CXX_COMPILER "gcc")
cmake_minimum_required(VERSION 2.8.12)
project(sail_c LANGUAGES CXX C) # CUDA
# enable_language(CUDA)

file(GLOB_RECURSE SRC_FILES [*.csrc])
file(GLOB_RECURSE PYTHON_BIND_HEADERS "python/*.h")
file(GLOB_RECURSE PYTHON_BIND_FILES "python/*.cpp")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE 1)

# ADD_DEFINITIONS(-DUSE_AVX2) # NEED FOR AVX2 SUPPORT IN KERNELS
FIND_PACKAGE( Boost 1.40 COMPONENTS program_options REQUIRED )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -w -fPIC -Ofast -pthread -march=native -fpermissive -ffast-math -mavx2 -Wno-unused-variable -Wno-pointer-arith")# -march=haswell -mavx2 -ftree-vectorize -fopt-info-loop-optimized")


add_subdirectory(src)
# find_package(PythonLibs REQUIRED)
# find_package(Python3 REQUIRED COMPONENTS NumPy)
# include_directories(${PYTHON_INCLUDE_DIRS})

set(SAIL_PYTHON_INCLUDE_DIRS
    ${PYTHON_INCLUDE_DIR}
    ${PYTHON_LIBRARY}
    ${PYTHON_EXECUTABLE}
    ${PYTHON_NUMPY_INCLUDE_DIR}
    )

include(FindBLAS)

add_library(sail_c MODULE ${PYTHON_BIND_HEADERS} ${PYTHON_BIND_FILES})
target_link_libraries(sail_c PRIVATE src)
# target_link_libraries(sail_c PUBLIC Python3::NumPy)
# target_link_libraries(sail_c PUBLIC Python3::NumPy)
target_include_directories(sail_c PUBLIC ${SAIL_PYTHON_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES( sail_c LINK_PUBLIC ${Boost_LIBRARIES} )
target_include_directories(sail_c PRIVATE "libs/xsimd/include")


# if (BLAS_FOUND)
#     target_compile_options(sail_c PRIVATE ${BLAS_LINKER_FLAGS})
#     target_link_libraries(sail_c ${BLAS_LIBRARIES})
# endif ()



include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(test test/test_cpp/test.cpp)
target_link_libraries(test src)
target_link_libraries(test gtest_main)
target_include_directories(test PRIVATE "libs/xsimd/include")


# if (BLAS_FOUND)
#     target_compile_options(test PRIVATE ${BLAS_LINKER_FLAGS})
#     target_link_libraries(test ${BLAS_LIBRARIES})
# endif ()


# include(GoogleTest)
# gtest_discover_tests(test)