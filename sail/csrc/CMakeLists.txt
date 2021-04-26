# set(CMAKE_CXX_COMPILER "gcc")
cmake_minimum_required(VERSION 2.8.12)
project(sail_c LANGUAGES CUDA CXX C)
enable_language(CUDA)

file(GLOB_RECURSE SRC_FILES [*.csrc])
file(GLOB_RECURSE PYTHON_BIND_HEADERS "python/*.h")
file(GLOB_RECURSE PYTHON_BIND_FILES "python/*.cpp")
# message(STATUS, ${PYTHON_BIND_FILES})
# add_custom_command(
#         COMMAND python template_converter.py ${SRC_FILES}
# )

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE 1)

ADD_DEFINITIONS(-DUSE_AVX2) # NEED FOR AVX2 SUPPORT IN KERNELS

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -Wextra -fPIC -Ofast -pthread -march=native -fpermissive -fopenmp -ffast-math -mavx2")# -march=haswell -mavx2 -ftree-vectorize -fopt-info-loop-optimized")


set(PYBIND11_PATH "libs/pybind11")

add_subdirectory(${PYBIND11_PATH})
include_directories(${PYBIND11_PATH})

set(CONAN_PATH "libs/conan")

# include(${CONAN_PATH}/conanbuildinfo.cmake) # Include Conan-generated file
# conan_basic_setup()

add_subdirectory(src)
# add_subdirectory(python)
# add_subdirectory(src/cuda)
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

add_library(sail_c MODULE ${PYTHON_BIND_HEADERS} ${PYTHON_BIND_FILES})
# target_include_directories(sail_c PUBLIC "$<BUILD_INTERFACE:${RANGE_PATH}/include>")
# target_link_libraries(sail_c ${CONAN_LIBS})
target_link_libraries(sail_c PRIVATE pybind11::module pybind11::lto pybind11::windows_extras)
target_link_libraries(sail_c PRIVATE src)


add_executable(test src/test.cpp)
target_link_libraries(test PRIVATE pybind11::module pybind11::lto pybind11::windows_extras)
target_link_libraries(test PRIVATE src)


# target_link_libraries(sail_c ${PYTHON_LIBRARIES})

# pybind11_extension(sail_c)
# pybind11_strip(sail_c)

