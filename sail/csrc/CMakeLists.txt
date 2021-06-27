# set(CMAKE_CXX_COMPILER "gcc")
cmake_minimum_required(VERSION 2.8.12)
project(sail_c LANGUAGES CXX C) # CUDA
# enable_language(CUDA)

file(GLOB_RECURSE PYTHON_BIND_HEADERS "python/*.h")
file(GLOB_RECURSE PYTHON_BIND_FILES "python/*.cpp")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE 1)

# ADD_DEFINITIONS(-DUSE_AVX) # NEED FOR AVX2 SUPPORT IN KERNELS
FIND_PACKAGE( Boost 1.40 COMPONENTS program_options REQUIRED )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -w -fpermissive -Wno-unused-variable -Wno-pointer-arith -fPIC -pthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -fopenmp -ffast-math -march=native")

if (COVERAGE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage -fprofile-abs-path -lgcov")# -march=haswell -mavx2 -ftree-vectorize -fopt-info-loop-optimized")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage -fprofile-abs-path -lgcov")# -march=haswell -mavx2 -ftree-vectorize -fopt-info-loop-optimized")
endif()

set(CMAKE_DIR_TEMP ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_CURRENT_SOURCE_DIR "core/")

add_subdirectory(core)
set(CMAKE_CURRENT_SOURCE_DIR ${CMAKE_DIR_TEMP})
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


# BASE SAIL PACKAGE
add_library(sail_c MODULE "python/sail_def.cpp" ${PYTHON_BIND_HEADERS})
target_link_libraries(sail_c PRIVATE core)
target_include_directories(sail_c PUBLIC ${SAIL_PYTHON_INCLUDE_DIRS})
target_include_directories(sail_c PUBLIC ./)
TARGET_LINK_LIBRARIES( sail_c LINK_PUBLIC ${Boost_LIBRARIES} )
if (COVERAGE)
target_link_libraries(sail_c PUBLIC "-lgcov")
endif()
target_include_directories(sail_c PRIVATE "libs/xsimd/include")


# MODULES
add_library(modules MODULE "python/module_def.cpp" ${PYTHON_BIND_HEADERS})
target_link_libraries(modules PRIVATE core)
target_include_directories(modules PUBLIC ${SAIL_PYTHON_INCLUDE_DIRS})
target_include_directories(modules PUBLIC ./)
TARGET_LINK_LIBRARIES( modules LINK_PUBLIC ${Boost_LIBRARIES} )
if (COVERAGE)
target_link_libraries(modules PUBLIC "-lgcov")
endif()
target_include_directories(modules PRIVATE "libs/xsimd/include")

# loss
add_library(losses MODULE "python/loss_def.cpp" ${PYTHON_BIND_HEADERS})
target_link_libraries(losses PRIVATE core)
target_include_directories(losses PUBLIC ${SAIL_PYTHON_INCLUDE_DIRS})
target_include_directories(losses PUBLIC ./)
TARGET_LINK_LIBRARIES( losses LINK_PUBLIC ${Boost_LIBRARIES} )
if (COVERAGE)
target_link_libraries(losses PUBLIC "-lgcov")
endif()
target_include_directories(losses PRIVATE "libs/xsimd/include")

# optimizers
add_library(optimizers MODULE "python/optimizer_def.cpp" ${PYTHON_BIND_HEADERS})
target_link_libraries(optimizers PRIVATE core)
target_include_directories(optimizers PUBLIC ${SAIL_PYTHON_INCLUDE_DIRS})
target_include_directories(optimizers PUBLIC ./)
TARGET_LINK_LIBRARIES( optimizers LINK_PUBLIC ${Boost_LIBRARIES} )
if (COVERAGE)
target_link_libraries(optimizers PUBLIC "-lgcov")
endif()
target_include_directories(optimizers PRIVATE "libs/xsimd/include")

# optimizers
add_library(random MODULE "python/random_def.cpp" ${PYTHON_BIND_HEADERS})
target_link_libraries(random PRIVATE core)
target_include_directories(random PUBLIC ${SAIL_PYTHON_INCLUDE_DIRS})
target_include_directories(random PUBLIC ./)
TARGET_LINK_LIBRARIES( random LINK_PUBLIC ${Boost_LIBRARIES} )
if (COVERAGE)
target_link_libraries(random PUBLIC "-lgcov")
endif()
target_include_directories(random PRIVATE "libs/xsimd/include")


file(GLOB_RECURSE TEST_FILES "test/*.cpp")

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(test ${TEST_FILES})
target_link_libraries(test PUBLIC core)
target_include_directories(test PUBLIC ./)
target_link_libraries(test PUBLIC gtest_main)
target_include_directories(test PRIVATE "libs/xsimd/include")
if (COVERAGE)
target_link_libraries(optimizers PUBLIC "-lgcov")
endif()
include(GoogleTest)
gtest_discover_tests(test)