# Dockerfile for libcudacxx_base:host_x86_64_centos_7.6.1810__target_x86_64_centos_7.6.1810__gcc_4.8_cxx_11

FROM centos:7.6.1810

MAINTAINER Bryce Adelstein Lelbach <blelbach@nvidia.com>

ARG LIBCUDACXX_SKIP_BASE_TESTS_BUILD
ARG LIBCUDACXX_COMPUTE_ARCHS

###############################################################################
# BUILD: The following is invoked when the image is built.

SHELL ["/usr/bin/env", "bash", "-c"]

RUN yum -y --enablerepo=extras install epel-release\
 && yum -y updateinfo\
 && yum -y install which make gcc-c++ libstdc++-static llvm-devel clang python python-pip\
 && pip install lit\
 && mkdir -p /sw/gpgpu/libcudacxx/build\
 && mkdir -p /sw/gpgpu/libcudacxx/libcxx/build

# The distro doesn't have CMake 3.8 in its repositories, so we need to install
# it ourselves.
ADD https://github.com/Kitware/CMake/releases/download/v3.8.2/cmake-3.8.2-Linux-x86_64.sh /tmp/cmake.sh
RUN sh /tmp/cmake.sh --skip-license --prefix=/usr

# For debugging.
#RUN yum -y install gdb strace vim

# We use ADD here because it invalidates the cache for subsequent steps, which
# is what we want, as we need to rebuild if the sources have changed.

# Copy NVCC and the CUDA runtime from the source tree.
ADD bin /sw/gpgpu/bin

# Copy the core CUDA headers from the source tree.
ADD cuda/import/*.h* /sw/gpgpu/cuda/import/
ADD cuda/common/*.h* /sw/gpgpu/cuda/common/
ADD cuda/tools/ /sw/gpgpu/cuda/tools/
ADD opencl/import/cl_rel/CL/*.h* /sw/gpgpu/opencl/import/cl_rel/CL/

# Copy libcu++ sources from the source tree.
ADD libcudacxx /sw/gpgpu/libcudacxx

# List out everything in /sw before the build.
RUN echo "Contents of /sw:" && cd /sw/ && find

# Build libc++ and configure libc++ tests.
RUN set -o pipefail; cd /sw/gpgpu/libcudacxx/libcxx/build\
 && cmake ..\
 -DLIBCXX_INCLUDE_TESTS=ON\
 -DLIBCXX_INCLUDE_BENCHMARKS=OFF\
 -DLIBCXX_CXX_ABI=libsupc++\
 -DLIBCXX_TEST_STANDARD_VER=c++11\
 -DLIBCXX_ABI_UNSTABLE=ON\
 -DLLVM_CONFIG_PATH=$(which llvm-config)\
 -DCMAKE_C_COMPILER=gcc\
 -DCMAKE_CXX_COMPILER=g++\
 && make -j\
 2>&1 | tee /sw/gpgpu/libcudacxx/build/cmake_libcxx.log

# Configure libcu++ tests.
RUN set -o pipefail; cd /sw/gpgpu/libcudacxx/build\
 && cmake ..\
 -DLIBCXX_TEST_STANDARD_VER=c++11\
 -DLLVM_CONFIG_PATH=$(which llvm-config)\
 -DCMAKE_CXX_COMPILER=/sw/gpgpu/bin/x86_64_Linux_release/nvcc\
 -DLIBCXX_NVCC_HOST_COMPILER=g++\
 2>&1 | tee /sw/gpgpu/libcudacxx/build/cmake_libcudacxx.log

# Build tests if requested.
# NOTE: libc++ tests are disabled until we can setup libc++abi.
RUN set -o pipefail; cd /sw/gpgpu/libcudacxx\
 && LIBCUDACXX_COMPUTE_ARCHS=$LIBCUDACXX_COMPUTE_ARCHS\
 LIBCUDACXX_SKIP_BASE_TESTS_BUILD=$LIBCUDACXX_SKIP_BASE_TESTS_BUILD\
 /sw/gpgpu/libcudacxx/utils/nvidia/linux/perform_tests.bash\
 --skip-tests-runs\
 --skip-libcxx-tests\
 2>&1 | tee /sw/gpgpu/libcudacxx/build/build_lit_all.log

# Build tests for sm6x if requested.
RUN set -o pipefail; cd /sw/gpgpu/libcudacxx\
 && LIBCUDACXX_COMPUTE_ARCHS="60 61 62"\
 LIBCUDACXX_SKIP_BASE_TESTS_BUILD=$LIBCUDACXX_SKIP_BASE_TESTS_BUILD\
 /sw/gpgpu/libcudacxx/utils/nvidia/linux/perform_tests.bash\
 --skip-tests-runs\
 --skip-libcxx-tests\
 2>&1 | tee /sw/gpgpu/libcudacxx/build/build_lit_sm6x.log

# Build tests for sm7x if requested.
RUN set -o pipefail; cd /sw/gpgpu/libcudacxx\
 && LIBCUDACXX_COMPUTE_ARCHS="70 72 75"\
 LIBCUDACXX_SKIP_BASE_TESTS_BUILD=$LIBCUDACXX_SKIP_BASE_TESTS_BUILD\
 /sw/gpgpu/libcudacxx/utils/nvidia/linux/perform_tests.bash\
 --skip-tests-runs\
 --skip-libcxx-tests\
 2>&1 | tee /sw/gpgpu/libcudacxx/build/build_lit_sm7x.log

# Build tests for sm8x if requested.
RUN set -o pipefail; cd /sw/gpgpu/libcudacxx\
 && LIBCUDACXX_COMPUTE_ARCHS="80"\
 LIBCUDACXX_SKIP_BASE_TESTS_BUILD=$LIBCUDACXX_SKIP_BASE_TESTS_BUILD\
 /sw/gpgpu/libcudacxx/utils/nvidia/linux/perform_tests.bash\
 --skip-tests-runs\
 --skip-libcxx-tests\
 2>&1 | tee /sw/gpgpu/libcudacxx/build/build_lit_sm8x.log

# Package the logs up, because `docker container cp` doesn't support wildcards,
# and I want to limit the number of places we have to make changes when we ship
# new architectures.
RUN cd /sw/gpgpu/libcudacxx/build && tar -cvf logs.tar *.log

WORKDIR /sw/gpgpu/libcudacxx

