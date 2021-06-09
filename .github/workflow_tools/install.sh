#!/bin/bash

set -ex

wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
# add to your apt sources keyring so that archives signed with this key will be trusted.
sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
# remove the public key
rm GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB

echo "deb https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
add-apt-repository "deb https://apt.repos.intel.com/oneapi all main"
apt install intel-oneapi-dnnl
apt install intel-oneapi-dnnl-devel

apt install intel-oneapi-mkl-common-2021.2.0 intel-oneapi-mkl-common-devel-2021.2.0 intel-oneapi-mkl-devel intel-oneapi-mkl-devel-2021.2.0 intel-oneapi-mpi-2021.2.0 intel-oneapi-mpi-devel-2021.2.0


chmod +x ./.github/workflow_tools/install_avx2.sh
chmod +x ./.github/workflow_tools/install_boost.sh
chmod +x ./.github/workflow_tools/install_cmake.sh
chmod +x ./.github/workflow_tools/install_gcc.sh

# ./builder/install_avx2.sh
./.github/workflow_tools/install_boost.sh
./.github/workflow_tools/install_cmake.sh
./.github/workflow_tools/install_gcc.sh
