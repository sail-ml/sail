#!/bin/bash

set -ex

wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
# add to your apt sources keyring so that archives signed with this key will be trusted.
sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
# remove the public key
rm GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB

echo "deb https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
add-apt-repository "deb https://apt.repos.intel.com/oneapi all main"

# apt install -y intel-oneapi-dnnl-devel intel-oneapi-mkl intel-oneapi-mpi-devel
apt install -y intel-oneapi-dnnl-devel intel-oneapi-mkl intel-oneapi-mkl-common-devel-2021.2.0 intel-oneapi-mpi-devel
# apt install -y intel-oneapi-mkl-common-devel-2021.2.0 
apt-get install libatlas-base-dev liblapack-dev libblas-dev

# Cleanup package manager
# apt-get autoclean && apt-get clean
# rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
