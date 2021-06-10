#!/bin/bash

set -ex

chmod +x ./docker/common/install_tools.sh
chmod +x ./docker/common/install_python.sh
chmod +x ./docker/common/install_boost.sh
chmod +x ./docker/common/install_cmake.sh
chmod +x ./docker/common/install_gcc.sh
chmod +x ./docker/common/install_oneapi.sh

./docker/common/install_tools.sh
./docker/common/install_python.sh
./docker/common/install_boost.sh
./docker/common/install_cmake.sh
./docker/common/install_gcc.sh
./docker/common/install_oneapi.sh

# Cleanup package manager
apt-get autoclean && apt-get clean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*