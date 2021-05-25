#!/bin/bash

set -ex

chmod +x ./builder/install_avx2.sh
chmod +x ./builder/install_boost.sh
chmod +x ./builder/install_cmake.sh
chmod +x ./builder/install_gcc.sh

# ./builder/install_avx2.sh
./builder/install_boost.sh
./builder/install_cmake.sh
./builder/install_gcc.sh