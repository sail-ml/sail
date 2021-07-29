#!/bin/bash

set -ex

GCC_VERSION=10
# Need the official toolchain repo to get alternate packages
add-apt-repository ppa:ubuntu-toolchain-r/test
apt-get update
apt-get install -y g++-$GCC_VERSION

update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-"$GCC_VERSION" 50
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-"$GCC_VERSION" 50
update-alternatives --install /usr/bin/gcov gcov /usr/bin/gcov-"$GCC_VERSION" 50

apt-get install -y ninja-build