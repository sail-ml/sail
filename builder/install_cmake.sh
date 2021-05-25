#!/bin/bash

set -ex

CMAKE_VERSION=3.14.0

# Turn 3.6.3 into v3.6
path=$(echo "${CMAKE_VERSION}" | sed -e 's/\([0-9].[0-9]\+\).*/v\1/')
file="cmake-${CMAKE_VERSION}-Linux-x86_64.tar.gz"

# Download and install specific CMake version in /usr/local
pushd /tmp
curl -Os --retry 3 "https://cmake.org/files/${path}/${file}"
tar -C /usr/local --strip-components 1 --no-same-owner -zxf ${file}
rm -f ${file}
popd