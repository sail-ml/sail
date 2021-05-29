#!/bin/bash

set -ex

chmod +x ./.github/workflow_tools/install_avx2.sh
chmod +x ./.github/workflow_tools/install_boost.sh
chmod +x ./.github/workflow_tools/install_cmake.sh
chmod +x ./.github/workflow_tools/install_gcc.sh

# ./builder/install_avx2.sh
./.github/workflow_tools/install_boost.sh
./.github/workflow_tools/install_cmake.sh
./.github/workflow_tools/install_gcc.sh