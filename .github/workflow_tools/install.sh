#!/bin/bash

set -ex

chmod +x ./.github/workflow_tools/install_tools.sh
chmod +x ./.github/workflow_tools/install_python.sh
chmod +x ./.github/workflow_tools/install_boost.sh
chmod +x ./.github/workflow_tools/install_cmake.sh
chmod +x ./.github/workflow_tools/install_gcc.sh
chmod +x ./.github/workflow_tools/install_oneapi.sh

./.github/workflow_tools/install_tools.sh
./.github/workflow_tools/install_python.sh
./.github/workflow_tools/install_boost.sh
./.github/workflow_tools/install_cmake.sh
./.github/workflow_tools/install_gcc.sh
./.github/workflow_tools/install_oneapi.sh

# Cleanup package manager
apt-get autoclean && apt-get clean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*