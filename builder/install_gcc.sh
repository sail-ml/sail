#!/bin/bash

set -ex

GCC_VERSION=8
  # Need the official toolchain repo to get alternate packages
  add-apt-repository ppa:ubuntu-toolchain-r/test
  apt-get update
    apt-get install -y g++-$GCC_VERSION
  fi

  update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-"$GCC_VERSION" 50
  update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-"$GCC_VERSION" 50
  update-alternatives --install /usr/bin/gcov gcov /usr/bin/gcov-"$GCC_VERSION" 50

  # Cleanup package manager
  apt-get autoclean && apt-get clean
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

