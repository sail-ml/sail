#!/bin/bash

set -ex

add-apt-repository multiverse
apt-get install libmkl-dev libmkl-avx2
