#!/bin/bash

set -ex

add-apt-repository multiverse
apt-get update
apt-get install libmkl-dev libmkl-avx2
