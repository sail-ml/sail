#!/bin/bash

set -ex

apt-get install -y software-properties-common
add-apt-repository ppa:deadsnakes/ppa   
apt-get update   

apt-get install -y git
apt-get install -y wget
apt-get install -y curl
apt-get install -y clang-tidy


