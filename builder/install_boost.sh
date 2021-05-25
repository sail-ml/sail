#!/bin/bash

set -ex

add-apt-repository universe
apt-get update
apt-get install libboost-all-dev
