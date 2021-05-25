#!/bin/bash

if [[ $1 = "WIN32" ]]
then
echo "win"
  export WIN32=1
else
echo "linux"
  export LINUX=1
fi

python setup.py install