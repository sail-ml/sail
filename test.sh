#!/bin/bash

# if ($1 == "valgrind") {
#     valgrind --leak-check=full ./test
# } else {
#     ./test
# }

if [ $1 = "valgrind" ]; then
    cd build/temp.linux-x86_64-3.7/sail/csrc
     valgrind --track-origins=yes --keep-stacktraces=alloc-and-free --leak-check=full --show-leak-kinds=all ./test
elif [ $1 = "gdb" ]; then
    cd build/temp.linux-x86_64-3.7/sail/csrc
    gdb ./test
elif [ $1 = "python" ]; then
    python -m pytest --cov=sail
else
    cd build/temp.linux-x86_64-3.7/sail/csrc
    ./test
fi

# ./build/temp.linux-x86_64-3.7/sail/csrc/test
# ctest -V