#!/bin/bash
cd build/temp.linux-x86_64-3.7/sail/csrc

# if ($1 == "valgrind") {
#     valgrind --leak-check=full ./test
# } else {
#     ./test
# }

if [ $1 = "valgrind" ]; then
     valgrind --track-origins=yes --keep-stacktraces=alloc-and-free --leak-check=full ./test
elif [ $1 = "gdb" ]; then
    gdb ./test
else
    ./test
fi

# ./build/temp.linux-x86_64-3.7/sail/csrc/test
# ctest -V