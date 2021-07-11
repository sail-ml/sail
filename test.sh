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
    python python_test/run.py
elif [ $1 = "all" ]; then
    python python_test/run.py
    cd build/t*/sail/csrc
    ctest -V
    cd ../../../../
else
    # cd build/temp.linux-x86_64-3.7/sail/csrc
    # ctest -V
    cd build/temp.linux-x86_64-3.7/sail/csrc
    ./test
fi

if [ $2 = "coverage-xml" ]; then 
    gcovr --filter sail/ --xml coverage.xml -s 2> /dev/null
fi

if [ $2 = "coverage" ]; then 
    gcovr --filter sail/ -s 2> /dev/null
fi
# ./build/temp.linux-x86_64-3.7/sail/csrc/test
# ctest -V