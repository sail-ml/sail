#!/bin/bash

valgrind --leak-check=full --show-leak-kinds=all --suppressions=valgrind-python.supp --track-origins=yes --verbose --log-file=valgrind-out.txt $1