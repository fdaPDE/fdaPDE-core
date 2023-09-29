#!/bin/sh

# set defaults
valgrind_check=false

# parse command line inputs
while getopts "c" flag; do
    case "${flag}" in
	c) valgrind_check=true;;
    esac
done

# cd into build directory
BUILD_DIR=build/
if [ -d "$BUILD_DIR" ];
then
    cd build/
else
    mkdir build/
    cd build/
fi

cmake -Wno-dev ../CMakeLists.txt
make

if [ "$valgrind_check" = true ]; then
    valgrind --leak-check=full --track-origins=yes ./fdapde_test
else
    ./fdapde_test
fi

rm fdapde_test
