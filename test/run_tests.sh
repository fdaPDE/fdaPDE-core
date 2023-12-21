#!/bin/sh

# set defaults
SCRIPT_NAME=$(basename "$0")
BUILD_DIR=build/
MEMCHECK=false
COMPILER="gcc"

help()
{
    echo "Usage: .run_tests.sh [options]

       -m --memcheck         use valgrind to checks for memory errors
       -c --compiler         sets compiler (gcc/clang), default gcc
       -h --help             shows this message"
    exit 2
}

clean_build_dir()
{
    if [ -d "$BUILD_DIR" ];
    then
	rm -r build/CMakeCache.txt build/CMakeFiles/
    fi
}

## parse command line inputs
SHORT=m,c:,h
LONG=memcheck,compiler:,help
OPTS=$(getopt -a --n "$SCRIPT_NAME" --options $SHORT --longoptions $LONG -- "$@") 

eval set -- "$OPTS"

while :; do
    case "$1" in
	-m | --memcheck )
	    MEMCHECK=true
	    shift 1
	    ;;
	-c | --compiler )
	    COMPILER="$2"
	    shift 2
	    ;;
	-h | --help )
	    help
	    ;;
	--)
	    shift;
	    break
	    ;;
	*)
	    echo "Unexpected option: $1"
	    help
	    ;;
    esac
done

## set CMake compiler
if [ "$COMPILER" = "gcc" ]; then
    export CC=/usr/bin/gcc
    export CXX=/usr/bin/g++
elif [ "$COMPILER" = "clang" ]; then
    export CC=/usr/bin/clang
    export CXX=/usr/bin/clang++
fi

# cd into build directory
if [ -d "$BUILD_DIR" ];
then
    clean_build_dir
    cd build/
else
    mkdir build/
    cd build/
fi

cmake -Wno-dev ../CMakeLists.txt
make

if [ "$MEMCHECK" = true ]; then
    valgrind --leak-check=full --track-origins=yes ./fdapde_test
else
    ./fdapde_test
fi

rm fdapde_test
