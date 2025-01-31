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
    if [ -d "$BUILD_DIR" ] && [ -f "$BUILD_DIR/CMakeCache.txt" ] && [ -d "$BUILD_DIR/CMakeFiles" ];
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
    export CC=$(which gcc)
    export CXX=$(which g++)
elif [ "$COMPILER" = "clang" ]; then
    export CC=$(which clang)
    export CXX=$(which clang++)
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
    valgrind --leak-check=full --track-origins=yes mpirun --use-hwthread-cpus ./mumps_fdapde_test 2>&1 | tee ../test_output.log

else
    mpirun --use-hwthread-cpus ./mumps_fdapde_test 2>&1 | tee ../test_output.log

fi
rm mumps_fdapde_test
