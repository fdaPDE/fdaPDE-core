#!/bin/sh

SEED=${1:-42}
BUILD_DIR=data/build/

clean_build_dir()
{
    if [ -d "$BUILD_DIR" ] && [ -f "$BUILD_DIR/CMakeCache.txt" ] && [ -d "$BUILD_DIR/CMakeFiles" ];
    then
	rm -r data/build/CMakeCache.txt build/CMakeFiles/
    fi
}

if [ -d "$BUILD_DIR" ];
then
    clean_build_dir
    cd data/build/
else
    mkdir data/build/
    cd data/build/
fi

cmake -Wno-dev ../CMakeLists.txt
make

./MatrixGenerator $SEED