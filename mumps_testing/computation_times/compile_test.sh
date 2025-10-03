#!/bin/sh

BUILD_DIR=build/

if [ -d $BUILD_DIR ]; then
    rm -rf $BUILD_DIR
fi

mkdir $BUILD_DIR
cd $BUILD_DIR

mpicxx -fopenmp -c ../src/gen_data_srpde.cpp -o test_main.o -O2 -std=c++20 -g -march=native \
    -I/home/matteo/Desktop/fdaPDE-cpp \
    -I/home/matteo/Desktop/fdaPDE-cpp/fdaPDE/core \
    -I/home/matteo/Desktop/eigen \
    -I/usr/local/include/Mumps

mpif77 -O -fopenmp -o test_main test_main.o \
    -L/usr/local/lib/Mumps -ldmumps -lmumps_common \
    -L/usr/lib -lparmetis -lmetis \
    -L/home/matteo/Desktop/MUMPS-openMPI-noBLACS/PORD/lib -lpord \
    -L/usr/lib -lptesmumps -lptscotch -lptscotcherr \
    -lscalapack-openmpi -llapack -lblas -lpthread \
    -L/usr/lib/gcc/x86_64-linux-gnu/13/ -lstdc++ \
    -lmpi_cxx
