#!/bin/sh

BUILD_DIR=build/

if [ -d $BUILD_DIR ]; then
    rm -rf $BUILD_DIR
fi

mkdir $BUILD_DIR
cd $BUILD_DIR

g++ --version

g++ -std=c++20 -O -fopenmp -c ../src/gen_data_mesh.cpp -o gen_data_mesh.o \
    -I$(spack location -i eigen)/include/eigen3 \
    -I$(spack location -i mumps)/include \
    -I$(spack location -i openmpi)/include

gfortran -o gen_data_mesh -O -fopenmp gen_data_mesh.o \
    -L$(spack location -i mumps)/lib -ldmumps -lmumps_common -lpord \
    -L$(spack location -i gcc)/lib64 -lstdc++ \
    -L$(spack location -i openmpi)/lib -lmpi \
    -L$(spack location -i netlib-scalapack)/lib -lscalapack \
    -L$(spack location -i netlib-lapack)/lib64 -llapack -lblas \
    -L$(spack location -i scotch)/lib -lptscotch -lptscotcherr \
    -L$(spack location -i metis)/lib -lmetis \
    -L$(spack location -i parmetis)/lib -lparmetis \
    -lpthread
