#!/bin/sh

BUILD_DIR=build/
PROCESSES=--use-hwthread-cpus

# parse arguments
while [ "$1" != "" ]; do
    case $1 in
    -p | --processes )     
        shift
        PROCESSES="-np $1"
        ;;
    esac
    shift
done

cd $BUILD_DIR
mpirun $PROCESSES ./gen_data_mesh