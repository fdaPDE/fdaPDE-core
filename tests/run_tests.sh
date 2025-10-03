#!/bin/sh

set -e  # exit on error

SCRIPT_NAME=$(basename "$0")
BUILD_DIR="build"
MEMCHECK=false
COMPILER="gcc"

help() {
    echo "Usage: $SCRIPT_NAME [options]

Options:
  -m, --memcheck        Run tests under valgrind
  -c, --compiler <cc>   Choose compiler: gcc (default) or clang
  -h, --help            Show this help message"
    exit 2
}

# Parse command line inputs
SHORT="m,c:,h"
LONG="memcheck,compiler:,help"
OPTS=$(getopt -a --name "$SCRIPT_NAME" --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"

while :; do
    case "$1" in
        -m|--memcheck)
            MEMCHECK=true
            shift
            ;;
        -c|--compiler)
            COMPILER="$2"
            shift 2
            ;;
        -h|--help)
            help
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Unexpected option: $1"
            help
            ;;
    esac
done

# Set compiler environment
if [ "$COMPILER" = "gcc" ]; then
    export CC=$(command -v gcc)
    export CXX=$(command -v g++)
    BUILD_DIR="build_gcc"
elif [ "$COMPILER" = "clang" ]; then
    export CC=$(command -v clang)
    export CXX=$(command -v clang++)
    BUILD_DIR="build_clang"
else
    echo "Unsupported compiler: $COMPILER"
    exit 1
fi

echo "==> Using compiler: $COMPILER"
echo "==> Build directory: $BUILD_DIR"

# Clean and create build dir
rm -rf "$BUILD_DIR"

# Configure CMake out-of-source
cmake -S . -B "$BUILD_DIR" \
      -DCMAKE_C_COMPILER="$CC" \
      -DCMAKE_CXX_COMPILER="$CXX" \
      -Wno-dev

# Build
cmake --build "$BUILD_DIR" --parallel

# Run tests
if [ "$MEMCHECK" = true ]; then
    echo "==> Running under valgrind..."
    valgrind --leak-check=full --track-origins=yes "$BUILD_DIR/fdapde_test"
else
    "$BUILD_DIR/fdapde_test"
fi
