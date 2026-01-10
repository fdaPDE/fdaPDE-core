#!/bin/sh

## some formatting tools
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
NC='\033[0m' # no color

section() {
    echo -e "${BLUE}==>${NC} $1"
}
success() {
    echo -e "${GREEN}✔${NC} $1"
}
fail() {
    echo -e "${RED}✘${NC} $1"
}

set -e  # exit on error

SCRIPT_NAME=$(basename "$0")
BUILD_DIR="build"
MEMCHECK=false
COMPILER="gcc"

help() {
    echo "Usage: $SCRIPT_NAME [options]

Options:
  -m         Run tests under valgrind
  -c <cc>    Choose compiler: gcc (default) or clang
  -h         Show this help message"
    exit 2
}

# parse command line inputs
SHORT="mc:h"
while getopts $SHORT opt; do
    case "$opt" in
        m) MEMCHECK=true ;;
        c) COMPILER="$OPTARG" ;;
        h) help ;;
        *) help ;;
    esac
done

shift $((OPTIND - 1))

# set compiler environment depending on request
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

# detect compiler versions
COMPILER_VERSION=$("$CC" --version | head -n1)
CMAKE_VERSION=$(cmake --version | head -n1)

# detect hardware
CPU_MODEL=$(lscpu | grep -m1 "Model name:" | cut -d: -f2- | sed 's/^ *//')
NTHREADS=$(nproc)
THREADS_PER_CORE=$(lscpu | grep -m1 "Thread(s) per core:" | cut -d: -f2 | tr -d ' ')

echo "=============================================="
echo "   fdaPDE testing framework"
echo "----------------------------------------------"
echo "  CPU model     : $CPU_MODEL"
echo "  Threads       : $NTHREADS (SMT : $THREADS_PER_CORE)"
echo "  Compiler      : $COMPILER_VERSION"
echo "  C++ Compiler  : $CXX"
echo "  CMake version : $CMAKE_VERSION"
echo "  Build dir     : $BUILD_DIR"
echo "=============================================="

# Configure CMake out-of-source
section "Configuring CMake"
if cmake -S . -B "$BUILD_DIR" \
        -DCMAKE_C_COMPILER="$CC" \
        -DCMAKE_CXX_COMPILER="$CXX" \
        -DCMAKE_C_FLAGS="-fdiagnostics-color=always" \
        -DCMAKE_CXX_FLAGS="-fdiagnostics-color=always" \
        -Wno-dev > /dev/null 2>cmake_configure.err; then
    success "CMake configuration complete"
    rm -f cmake_configure.err   # clean up if all good
else
    fail "CMake configuration failed"
    cat cmake_configure.err
    exit 1
fi

# Build
section "Building tests"
if cmake --build "$BUILD_DIR" --parallel >build.out 2>&1; then
    success "Build finished"
    rm -f build.out
else
    fail "Build failed"
    cat build.out
    rm -f build.out
    exit 1
fi

# Run tests
section "Running tests"
if [ "$MEMCHECK" = true ]; then
    echo -e "${YELLOW}Running under valgrind...${NC}"
    valgrind --leak-check=full --track-origins=yes "$BUILD_DIR/fdapde_test"
else
    "$BUILD_DIR/fdapde_test"
fi
