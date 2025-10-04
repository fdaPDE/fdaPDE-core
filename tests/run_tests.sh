#!/bin/sh

## some formatting tools
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
NC='\033[0m' # No Color

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
  -m, --memcheck        Run tests under valgrind
  -c, --compiler <cc>   Choose compiler: gcc (default) or clang
  -h, --help            Show this help message"
    exit 2
}

# Parse command line inputs
SHORT="m,c:,h"
LONG="memcheck,compiler:,help"
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
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

# Detect versions
COMPILER_VERSION=$("$CC" --version | head -n1)
CMAKE_VERSION=$(cmake --version | head -n1)

echo "=============================================="
echo "   fdaPDE testing framework"
echo "----------------------------------------------"
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
