#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define build directory
BUILD_DIR="build"

# Step 1: Create or clean the build directory
if [ -d "$BUILD_DIR" ]; then
    echo "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi
echo "Creating build directory..."
mkdir "$BUILD_DIR"

# Step 2: Navigate to the build directory
cd "$BUILD_DIR"

# Step 3: Run CMake to configure the build system
echo "Configuring the project with CMake..."
cmake ..  #-DCMAKE_CXX_COMPILER=/usr/local/bin/g++-14

# Step 4: Build the project
echo "Building the project..."
cmake --build .

# Step 5: Run the resulting executable
echo "Running the executable..."
./IsoTest #--gtest_filter=NurbsTest.NurbsSBasis2D
