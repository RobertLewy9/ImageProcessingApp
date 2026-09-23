#!/bin/bash

# Navigate to project directory
cd "$(dirname "$0")"

# Create build directory if it doesn't exist
mkdir -p build

# Change to build directory
cd build

# Run CMake to generate build files
cmake ..

# Compile the project
make

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "You can now run the application with ./ImageProcessingApp"
else
    echo "Build failed!"
fi