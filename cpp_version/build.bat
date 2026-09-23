@echo off
REM Build script for Image Processing App

echo Building Image Processing Application...

REM Create build directory if it doesn't exist
if not exist "build" mkdir build
cd build

REM Generate build files with CMake
B:\cmake-4.2.0\bin\cmake .. -G "MinGW Makefiles"

REM Build the project
mingw32-make

echo Build completed!

pause