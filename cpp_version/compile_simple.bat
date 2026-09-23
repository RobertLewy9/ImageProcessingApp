@echo off
REM Simple compilation script for Image Processing App
REM Assumes OpenCV is installed and accessible

echo Compiling Image Processing Application...

REM Create build directory if it doesn't exist
if not exist "build" mkdir build
cd build

REM Compile the main application
g++ -std=c++17 ^
../src/main.cpp ^
../src/image_processing/ImageProcessor.cpp ^
../src/image_processing/filters/GrayscaleFilter.cpp ^
../src/image_processing/filters/CannyEdgeDetector.cpp ^
../src/image_processing/filters/SobelEdgeDetector.cpp ^
../src/image_processing/filters/LaplacianEdgeDetector.cpp ^
../src/image_processing/filters/GaussianBlurFilter.cpp ^
../src/image_processing/filters/SharpenFilter.cpp ^
../src/ui/MainWindow.cpp ^
-I../include ^
-ID:\OpenCV\build\include ^
-LD:\OpenCV\build\x64\vc15\lib ^
-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_features2d -lopencv_videoio ^
-o ImageProcessingApp.exe

echo Compilation completed!

REM Compile the test program
g++ -std=c++17 ^
../src/tests/TestImageProcessor.cpp ^
../src/image_processing/ImageProcessor.cpp ^
../src/image_processing/filters/GrayscaleFilter.cpp ^
../src/image_processing/filters/CannyEdgeDetector.cpp ^
../src/image_processing/filters/SobelEdgeDetector.cpp ^
../src/image_processing/filters/LaplacianEdgeDetector.cpp ^
../src/image_processing/filters/GaussianBlurFilter.cpp ^
../src/image_processing/filters/SharpenFilter.cpp ^
-I../include ^
-ID:\OpenCV\build\include ^
-LD:\OpenCV\build\x64\vc15\lib ^
-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_features2d -lopencv_videoio ^
-o TestImageProcessor.exe

echo Test program compilation completed!

pause