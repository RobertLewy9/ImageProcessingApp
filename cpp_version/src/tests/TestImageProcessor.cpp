#include <iostream>
#include <opencv2/opencv.hpp>
#include "image_processing/ImageProcessor.h"

int main() {
    // Create a simple test image (white rectangle on black background)
    cv::Mat testImage(200, 200, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::rectangle(testImage, cv::Point(50, 50), cv::Point(150, 150), cv::Scalar(255, 255, 255), -1);
    
    std::cout << "Testing ImageProcessor functionalities...\n";
    
    // Test grayscale conversion
    cv::Mat grayImage = ImageProcessor::processImage(testImage, ImageProcessor::Operation::GRAYSCALE);
    if (!grayImage.empty() && grayImage.channels() == 1) {
        std::cout << "✓ Grayscale conversion test passed\n";
    } else {
        std::cout << "✗ Grayscale conversion test failed\n";
    }
    
    // Test Canny edge detection
    cv::Mat cannyImage = ImageProcessor::processImage(testImage, ImageProcessor::Operation::CANNY);
    if (!cannyImage.empty() && cannyImage.channels() == 3) {
        std::cout << "✓ Canny edge detection test passed\n";
    } else {
        std::cout << "✗ Canny edge detection test failed\n";
    }
    
    // Test Sobel edge detection
    cv::Mat sobelImage = ImageProcessor::processImage(testImage, ImageProcessor::Operation::SOBEL);
    if (!sobelImage.empty() && sobelImage.channels() == 3) {
        std::cout << "✓ Sobel edge detection test passed\n";
    } else {
        std::cout << "✗ Sobel edge detection test failed\n";
    }
    
    // Test Laplacian edge detection
    cv::Mat laplacianImage = ImageProcessor::processImage(testImage, ImageProcessor::Operation::LAPLACIAN);
    if (!laplacianImage.empty() && laplacianImage.channels() == 3) {
        std::cout << "✓ Laplacian edge detection test passed\n";
    } else {
        std::cout << "✗ Laplacian edge detection test failed\n";
    }
    
    // Test Gaussian blur
    cv::Mat gaussianImage = ImageProcessor::processImage(testImage, ImageProcessor::Operation::GAUSSIAN_BLUR);
    if (!gaussianImage.empty() && gaussianImage.channels() == 3) {
        std::cout << "✓ Gaussian blur test passed\n";
    } else {
        std::cout << "✗ Gaussian blur test failed\n";
    }
    
    // Test sharpen
    cv::Mat sharpenImage = ImageProcessor::processImage(testImage, ImageProcessor::Operation::SHARPEN);
    if (!sharpenImage.empty() && sharpenImage.channels() == 3) {
        std::cout << "✓ Sharpen test passed\n";
    } else {
        std::cout << "✗ Sharpen test failed\n";
    }
    
    std::cout << "All tests completed!\n";
    
    return 0;
}