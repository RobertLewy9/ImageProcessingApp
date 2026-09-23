#include "image_processing/filters/SobelEdgeDetector.h"
#include "image_processing/filters/GrayscaleFilter.h"

cv::Mat SobelEdgeDetector::apply(const cv::Mat& inputImage) {
    // Convert to grayscale first if needed
    cv::Mat grayImage = inputImage;
    if (inputImage.channels() > 1) {
        grayImage = GrayscaleFilter::apply(inputImage);
    }
    
    // Apply Sobel edge detection
    cv::Mat gradX, gradY;
    cv::Sobel(grayImage, gradX, CV_16S, 1, 0, 3);
    cv::Sobel(grayImage, gradY, CV_16S, 0, 1, 3);
    
    // Convert back to CV_8U
    cv::convertScaleAbs(gradX, gradX);
    cv::convertScaleAbs(gradY, gradY);
    
    // Combine gradients
    cv::Mat sobelEdges;
    cv::addWeighted(gradX, 0.5, gradY, 0.5, 0, sobelEdges);
    
    // Convert back to 3-channel image for consistent output
    cv::Mat outputImage;
    cv::cvtColor(sobelEdges, outputImage, cv::COLOR_GRAY2BGR);
    
    return outputImage;
}