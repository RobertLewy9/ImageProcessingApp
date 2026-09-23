#include "image_processing/filters/LaplacianEdgeDetector.h"
#include "image_processing/filters/GrayscaleFilter.h"

cv::Mat LaplacianEdgeDetector::apply(const cv::Mat& inputImage) {
    // Convert to grayscale first if needed
    cv::Mat grayImage = inputImage;
    if (inputImage.channels() > 1) {
        grayImage = GrayscaleFilter::apply(inputImage);
    }
    
    // Apply Laplacian edge detection
    cv::Mat laplacianEdges;
    cv::Laplacian(grayImage, laplacianEdges, CV_16S, 3);
    
    // Convert back to CV_8U
    cv::convertScaleAbs(laplacianEdges);
    
    // Convert back to 3-channel image for consistent output
    cv::Mat outputImage;
    cv::cvtColor(laplacianEdges, outputImage, cv::COLOR_GRAY2BGR);
    
    return outputImage;
}