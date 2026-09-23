#include "image_processing/filters/CannyEdgeDetector.h"
#include "image_processing/filters/GrayscaleFilter.h"

cv::Mat CannyEdgeDetector::apply(const cv::Mat& inputImage) {
    // Convert to grayscale first if needed
    cv::Mat grayImage = inputImage;
    if (inputImage.channels() > 1) {
        grayImage = GrayscaleFilter::apply(inputImage);
    }
    
    // Apply Canny edge detection
    cv::Mat edges;
    cv::Canny(grayImage, edges, 100, 200);
    
    // Convert back to 3-channel image for consistent output
    cv::Mat outputImage;
    cv::cvtColor(edges, outputImage, cv::COLOR_GRAY2BGR);
    
    return outputImage;
}