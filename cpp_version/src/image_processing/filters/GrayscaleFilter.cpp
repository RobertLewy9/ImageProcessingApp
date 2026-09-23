#include "image_processing/filters/GrayscaleFilter.h"

cv::Mat GrayscaleFilter::apply(const cv::Mat& inputImage) {
    cv::Mat grayImage;
    
    if (inputImage.channels() == 3) {
        cv::cvtColor(inputImage, grayImage, cv::COLOR_BGR2GRAY);
    } else if (inputImage.channels() == 1) {
        grayImage = inputImage.clone();
    } else {
        // For other channel counts, convert to BGR first then to grayscale
        cv::Mat bgrImage;
        cv::cvtColor(inputImage, bgrImage, cv::COLOR_BGRA2BGR);
        cv::cvtColor(bgrImage, grayImage, cv::COLOR_BGR2GRAY);
    }
    
    return grayImage;
}