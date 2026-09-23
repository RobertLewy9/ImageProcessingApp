#include "image_processing/filters/GaussianBlurFilter.h"

cv::Mat GaussianBlurFilter::apply(const cv::Mat& inputImage) {
    cv::Mat blurredImage;
    // Apply Gaussian blur with kernel size 5x5 and sigma values of 0 (automatically calculated)
    cv::GaussianBlur(inputImage, blurredImage, cv::Size(5, 5), 0, 0);
    
    return blurredImage;
}