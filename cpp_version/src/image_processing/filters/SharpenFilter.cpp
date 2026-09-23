#include "image_processing/filters/SharpenFilter.h"

cv::Mat SharpenFilter::apply(const cv::Mat& inputImage) {
    // Define sharpen kernel
    cv::Mat kernel = (cv::Mat_<float>(3, 3) << 
        0, -1, 0,
        -1, 5, -1,
        0, -1, 0);
    
    cv::Mat sharpenedImage;
    // Apply sharpen filter using filter2D
    cv::filter2D(inputImage, sharpenedImage, -1, kernel);
    
    return sharpenedImage;
}