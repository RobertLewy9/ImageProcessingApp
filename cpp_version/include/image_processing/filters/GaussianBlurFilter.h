#ifndef GAUSSIANBLURFILTER_H
#define GAUSSIANBLURFILTER_H

#include <opencv2/opencv.hpp>

class GaussianBlurFilter {
public:
    static cv::Mat apply(const cv::Mat& inputImage);
};

#endif // GAUSSIANBLURFILTER_H