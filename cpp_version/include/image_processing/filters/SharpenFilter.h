#ifndef SHARPENFILTER_H
#define SHARPENFILTER_H

#include <opencv2/opencv.hpp>

class SharpenFilter {
public:
    static cv::Mat apply(const cv::Mat& inputImage);
};

#endif // SHARPENFILTER_H