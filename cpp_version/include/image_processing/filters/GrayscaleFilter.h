#ifndef GRAYSCALEFILTER_H
#define GRAYSCALEFILTER_H

#include <opencv2/opencv.hpp>

class GrayscaleFilter {
public:
    static cv::Mat apply(const cv::Mat& inputImage);
};

#endif // GRAYSCALEFILTER_H