#ifndef CANNYEDGEDETECTOR_H
#define CANNYEDGEDETECTOR_H

#include <opencv2/opencv.hpp>

class CannyEdgeDetector {
public:
    static cv::Mat apply(const cv::Mat& inputImage);
};

#endif // CANNYEDGEDETECTOR_H