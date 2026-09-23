#ifndef SOBELEDGEDETECTOR_H
#define SOBELEDGEDETECTOR_H

#include <opencv2/opencv.hpp>

class SobelEdgeDetector {
public:
    static cv::Mat apply(const cv::Mat& inputImage);
};

#endif // SOBELEDGEDETECTOR_H