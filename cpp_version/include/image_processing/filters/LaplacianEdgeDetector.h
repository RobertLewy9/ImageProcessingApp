#ifndef LAPLACIANEDGEDETECTOR_H
#define LAPLACIANEDGEDETECTOR_H

#include <opencv2/opencv.hpp>

class LaplacianEdgeDetector {
public:
    static cv::Mat apply(const cv::Mat& inputImage);
};

#endif // LAPLACIANEDGEDETECTOR_H