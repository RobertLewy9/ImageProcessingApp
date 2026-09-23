#ifndef IMAGEPROCESSOR_H
#define IMAGEPROCESSOR_H

#include <opencv2/opencv.hpp>
#include <string>

class ImageProcessor {
public:
    enum class Operation {
        GRAYSCALE,
        CANNY,
        SOBEL,
        LAPLACIAN,
        GAUSSIAN_BLUR,
        SHARPEN
    };

    static cv::Mat processImage(const cv::Mat& inputImage, Operation operation);
    
private:
    static cv::Mat applyGrayscale(const cv::Mat& inputImage);
    static cv::Mat applyCanny(const cv::Mat& inputImage);
    static cv::Mat applySobel(const cv::Mat& inputImage);
    static cv::Mat applyLaplacian(const cv::Mat& inputImage);
    static cv::Mat applyGaussianBlur(const cv::Mat& inputImage);
    static cv::Mat applySharpen(const cv::Mat& inputImage);
};

#endif // IMAGEPROCESSOR_H