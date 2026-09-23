#include "image_processing/ImageProcessor.h"
#include "image_processing/filters/GrayscaleFilter.h"
#include "image_processing/filters/CannyEdgeDetector.h"
#include "image_processing/filters/SobelEdgeDetector.h"
#include "image_processing/filters/LaplacianEdgeDetector.h"
#include "image_processing/filters/GaussianBlurFilter.h"
#include "image_processing/filters/SharpenFilter.h"

cv::Mat ImageProcessor::processImage(const cv::Mat& inputImage, Operation operation) {
    switch (operation) {
        case Operation::GRAYSCALE:
            return GrayscaleFilter::apply(inputImage);
        case Operation::CANNY:
            return CannyEdgeDetector::apply(inputImage);
        case Operation::SOBEL:
            return SobelEdgeDetector::apply(inputImage);
        case Operation::LAPLACIAN:
            return LaplacianEdgeDetector::apply(inputImage);
        case Operation::GAUSSIAN_BLUR:
            return GaussianBlurFilter::apply(inputImage);
        case Operation::SHARPEN:
            return SharpenFilter::apply(inputImage);
        default:
            return inputImage;
    }
}