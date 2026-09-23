# Adding New Filters to the Image Processing Application

This document explains how to add new image processing filters to the application.

## Steps to Add a New Filter

### 1. Create Filter Class

Create a new filter class in `include/image_processing/filters/`:

```cpp
// ExampleFilter.h
#ifndef EXAMPLE_FILTER_H
#define EXAMPLE_FILTER_H

#include <opencv2/opencv.hpp>

class ExampleFilter {
public:
    static cv::Mat apply(const cv::Mat& input);
};

#endif // EXAMPLE_FILTER_H
```

Implement the filter in `src/image_processing/filters/`:

```cpp
// ExampleFilter.cpp
#include "image_processing/filters/ExampleFilter.h"

cv::Mat ExampleFilter::apply(const cv::Mat& input) {
    cv::Mat output;
    // Implement your filter logic here
    // Example: cv::filter2D(input, output, -1, kernel);
    return output;
}
```

### 2. Update ImageProcessor

Add a new operation enum in `include/image_processing/ImageProcessor.h`:

```cpp
enum class Operation {
    GRAYSCALE,
    CANNY,
    SOBEL,
    LAPLACIAN,
    GAUSSIAN_BLUR,
    SHARPEN,
    EXAMPLE_FILTER  // Add your new filter here
};
```

Update the `processImage` method in `src/image_processing/ImageProcessor.cpp`:

```cpp
case Operation::EXAMPLE_FILTER:
    return ExampleFilter::apply(input);
```

### 3. Update UI (Optional)

If you want to add the filter to the GUI, update the following files:

1. In `include/ui/MainWindow.h`, add the new filter to the `FilterType` enum if needed.
2. In `src/ui/MainWindow.cpp`, add the new filter to the dropdown menu and implement the processing logic.

### 4. Update CMakeLists.txt

Make sure your new source files are included in the build by updating the `CMakeLists.txt` file if necessary.

### 5. Test Your Filter

Create a test for your new filter in `src/tests/TestImageProcessor.cpp`:

```cpp
// Test example filter
cv::Mat exampleImage = ImageProcessor::processImage(testImage, ImageProcessor::Operation::EXAMPLE_FILTER);
if (!exampleImage.empty()) {
    std::cout << "✓ Example filter test passed\n";
} else {
    std::cout << "✗ Example filter test failed\n";
}
```

## Best Practices

1. **Input Validation**: Always check if the input image is valid before processing.
2. **Error Handling**: Handle potential errors gracefully and return meaningful results.
3. **Performance**: Optimize your filter for performance, especially for large images.
4. **Documentation**: Document your filter's purpose, parameters, and usage.
5. **Testing**: Write tests to verify that your filter works correctly.