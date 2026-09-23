cmake_minimum_required(VERSION 3.16)
project(ImageProcessingApp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set OpenCV directory
set(OpenCV_DIR "D:/OpenCV/build")

# Find required packages
find_package(OpenCV REQUIRED)

# Prefer Qt6, fall back to Qt5 if Qt6 is not available
find_package(Qt6 COMPONENTS Core Widgets QUIET)
if (Qt6_FOUND)
    set(QT_LIBS Qt6::Core Qt6::Widgets)
else()
    find_package(Qt5 COMPONENTS Core Widgets REQUIRED)
    set(QT_LIBS Qt5::Core Qt5::Widgets)
endif()

# Enable automatic MOC, RCC and UIC processing
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Include directories
include_directories(include)

# Use the include dirs provided by OpenCV (avoid hardcoding paths)
if (DEFINED OpenCV_INCLUDE_DIRS)
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()

# Source files
file(GLOB_RECURSE SOURCES 
    "src/*.cpp"
    "src/image_processing/*.cpp"
    "src/image_processing/filters/*.cpp"
    "src/ui/*.cpp"
)

# Exclude main.cpp from SOURCES to avoid conflicts with tests
list(FILTER SOURCES EXCLUDE REGEX "src/main.cpp")

# Header files
file(GLOB_RECURSE HEADERS 
    "include/*.h"
    "include/image_processing/*.h"
    "include/image_processing/filters/*.h"
    "include/ui/*.h"
)

# Resource files
set(RESOURCES resources/resources.qrc)

# Create main executable
add_executable(${PROJECT_NAME} src/main.cpp ${SOURCES} ${HEADERS} ${RESOURCES})

# Link libraries for main executable
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} ${QT_LIBS})

# Add tests subdirectory
enable_testing()
add_subdirectory(src/tests)