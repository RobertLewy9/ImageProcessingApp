# Create test executable
add_executable(TestImageProcessor TestImageProcessor.cpp)

# Include OpenCV include dirs if available (avoid hardcoding paths)
if (DEFINED OpenCV_INCLUDE_DIRS)
	target_include_directories(TestImageProcessor PRIVATE ${OpenCV_INCLUDE_DIRS})
endif()

# Project include directory
target_include_directories(TestImageProcessor PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Link libraries (use PRIVATE)
target_link_libraries(TestImageProcessor PRIVATE ${OpenCV_LIBS})