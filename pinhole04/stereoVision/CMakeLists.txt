cmake_minimum_required(VERSION 3.10)
project(stereoVision)

set(CMAKE_CXX_STANDARD 14)
include_directories("/usr/include/eigen3")
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(stereoVision main.cpp)
target_link_libraries(stereoVision ${OpenCV_LIBS} ${PCL_LIBRARIES})