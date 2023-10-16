cmake_minimum_required(VERSION 3.10)
project(ceresOpt)

set(CMAKE_CXX_STANDARD 14)
include_directories("/usr/include/eigen3")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

add_executable(ceresOpt main.cpp)
target_link_libraries(ceresOpt ${OpenCV_LIBS} ${CERES_LIBRARIES})