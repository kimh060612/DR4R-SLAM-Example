cmake_minimum_required(VERSION 3.10)
project(gaussNewtonOpt)

set(CMAKE_CXX_STANDARD 14)
include_directories("/usr/include/eigen3")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(gaussNewtonOpt main.cpp)
target_link_libraries(gaussNewtonOpt ${OpenCV_LIBS})