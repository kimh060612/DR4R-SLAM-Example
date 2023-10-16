#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <eigen3/Eigen/Core>
#include <unistd.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>
#include <chrono>
#include <thread>

using namespace std;
using namespace std::chrono_literals;
using namespace Eigen;

string left_file = "../data/left.png";
string right_file = "../data/right.png";

int main(int argc, char **argv) {
    // Camera Intrinsic matrix K
    double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
    double b = 0.573; // baseline

    cv::Mat left = cv::imread(left_file, 0);
    cv::Mat right = cv::imread(right_file, 0);
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, 96, 9, 8 * 9 * 9, 32 * 9 * 9, 1, 63, 10, 100, 32); 
    cv::Mat disparity_sgbm, disparity;
    sgbm->compute(left, right, disparity_sgbm);
    disparity_sgbm.convertTo(disparity, CV_32F, 1.0 / 16.0f);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

    for (int v = 0; v < left.rows; v++) {
        for (int u = 0; u < left.cols; u++) {
            if (disparity.at<float>(v, u) <= 0.0 || disparity.at<float>(v, u) >= 96.0) continue;

            double x = (u - cx) / fx;
            double y = (v - cy) / fy;
            double depth = fx * b / (disparity.at<float>(v, u));
            pcl::PointXYZRGB pt;
            pt.x = x * depth;
            pt.y = y * depth;
            pt.z = depth;
            pt.r = left.at<uchar>(v, u);
            pt.g = left.at<uchar>(v, u);
            pt.b = left.at<uchar>(v, u);
            cloud->push_back(pt);
        }
    }

    cv::imwrite("./disparity.png", (disparity / 96.0) * 255.); 
    pcl::visualization::PCLVisualizer viewer("Point Cloud");
    viewer.addPointCloud<pcl::PointXYZRGB>(cloud, "point_cloud");
    viewer.setCameraPosition(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0);

    while (!viewer.wasStopped()) {
        viewer.spin();
    }   

    return 0;
}
