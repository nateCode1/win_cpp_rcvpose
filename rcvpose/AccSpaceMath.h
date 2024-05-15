#pragma once

#include "utils.hpp"
#include "options.hpp"
#include "FastFor.cu"
#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <open3d/io/PointCloudIO.h>
#include <open3d/Open3D.h>
#include <open3d/utility/FileSystem.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <omp.h> 
#include <atomic>
#include <mutex>
#include <iomanip>
#include <open3d/geometry/PointCloud.h>

Eigen::MatrixXd vectorToEigenMatrix(const std::vector<double>& vec);

Eigen::MatrixXd vectorOfVectorToEigenMatrix(const std::vector<std::vector<double>>& vec);

Eigen::MatrixXd transformKeyPoint(const Eigen::MatrixXd& keypoint, const Eigen::MatrixXd& RTGT, const bool& debug);

void project(const Eigen::MatrixXd& xyz, const Eigen::MatrixXd& K, const Eigen::MatrixXd& RT, Eigen::MatrixXd& xy, Eigen::MatrixXd& actual_xyz);

std::vector<Vertex> rgbd_to_point_cloud(const std::array<std::array<double, 3>, 3>& K, const cv::Mat& depth);

Eigen::MatrixXd rgbd_to_point_cloud(const std::array<std::array<double, 3>, 3>& K, const cv::Mat& depth, const cv::Mat& sem, const cv::Mat& radii);

Eigen::MatrixXd rgbd_to_point_cloud(const cv::Mat& depth, const cv::Mat& radii);
std::vector<Vertex> perspectiveDepthImageToPointCloud2(const cv::Mat& image_depth);
std::vector<Vertex> perspectiveDepthImageToPointCloud(const cv::Mat& depthImg, const std::string& jsonPath = "parameter.json");

void divideByLargest(cv::Mat& matrix, const bool& debug);

void normalizeMat(cv::Mat& input, const bool& debug);

Eigen::Vector3d Accumulator_3D(const std::vector<Vertex>& xyz, const std::vector<double>& radial_list, const bool& debug);
Eigen::Vector3d Accumulator_3D(const Eigen::MatrixXd& xyz_r, const bool& debug);


