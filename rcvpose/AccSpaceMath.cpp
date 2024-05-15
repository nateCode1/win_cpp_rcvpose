#include "AccSpaceMath.h"
#include <nlohmann/json.hpp>

using namespace std;
using namespace open3d;
using namespace Eigen;
using json = nlohmann::json;

typedef shared_ptr<geometry::PointCloud> pc_ptr;
typedef geometry::PointCloud pc;



Eigen::MatrixXd vectorToEigenMatrix(const vector<double>& vec) {
    int size = vec.size();
    Eigen::MatrixXd matrix(1, size);


    for (int i = 0; i < size; i++) {
        matrix(0, i) = vec[i];
    }

    return matrix;
}

Eigen::MatrixXd vectorOfVectorToEigenMatrix(const vector<vector<double>>& vec) {
    int rows = vec.size();
    int cols = vec[0].size();
    Eigen::MatrixXd matrix(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix(i, j) = vec[i][j];
        }
    }

    return matrix;
}

Eigen::MatrixXd transformKeyPoint(const Eigen::MatrixXd& keypoint, const Eigen::MatrixXd& RTGT, const bool& debug) {

    int rows = keypoint.rows();
    int cols = RTGT.cols() - 1;

    if (debug) {
        cout << "Keypoint shape: " << keypoint.rows() << " " << keypoint.cols() << endl;
        cout << "RTGT shape: " << RTGT.rows() << " " << RTGT.cols() << endl;
        cout << "cols: " << cols << endl;
        cout << "rows: " << rows << endl;
    }

    Eigen::MatrixXd keypointTransformed = keypoint * RTGT.block(0, 0, cols, 3).transpose();
    keypointTransformed += RTGT.block(0, 3, cols, 1).transpose();

    return keypointTransformed * 1000.0;
}


void project(const MatrixXd& xyz, const MatrixXd& K, const MatrixXd& RT, MatrixXd& xy, MatrixXd& actual_xyz)
{
    /*
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    */

    actual_xyz = xyz * (RT.block(0, 0, RT.rows(), 3).transpose());
    MatrixXd RTslice = RT.block(0, 3, RT.rows(), 1).transpose();

    for (int i = 0; i < actual_xyz.rows(); i++) {
        actual_xyz.row(i) += RTslice;
    }

    MatrixXd xy_temp = actual_xyz * K.transpose();

    xy = xy_temp.leftCols(2).array() / xy_temp.col(2).array().replicate(1, 2);
}


vector<Vertex> rgbd_to_point_cloud(const array<array<double, 3>, 3>& K, const cv::Mat& depth) {
    cv::Mat depth64F;
    depth.convertTo(depth64F, CV_64F);  // Convert depth image to CV_64F

    vector<cv::Point> nonzeroPoints;
    cv::findNonZero(depth64F, nonzeroPoints);

    vector<double> zs(nonzeroPoints.size());
    vector<double> xs(nonzeroPoints.size());
    vector<double> ys(nonzeroPoints.size());


    for (int i = 0; i < nonzeroPoints.size(); i++) {
        int v, u;
        v = nonzeroPoints[i].y;
        u = nonzeroPoints[i].x;
        zs[i] = depth64F.at<double>(v, u);
    }


    for (int i = 0; i < nonzeroPoints.size(); i++) {
        int v, u;
        v = nonzeroPoints[i].y;
        u = nonzeroPoints[i].x;
        xs[i] = ((u - K[0][2]) * zs[i]) / K[0][0];
        ys[i] = ((v - K[1][2]) * zs[i]) / K[1][1];
    }

    vector<Vertex> pointCloud(nonzeroPoints.size());

    for (int i = 0; i < xs.size(); i++) {
        pointCloud[i].x = xs[i];
        pointCloud[i].y = ys[i];
        pointCloud[i].z = zs[i];
        
    }

    return pointCloud;
}

Eigen::MatrixXd rgbd_to_point_cloud(const array<array<double, 3>, 3>& K, const cv::Mat& depth, const cv::Mat& sem, const cv::Mat& radii) {
    cv::Mat depth64F;
    depth.convertTo(depth64F, CV_64F);  // Convert depth image to CV_64F
    
    std::cout<<"asdf"<<std::endl;
    vector<cv::Point> nonzeroPoints;
    cv::findNonZero(sem, nonzeroPoints);

    vector<double> zs(nonzeroPoints.size());
    vector<double> xs(nonzeroPoints.size());
    vector<double> ys(nonzeroPoints.size());
    vector<double> rs(nonzeroPoints.size());


    for (int i = 0; i < nonzeroPoints.size(); i++) {
        int v, u;
        v = nonzeroPoints[i].y;
        u = nonzeroPoints[i].x;
        zs[i] = depth64F.at<double>(v, u);
        rs[i] = radii.at<double>(v, u);
    }


    for (int i = 0; i < nonzeroPoints.size(); i++) {
        int v, u;
        v = nonzeroPoints[i].y;
        u = nonzeroPoints[i].x;
        xs[i] = ((u - K[0][2]) * zs[i]) / K[0][0];
        ys[i] = ((v - K[1][2]) * zs[i]) / K[1][1];
    }

    Eigen::MatrixXd pointCloud_r(nonzeroPoints.size(), 4);

    for (int i = 0; i < xs.size(); i++) {
        pointCloud_r(i, 0) = xs[i];
        pointCloud_r(i, 1) = ys[i];
        pointCloud_r(i, 2) = zs[i];
        pointCloud_r(i, 3) = rs[i];
      
    }

    return pointCloud_r;
}

inline double deg2rad(double degrees) {
    return degrees * M_PI / 180.0;
}

Eigen::MatrixXd rgbd_to_point_cloud(const cv::Mat& cv_image_depth, const cv::Mat& radii)
{
    double perspectiveAngle = 53;
    double clip_start = 0.20;
    double clip_end = 0.40;
    int resolutionX = 640;
    int resolutionY = 640;
    int resolution_big = 640;
    int pixelOffset_X_KoSyTopLeft = 0;
    int pixelOffset_Y_KoSyTopLeft = 0;

    cv::Mat cv_image_big = cv::Mat::zeros(resolution_big, resolution_big, cv_image_depth.type());
    cv::Mat targetROI = cv_image_big(cv::Rect(pixelOffset_X_KoSyTopLeft, pixelOffset_Y_KoSyTopLeft, resolutionX, resolutionY));
    cv_image_depth.copyTo(targetROI);

    double range_ = clip_end - clip_start;
    double defaultValue = 0.0;

    // Pre-compute orthographic size multipliers to avoid redundancy in loop
    double tan_half_perspectiveAngle = std::tan(deg2rad(perspectiveAngle / 2));

    // Prepare to count non-zero elements
    int nonZeroCount = cv::countNonZero(cv_image_big);
    std::cout<<nonZeroCount<<std::endl;
    Eigen::MatrixXd point_cloud_matrix(nonZeroCount, 4);
    int row = 0;

    for (int j = 0; j < cv_image_big.rows; ++j) {
        for (int i = 0; i < cv_image_big.cols; ++i) {
            double depthValue = cv_image_big.at<float>(j, i); // Assuming cv_image_depth is CV_32F
            if (depthValue == defaultValue) continue;
            
            double world_z = (depthValue * range_) + clip_start;
            //std::cout<<world_z<<std::endl;
            //if(world_z<0.39)
            	//std::cout<<world_z<<std::endl;
            double orthoSizeZ_x = tan_half_perspectiveAngle * world_z * 2 * resolutionX / resolution_big;
            double orthoSizeZ_y = tan_half_perspectiveAngle * world_z * 2 * resolutionY / resolution_big;

            double meterPerPixel_x = orthoSizeZ_x / resolutionX;
            double meterPerPixel_y = orthoSizeZ_y / resolutionY;

            double world_x = (i + 0.5 - resolution_big / 2) * meterPerPixel_x;
            double world_y = (j + 0.5 - resolution_big / 2) * meterPerPixel_y;

            point_cloud_matrix(row, 0) = world_x;
            point_cloud_matrix(row, 1) = world_y;
            point_cloud_matrix(row, 2) = world_z;
            point_cloud_matrix(row, 3) = radii.at<float>(j, i);
            //std::cout<<radii.at<float>(j, i)<<"  ,  ";
            ++row;
        }
    }
	//std::cout<<"$$$$$$$$$$$$$$$$$$$"<<std::endl;
    return point_cloud_matrix;
}

std::vector<Vertex> perspectiveDepthImageToPointCloud2(const cv::Mat& image_depth) {
    double perspectiveAngle = 53;
    double clip_start = 0.20;
    double clip_end = 0.40;
    int resolutionX = 640;
    int resolutionY = 640;
    int resolution_big = 640;
    int pixelOffset_X_KoSyTopLeft = 0;
    int pixelOffset_Y_KoSyTopLeft = 0;
    
    assert(image_depth.rows == resolutionY && image_depth.cols == resolutionX);

    cv::Mat image_big = cv::Mat::zeros(resolution_big, resolution_big, image_depth.type());
    cv::Rect roi(pixelOffset_X_KoSyTopLeft, pixelOffset_Y_KoSyTopLeft, resolutionX, resolutionY);
    image_depth(roi).copyTo(image_big(roi));

    std::vector<Vertex> point_cloud;
    double range_ = clip_end - clip_start;
    double defaultValue = 0.0;

    for (int j = 0; j < image_big.rows; ++j) {
        for (int i = 0; i < image_big.cols; ++i) {
            double depth_value = image_big.at<float>(j, i); // Assuming image_depth is CV_32F
            
            if (depth_value == defaultValue) continue;
            double world_z = (depth_value * range_) + clip_start;
            if(world_z<0.39)
            	std::cout<<world_z<<std::endl;
            double orthoSizeZ_x = std::tan(deg2rad(perspectiveAngle / 2)) * world_z * 2 * resolutionX / resolution_big;
            double orthoSizeZ_y = std::tan(deg2rad(perspectiveAngle / 2)) * world_z * 2 * resolutionY / resolution_big;

            double meterPerPixel_x = orthoSizeZ_x / resolutionX;
            double meterPerPixel_y = orthoSizeZ_y / resolutionY;

            double world_x = (i + 0.5 - resolution_big / 2) * meterPerPixel_x;
            double world_y = (j + 0.5 - resolution_big / 2) * meterPerPixel_y;

            Vertex v;
            v.x = world_x;
            v.y = world_y;
            v.z = world_z;
            //std::cout<<v.x<<"  ,  "<<v.y<<"  ,  "<<v.z<<std::endl;
            point_cloud.push_back(v);
        }
    }

    return point_cloud;
}

std::vector<Vertex> perspectiveDepthImageToPointCloud(const cv::Mat& depthImg, const std::string& jsonPath) {
    std::ifstream i(jsonPath);
    json data;
    i >> data;

    Eigen::Matrix4f Base_transformation_wrt_Camera = Eigen::Matrix4f::Identity();
    Base_transformation_wrt_Camera(2, 2) = -1;
    Base_transformation_wrt_Camera(1, 1) = -1;
    Base_transformation_wrt_Camera(2, 3) = data["location"][2];
    Eigen::Vector3f camera_eye_position(0, 0, data["location"][2]);

    // Assuming you have these view and projection matrices computed somewhere
    Eigen::Matrix4f viewMatrix; // Populate this
    viewMatrix << 1.0, -0.0,  0.0, -0.0,
              0.0,  1.0,  0.0, -0.0,
             -0.0, -0.0,  1.0, -0.40000001,
              0.0,  0.0,  0.0,  1.0;
    Eigen::Matrix4f projectionMatrix; // Populate this
    projectionMatrix << 2.00568962, 0.0,        0.0, 0.0,
                    0.0,        2.00568962, 0.0, 0.0,
                    0.0,        0.0,       -3.0, -0.80000007,
                    0.0,        0.0,       -1.0, 0.0;

    int img_height = data["resolutionY"];
    int img_width = data["resolutionX"];
    
    int stepX = 1;
    int stepY = 1;
    std::vector<Vertex> pointCloud;

    Eigen::Matrix4f tran_pix_world = (projectionMatrix * viewMatrix).inverse();
    std::cout<<tran_pix_world<<std::endl;

    for (int h = 0; h < img_height; h += stepY) {
        for (int w = 0; w < img_width; w += stepX) {
            float depth = depthImg.at<float>(h, w);
            if (depth == 0) continue;
            //if(depth<0.39)
            //	std::cout<<depth<<std::endl;

            float x = (2.0f * w - img_width) / img_width;
            float y = -(2.0f * h - img_height) / img_height;
            float z = 2.0f * depth - 1;
            //std::cout<<x<<"  ,  "<<y<<"  ,  "<<z<<std::endl;
            Eigen::Vector4f pixPos(x, y, z, 1);
            Eigen::Vector4f position = tran_pix_world * pixPos;

            Eigen::Vector3f point_world_loc = position.head<3>() / position[3];
            //std::cout<<point_world_loc<<std::endl;
            Eigen::Vector3f point_loc = Base_transformation_wrt_Camera.block<3, 3>(0, 0) * point_world_loc + Base_transformation_wrt_Camera.block<3, 1>(0, 3);
            pointCloud.push_back(Vertex{point_loc.x(), point_loc.y(), point_loc.z()});
            //std::cout<<point_loc.x()<<"  ,  "<< point_loc.y()<<"  ,  "<< point_loc.z()<<std::endl;
        }
    }

    return pointCloud;
}

void divideByLargest(cv::Mat& matrix, const bool& debug = false) {
    double maxVal;
    cv::minMaxLoc(matrix, nullptr, &maxVal);
    if (debug) {
        cout << "Max value in cv::Mat: " << maxVal << endl;
    }
    matrix /= maxVal;
}

void normalizeMat(cv::Mat& input, const bool& debug = false) {
    double minVal, maxVal;
    cv::minMaxLoc(input, &minVal, &maxVal);

    if (debug) {
        cout << "Normalizing data between 0-1" << endl;
        cout << "\tLargest Value: " << maxVal << endl;
        cout << "\tSmallest Value: " << minVal << endl;
    }

    if (maxVal - minVal != 0) {
        double scale = 1.0 / (maxVal - minVal);
        double shift = -minVal / (maxVal - minVal);
        input.convertTo(input, CV_32FC1, scale, shift);
    }
}

//__global__ void cuda_internal(const double* xyz_mm, const double* radial_list_mm, int num_points, double* VoteMap_3D, int map_size_x, int map_size_y, int map_size_z) {
//    int tid = blockIdx.x * blockDim.x + threadIdx.x;
//    if (tid < num_points) {
//        double xyz[3] = { xyz_mm[tid * 3], xyz_mm[tid * 3 + 1], xyz_mm[tid * 3 + 2] };
//        double radius = radial_list_mm[tid];
//        double factor = (3.0 * sqrt(3.0)) / 4.0;
//
//        for (int i = 0; i < map_size_x; ++i) {
//            for (int j = 0; j < map_size_y; ++j) {
//                for (int k = 0; k < map_size_z; ++k) {
//                    double distance = sqrt(pow(i - xyz[0], 2) + pow(j - xyz[1], 2) + pow(k - xyz[2], 2));
//                    if (radius - distance < factor && radius - distance >= 0) {
//                        //Race condition error, figure out how to atomic add
//                        VoteMap_3D[i * map_size_y * map_size_z + j * map_size_z + k] += 1;                                         
//                    }
//                }
//            }
//        }
//    }
//}



//TODO :: Check if racecondition affects performance
void fast_for_cpu(const std::vector<Vertex>& xyz_mm, const std::vector<double>& radial_list_mm, int* VoteMap_3D, const int& vote_map_size) {
    const double factor = (std::pow(3, 0.5) / 4.0);
    const int start = 0;

#pragma omp parallel for
    for (int count = 0; count < xyz_mm.size(); ++count) {
        const Vertex xyz = xyz_mm[count];
        const int radius = round(radial_list_mm[count]);

        for (int i = start; i < vote_map_size; i++) {
            double x_diff = i - xyz.x;

            for (int j = start; j < vote_map_size; j++) {
                double y_diff = j - xyz.y;

                for (int k = start; k < vote_map_size; k++) {
                    double z_diff = k - xyz.z;
                    double distance = sqrt(x_diff * x_diff + y_diff * y_diff + z_diff * z_diff);

                    if (radius - distance < factor && radius - distance > 0) {
                        int index = i * vote_map_size * vote_map_size + j * vote_map_size + k;
                        //Possible race condition, check if the accuracy decreases here
                        VoteMap_3D[index] += 1;
                    }
                }
            }
        }
    }
}

void fast_for_cpu(const Eigen::MatrixXd& xyz_mm, const Eigen::MatrixXd& radial_list_mm, int* VoteMap_3D, const int& vote_map_size) {
    const double factor = (std::pow(3, 0.5) / 4.0);
    const int start = 0;

#pragma omp parallel for
    for (int count = 0; count < xyz_mm.rows(); ++count) {
        double x = xyz_mm(count,0);
        double y = xyz_mm(count,1);
        double z = xyz_mm(count,2);
        double radius = radial_list_mm(count,0);

        for (int i = start; i < vote_map_size; i++) {
            double x_diff = i - x;

            for (int j = start; j < vote_map_size; j++) {
                double y_diff = j - y;

                for (int k = start; k < vote_map_size; k++) {
                    double z_diff = k - z;
                    double distance = sqrt(x_diff * x_diff + y_diff * y_diff + z_diff * z_diff);

                    if (radius - distance < factor && radius - distance > 0) {
                        int index = i * vote_map_size * vote_map_size + j * vote_map_size + k;
                        //Possible race condition, check if the accuracy decreases here
                        VoteMap_3D[index] += 1;
                    }
                }
            }
        }
    }
}




Vector3d Accumulator_3D(const vector<Vertex>& xyz, const vector<double>& radial_list, const bool& debug = false) {

    double acc_unit = 5;
    // unit 5mm
    vector<Vertex> xyz_mm(xyz.size());



    for (int i = 0; i < xyz.size(); i++) {
        xyz_mm[i].x = xyz[i].x * 1000 / acc_unit;
        xyz_mm[i].y = xyz[i].y * 1000 / acc_unit;
        xyz_mm[i].z = xyz[i].z * 1000 / acc_unit;
    }

    double x_mean_mm = 0;
    double y_mean_mm = 0;
    double z_mean_mm = 0;

    for (int i = 0; i < xyz_mm.size(); i++) {
        x_mean_mm += xyz_mm[i].x;
        y_mean_mm += xyz_mm[i].y;
        z_mean_mm += xyz_mm[i].z;
    }

    x_mean_mm /= xyz_mm.size();
    y_mean_mm /= xyz_mm.size();
    z_mean_mm /= xyz_mm.size();


    for (int i = 0; i < xyz_mm.size(); i++) {
        xyz_mm[i].x -= x_mean_mm;
        xyz_mm[i].y -= y_mean_mm;
        xyz_mm[i].z -= z_mean_mm;
    }


    vector<double> radial_list_mm(radial_list.size());

    for (int i = 0; i < radial_list.size(); ++i) {
        radial_list_mm[i] = radial_list[i] * 100 / acc_unit;
    }


    double x_mm_min = 0;
    double y_mm_min = 0;
    double z_mm_min = 0;

    for (int i = 0; i < xyz_mm.size(); i++) {
        x_mm_min = min(x_mm_min, xyz_mm[i].x);
        y_mm_min = min(y_mm_min, xyz_mm[i].y);
        z_mm_min = min(z_mm_min, xyz_mm[i].z);
    }

    double xyz_mm_min = min(x_mm_min, min(y_mm_min, z_mm_min));

    double radius_max = radial_list_mm[0];

    for (int i = 0; i < radial_list_mm.size(); i++) {
        if (radius_max < radial_list_mm[i]) {
            radius_max = radial_list_mm[i];
        }
    }

    int zero_boundary = static_cast<int>(xyz_mm_min - radius_max) + 1;



    if (zero_boundary < 0) {
        for (int i = 0; i < xyz_mm.size(); i++) {
            xyz_mm[i].x -= zero_boundary;
            xyz_mm[i].y -= zero_boundary;
            xyz_mm[i].z -= zero_boundary;
        }
    }

    double x_mm_max = 0;
    double y_mm_max = 0;
    double z_mm_max = 0;

    for (int i = 0; i < xyz_mm.size(); i++) {
        x_mm_max = max(x_mm_max, xyz_mm[i].x);
        y_mm_max = max(y_mm_max, xyz_mm[i].y);
        z_mm_max = max(z_mm_max, xyz_mm[i].z);
    }

    double xyz_mm_max = max(x_mm_max, max(y_mm_max, z_mm_max));

    int length = static_cast<int>(xyz_mm_max);

    int vote_map_dim = length + static_cast<int>(radius_max);

    int total_size = vote_map_dim * vote_map_dim * vote_map_dim;

    int* VoteMap_3D = new int[total_size]();


    //if (use_cuda && !debug) {
        //cout << "Using GPU for fast_for" << endl;
        ////Initialize fast for on GPU
        //double* device_xyz_mm;
        //double* device_radial_list_mm;
        //double* device_VoteMap_3D;
        //
        //cudaMalloc((void**)&device_xyz_mm, xyz_mm.size() * sizeof(double));
        //cudaMalloc((void**)&device_radial_list_mm, radial_list_mm.size() * sizeof(double));
        //cudaMalloc((void**)&device_VoteMap_3D, VoteMap_3D.size() * sizeof(double));
        //
        //cudaMemcpy(device_xyz_mm, xyz_mm.data(), xyz_mm.size() * sizeof(double), cudaMemcpyHostToDevice);
        //cudaMemcpy(device_radial_list_mm, radial_list_mm.data(), radial_list_mm.size() * sizeof(double), cudaMemcpyHostToDevice);
        //cudaMemset(device_VoteMap_3D, 0, VoteMap_3D.size() * sizeof(double));
        //
        //int num_points = static_cast<int>(xyz.points_.size());
        //
        //int threads_per_block = 256;
        //int blocks_per_grid = (num_points + threads_per_block - 1) / threads_per_block;
        //
        //cuda_internal <<<blocks_per_grid, threads_per_block>>> (device_xyz_mm, device_radial_list_mm, num_points, device_VoteMap_3D, VoteMap_3D.size(), VoteMap_3D[0].size(), VoteMap_3D[0][0].size());
        //
        //cudaMemcpy(VoteMap_3D.data(), device_VoteMap_3D, VoteMap_3D.size() * sizeof(double), cudaMemcpyDeviceToHost);
        //
        //cudaFree(device_xyz_mm);
        //cudaFree(device_radial_list_mm);
        //cudaFree(device_VoteMap_3D);
    //}

    fast_for_cpu(xyz_mm, radial_list_mm, VoteMap_3D, vote_map_dim);

    vector<Eigen::Vector3i> centers;

    int max_vote = 0;

    for (int i = 0; i < vote_map_dim; i++) {
        for (int j = 0; j < vote_map_dim; j++) {
            for (int k = 0; k < vote_map_dim; k++) {
                int index = i * vote_map_dim * vote_map_dim + j * vote_map_dim + k;
                int current_vote = VoteMap_3D[index];
                if (current_vote > max_vote) {
                    centers.clear();
                    max_vote = current_vote;
                    centers.push_back(Eigen::Vector3i(i, j, k));
                }
                else if (current_vote == max_vote) {
                    centers.push_back(Eigen::Vector3i(i, j, k));
                }
            }
        }
    }

    delete[] VoteMap_3D;

    //cout << "Centers: " << endl;
    //for (auto center : centers) {
    //    cout << "\t" << center << endl;
    //}

    if (debug) {
        cout << "\tMax vote: " << max_vote << endl;
        cout << "\tCenter: " << centers[0][0] << " " << centers[0][1] << " " << centers[0][2] << endl;
    }

    //if (centers.size() > 1) {
    //    cout << centers.size() << " centers located." << endl;
    //}

    Eigen::Vector3d center = centers[0].cast<double>();
    if (zero_boundary < 0) {
        center.array() += zero_boundary;
    }

    center[0] = (center[0] + x_mean_mm + 0.5) * acc_unit;
    center[1] = (center[1] + y_mean_mm + 0.5) * acc_unit;
    center[2] = (center[2] + z_mean_mm + 0.5) * acc_unit;

    return center;
}

Vector3d Accumulator_3D(const Eigen::MatrixXd& xyz_r, const bool& debug = false) {

    double acc_unit = 5;
    // unit 5mm
    //vector<Vertex> xyz_mm(xyz.size());
    Eigen::MatrixXd xyz_mm = xyz_r.leftCols(3).array()*double(1000)/acc_unit;
    Eigen::MatrixXd radial_list_mm = xyz_r.col(3).array()* 100 / acc_unit;



    // for (int i = 0; i < xyz.size(); i++) {
    //     xyz_mm[i].x = xyz[i].x * 1000 / acc_unit;
    //     xyz_mm[i].y = xyz[i].y * 1000 / acc_unit;
    //     xyz_mm[i].z = xyz[i].z * 1000 / acc_unit;
    // }

    double x_mean_mm = 0;
    double y_mean_mm = 0;
    double z_mean_mm = 0;

    // for (int i = 0; i < xyz_mm.size(); i++) {
    //     x_mean_mm += xyz_mm[i].x;
    //     y_mean_mm += xyz_mm[i].y;
    //     z_mean_mm += xyz_mm[i].z;
    // }

    // x_mean_mm /= xyz_mm.size();
    // y_mean_mm /= xyz_mm.size();
    // z_mean_mm /= xyz_mm.size();
    x_mean_mm = xyz_mm.col(0).mean();
    y_mean_mm = xyz_mm.col(1).mean();
    z_mean_mm = xyz_mm.col(2).mean();

    //auto col_count = xyz_mm.cols();
    xyz_mm.col(0).array() -= x_mean_mm;
    xyz_mm.col(1).array() -= y_mean_mm;
    xyz_mm.col(2).array() -= z_mean_mm;
    // xyz_mm.col(0) = xyz_mm.col(0) - x_mean_mm;
    // xyz_mm.col(1) = xyz_mm.col(1) - y_mean_mm;
    // xyz_mm.col(2) = xyz_mm.col(2) - z_mean_mm;

    double x_mm_min = 0;
    double y_mm_min = 0;
    double z_mm_min = 0;

    x_mm_min = xyz_mm.col(0).minCoeff();
    y_mm_min = xyz_mm.col(1).minCoeff();
    z_mm_min = xyz_mm.col(2).minCoeff();
    //cout<<"x_min_mm: "<< x_mm_min <<endl;
    //cout<<"y_min_mm: "<< y_mm_min <<endl;
    //cout<<"z_min_mm: "<< z_mm_min <<endl;
    // for (int i =0; i < xyz_mm.rows(); i++) {
    //     x_mm_min = min(x_mm_min, xyz_mm(i,0));
    //     y_mm_min = min(y_mm_min, xyz_mm(i,1));
    //     z_mm_min = min(z_mm_min, xyz_mm(i,2));
    // }

    // cout<<"x_min_mm: "<< x_mm_min <<endl;
    // cout<<"y_min_mm: "<< y_mm_min <<endl;
    // cout<<"z_min_mm: "<< z_mm_min <<endl;


    // for (int i = 0; i < xyz_mm.size(); i++) {
    //     xyz_mm[i].x -= x_mean_mm;
    //     xyz_mm[i].y -= y_mean_mm;
    //     xyz_mm[i].z -= z_mean_mm;
    // }


    // vector<double> radial_list_mm(radial_list.size());

    // for (int i = 0; i < radial_list.size(); ++i) {
    //     radial_list_mm[i] = radial_list[i] * 100 / acc_unit;
    // }




    // for (int i = 0; i < xyz_mm.size(); i++) {
    //     // x_mm_min = min(x_mm_min, xyz_mm[i].x);
    //     // y_mm_min = min(y_mm_min, xyz_mm[i].y);
    //     // z_mm_min = min(z_mm_min, xyz_mm[i].z);
    // }

    double xyz_mm_min = min(x_mm_min, min(y_mm_min, z_mm_min));

    // double radius_max = radial_list_mm(0);

    // for (int i = 0; i < radial_list_mm.size(); i++) {
    //     if (radius_max < radial_list_mm(i)) {
    //         radius_max = radial_list_mm(i);
    //     }
    // }
    // cout<<"radius_max: "<<radius_max<<endl;
    double radius_max = radial_list_mm.maxCoeff();
    //cout<<"radius_max: "<<radius_max<<endl;

    int zero_boundary = static_cast<int>(xyz_mm_min - radius_max) + 1;



    if (zero_boundary < 0) {
        // for (int i = 0; i < xyz_mm.size(); i++) {
        //     xyz_mm[i].x -= zero_boundary;
        //     xyz_mm[i].y -= zero_boundary;
        //     xyz_mm[i].z -= zero_boundary;
        // }
        xyz_mm.array()-=zero_boundary;
    }

    double x_mm_max = 0;
    double y_mm_max = 0;
    double z_mm_max = 0;

    // for (int i = 0; i < xyz_mm.size(); i++) {
    //     x_mm_max = max(x_mm_max, xyz_mm[i].x);
    //     y_mm_max = max(y_mm_max, xyz_mm[i].y);
    //     z_mm_max = max(z_mm_max, xyz_mm[i].z);
    // }
    x_mm_max = xyz_mm.col(0).maxCoeff();
    y_mm_max = xyz_mm.col(1).maxCoeff();
    z_mm_max = xyz_mm.col(2).maxCoeff();

    double xyz_mm_max = max(x_mm_max, max(y_mm_max, z_mm_max));

    int length = static_cast<int>(xyz_mm_max);

    int vote_map_dim = length + static_cast<int>(radius_max);

    int total_size = vote_map_dim * vote_map_dim * vote_map_dim;

    int* VoteMap_3D = new int[total_size]();


    //if (use_cuda && !debug) {
        //cout << "Using GPU for fast_for" << endl;
        ////Initialize fast for on GPU
        //double* device_xyz_mm;
        //double* device_radial_list_mm;
        //double* device_VoteMap_3D;
        //
        //cudaMalloc((void**)&device_xyz_mm, xyz_mm.size() * sizeof(double));
        //cudaMalloc((void**)&device_radial_list_mm, radial_list_mm.size() * sizeof(double));
        //cudaMalloc((void**)&device_VoteMap_3D, VoteMap_3D.size() * sizeof(double));
        //
        //cudaMemcpy(device_xyz_mm, xyz_mm.data(), xyz_mm.size() * sizeof(double), cudaMemcpyHostToDevice);
        //cudaMemcpy(device_radial_list_mm, radial_list_mm.data(), radial_list_mm.size() * sizeof(double), cudaMemcpyHostToDevice);
        //cudaMemset(device_VoteMap_3D, 0, VoteMap_3D.size() * sizeof(double));
        //
        //int num_points = static_cast<int>(xyz.points_.size());
        //
        //int threads_per_block = 256;
        //int blocks_per_grid = (num_points + threads_per_block - 1) / threads_per_block;
        //
        //cuda_internal <<<blocks_per_grid, threads_per_block>>> (device_xyz_mm, device_radial_list_mm, num_points, device_VoteMap_3D, VoteMap_3D.size(), VoteMap_3D[0].size(), VoteMap_3D[0][0].size());
        //
        //cudaMemcpy(VoteMap_3D.data(), device_VoteMap_3D, VoteMap_3D.size() * sizeof(double), cudaMemcpyDeviceToHost);
        //
        //cudaFree(device_xyz_mm);
        //cudaFree(device_radial_list_mm);
        //cudaFree(device_VoteMap_3D);
    //}

    fast_for_cpu(xyz_mm, radial_list_mm, VoteMap_3D, vote_map_dim);

    vector<Eigen::Vector3i> centers;

    int max_vote = 0;

    for (int i = 0; i < vote_map_dim; i++) {
        for (int j = 0; j < vote_map_dim; j++) {
            for (int k = 0; k < vote_map_dim; k++) {
                int index = i * vote_map_dim * vote_map_dim + j * vote_map_dim + k;
                int current_vote = VoteMap_3D[index];
                if (current_vote > max_vote) {
                    centers.clear();
                    max_vote = current_vote;
                    centers.push_back(Eigen::Vector3i(i, j, k));
                }
                else if (current_vote == max_vote) {
                    centers.push_back(Eigen::Vector3i(i, j, k));
                }
            }
        }
    }

    delete[] VoteMap_3D;

    //cout << "Centers: " << endl;
    //for (auto center : centers) {
    //    cout << "\t" << center << endl;
    //}

    if (debug) {
        cout << "\tMax vote: " << max_vote << endl;
        cout << "\tCenter: " << centers[0][0] << " " << centers[0][1] << " " << centers[0][2] << endl;
    }

    //if (centers.size() > 1) {
    //    cout << centers.size() << " centers located." << endl;
    //}

    Eigen::Vector3d center = centers[0].cast<double>();
    if (zero_boundary < 0) {
        center.array() += zero_boundary;
    }

    center[0] = (center[0] + x_mean_mm + 0.5) * acc_unit;
    center[1] = (center[1] + y_mean_mm + 0.5) * acc_unit;
    center[2] = (center[2] + z_mean_mm + 0.5) * acc_unit;

    return center;
}
