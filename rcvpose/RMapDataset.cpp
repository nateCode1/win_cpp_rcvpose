#include "RMapDataset.h"


namespace fs = std::filesystem;


RMapDataset::RMapDataset(
	const std::string& root,
	const std::string& dname,
	const std::string& set,
	const std::string& obj_name
) :
	root_(root),
	dname_(dname),
	set_(set),
	obj_name_(obj_name)
{
	if (dname_ == "lm") {
		imgpath_ = root_ + "/LINEMOD/" + obj_name + "/JPEGImages/";
		radialpath1_ = root_ + "/LINEMOD/" + obj_name + "/Out_pt1_dm/";
		radialpath2_ = root_ + "/LINEMOD/" + obj_name + "/Out_pt2_dm/";
		radialpath3_ = root_ + "/LINEMOD/" + obj_name + "/Out_pt3_dm/";
		imgsetpath_ = root_ + "/LINEMOD/" + obj_name + "/Split/%s.txt";
	}
	else if (dname == "ycb") {
		std::cout << "YCB Unfinished" << std::endl;
		h5path_ = root_ + "/YCB/" + obj_name + "/";
		imgpath_ = root_ + "/YCB/" + obj_name + "/Split/ ";
	}
	else {
		std::cout << "Loading Bluewrist data " << set_ << std::endl;
		imgpath_ = root_ + "/sim" + "/JPEGImages/";
		radialpath1_ = root_ + "/sim" + "/Out_pt1_dm/";
		radialpath2_ = root_ + "/sim" + "/Out_pt2_dm/";
		radialpath3_ = root_ + "/sim" + "/Out_pt3_dm/";
		imgsetpath_ = root_ + "/sim" + "/Split/%s.txt";
	}

	//Gather all img paths if they end with .jpg
	//for (const auto& entry : fs::directory_iterator(imgpath_)) {
	//	if (entry.path().extension() == ".jpg") {
	//		std::string img_id = entry.path().filename().string();
	//		ids_.push_back(img_id);
	//	}
	//}

	std::ifstream file(imgsetpath_.replace(imgsetpath_.find("%s"), 2, set_));
	if (file.is_open()) {
		std::string line;
		while (std::getline(file, line)) {

			//remove the \n at the end of the line
			line.erase(std::remove(line.begin(), line.end(), '\n'), line.end());

			ids_.push_back(line);
		}
		file.close();
	}
	else {
		throw std::runtime_error("Error opening file: " + imgsetpath_);
	}
}


//Overriden get method, if transform is not null, apply transform to img and lbl to return img, lbl, sem_lbl
CustomExample RMapDataset::get(size_t index) {
	std::string img_id = ids_[index];

	//std::cout << imgpath_ + img_id + ".jpg" << "  ---  ";
	//std::cout << radialpath1_ + img_id + ".npy" << std::endl;

	// TODO:
	// Check type of data and shape stored in radial .npy (may not be float)

	//std::cout<<imgpath_<<" , "<<img_id<<"  ,  "<<std::endl;
	//std::cout<<radialpath1_<<" , "<<img_id<<"  ,  "<<std::endl;
	//std::cout<<radialpath2_<<" , "<<img_id<<"  ,  "<<std::endl;
	//std::cout<<radialpath3_<<" , "<<img_id<<"  ,  "<<std::endl;
	
	cv::Mat img = cv::imread(imgpath_ + img_id + ".jpg", cv::IMREAD_COLOR);

	//cv::Mat radial_kpt1 = cv::imread(radialpath1_ + img_id + ".tiff", cv::IMREAD_UNCHANGED);
	//cv::Mat radial_kpt2 = cv::imread(radialpath2_ + img_id + ".tiff", cv::IMREAD_UNCHANGED);
	//cv::Mat radial_kpt3 = cv::imread(radialpath3_ + img_id + ".tiff", cv::IMREAD_UNCHANGED);

	//cv::Mat radial_kpt1_64, radial_kpt2_64, radial_kpt3_64;
	//radial_kpt1.convertTo(radial_kpt1_64, CV_64F);
	//radial_kpt2.convertTo(radial_kpt2_64, CV_64F);
	//radial_kpt3.convertTo(radial_kpt3_64, CV_64F);

	//cv::Mat radial_kpt1 = read_bin(radialpath1_ + img_id + ".bin").clone();
	//cv::Mat radial_kpt2 = read_bin(radialpath2_ + img_id + ".bin").clone();
	//cv::Mat radial_kpt3 = read_bin(radialpath3_ + img_id + ".bin").clone();

	cv::Mat radial_kpt1 = read_npy(radialpath1_ + img_id + ".npy");
	cv::Mat radial_kpt2 = read_npy(radialpath2_ + img_id + ".npy");
	cv::Mat radial_kpt3 = read_npy(radialpath3_ + img_id + ".npy");


	//auto calculateAndPrintStats = [](const cv::Mat& mat, const std::string& filepath) {
	//	// Calculate mean
	//	cv::Scalar meanScalar = cv::mean(mat);
	//	double meanValue = meanScalar[0];

	//	// Calculate min and max values
	//	double minVal, maxVal;
	//	cv::minMaxLoc(mat, &minVal, &maxVal);

	//	// Print the results
	//	std::cout << std::endl;
	//	std::cout << "File Path: " << filepath << std::endl;
	//	std::cout << "Mean: " << meanValue << std::endl;
	//	std::cout << "Max: " << maxVal << std::endl;
	//	};

	//std::cout << std::endl;
	//// Calculate and print stats for all three matrices
	//calculateAndPrintStats(radial_kpt1, radialpath1_ + img_id + ".npy");
	//calculateAndPrintStats(radial_kpt2, radialpath2_ + img_id + ".npy");
	//calculateAndPrintStats(radial_kpt3, radialpath3_ + img_id + ".npy");
	//std::cout << std::endl;

	
	//std::cout << "########## RADIAL MAP ###########" << std::endl;
	//double min;
	//double max;
	//std::cout << radialpath1_ + img_id + "" << std::endl;
	//cv::minMaxLoc(radial_kpt1, &min, &max);
	//std::cout << "Min: " << min << "    Max: " << max << std::endl;
	//int rows = radial_kpt1.rows;
	//int cols = radial_kpt1.cols;
	//int channels = radial_kpt1.channels();
	//std::cout << "Shape of radial_kpt1: (" << rows << ", " << cols << ", " << channels << ")" << std::endl;
	//std::cout << "(322, 322): " << radial_kpt1.at<double>(322, 322) << std::endl;

	std::vector<torch::Tensor> transfromed_data = transform(img, radial_kpt1, radial_kpt2, radial_kpt3);
	
	return CustomExample(transfromed_data[0], transfromed_data[1], transfromed_data[2], transfromed_data[3], transfromed_data[4]);
}


c10::optional<size_t> RMapDataset::size() const {
	return ids_.size();
}

cv::Mat RMapDataset::read_bin(const std::string& path)
{
	std::ifstream infile(path, std::ios::binary);

	const int rows = 640;
	const int cols = 640;
	const int channels = 3;

	std::vector<float> buffer(rows * cols * channels);

	infile.read(reinterpret_cast<char*>(buffer.data()), buffer.size() * sizeof(float));
	infile.close();

	cv::Mat array(rows, cols, CV_32FC3, buffer.data());

	return array;
}

cv::Mat RMapDataset::read_npy(const std::string& path)
{
	std::vector<double> data;
	std::vector<unsigned long> shape;
	bool fortran_order;

	npy::LoadArrayFromNumpy(path, shape, fortran_order, data);

	int rows = static_cast<int>(shape[0]);
	int cols = static_cast<int>(shape[1]);
	cv::Mat mat(rows, cols, CV_64F);

	if (fortran_order) {
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				mat.at<double>(i, j) = data[i + j * rows];
			}
		}
	}
	else {
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				mat.at<double>(i, j) = data[i * cols + j];
			}
		}
	}
	return mat;
}


