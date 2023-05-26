#include <iostream>
#include <opencv2/opencv.hpp>

std::string cameraPath = "C:\\Users\\z-3li\\OneDrive\\Documents\\LidarCameraSegmentation\\1505\\";
std::string dimmingPath = "C:\\Users\\z-3li\\OneDrive\\Documents\\LidarCameraSegmentation\\1505_super_dimmed\\";
int index = 0;
float alpha = 0.3; /*< Simple contrast control 1 being current brightness and 0 being black */
int main(int argc, char** argv) {
	while (true)
	{
		cv::imwrite(dimmingPath+"img_"+ std::to_string(index) + ".png", cv::imread(cameraPath + "img_" + std::to_string(index) + ".png") * 0.5);
		index++;
	}
	return 0;
}