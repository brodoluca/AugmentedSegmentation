//#include "helper.hpp"
//std::string lidarPath = "C:\\Users\\z-3li\\OneDrive\\Documents\\LidarCameraSegmentation\\LidarCpp\\1505_lidar\\";
//std::string reflectancePath = "C:\\Users\\z-3li\\OneDrive\\Documents\\LidarCameraSegmentation\\LidarCpp\\1505_reflectance\\";
//
//int main() {
//	cv::Mat lidarImage, reflectanceImage, reflectanceImageRGB, combinedImage;
//	int image_index = 0;
//	while (true) {
//		lidarImage = cv::imread(lidarPath + "img_" + std::to_string(image_index) + ".png");
//		reflectanceImage = cv::imread(reflectancePath + "img_" + std::to_string(image_index) + ".png");
//
//	
//
//		cv::merge(std::vector<cv::Mat>{ reflectanceImage, reflectanceImage, reflectanceImage }, reflectanceImageRGB);
//		//cv::addWeighted(lidarImage, 0.1, reflectanceImageRGB, 0.9, 0, combinedImage);
//		combinedImage = lidarImage * reflectanceImageRGB;
//		cv::imshow("combined", combinedImage);
//		cv::imshow("lidar", lidarImage);
//		cv::imshow("reflectance", reflectanceImageRGB);
//		image_index++;
//		if (cv::waitKey(1) == 'q') {
//			break;
//		}
//
//
//	}
//	
//
//
//
//	return 0;
//}
