#include <opencv2/opencv.hpp>

int notmain()
{
    cv::VideoCapture cap("rstp://192.168.1.202:8000");
    if (!cap.isOpened())
    {
        std::cerr << "Failed to open camera." << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (cap.read(frame))
    {
        // Process the frame here
        cv::imshow("IP Camera", frame);
        cv::waitKey(1);
    }

    return 0;
}
