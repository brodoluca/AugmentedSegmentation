#define _WINSOCK_DEPRECATED_NO_WARNINGS
#include <iostream>
#include <opencv2/opencv.hpp>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <stdio.h>
#include <iostream>

#pragma comment(lib, "Ws2_32.lib")
#define LEARN

char receive_buffer[1080];
#pragma pack(push, 1)
struct recvBufferType
{
    struct PreHeaderType
    {
        uchar SOP0;
        uchar SOP1;
        uchar ProtocolversionMajor;
        uchar ProtocolversionMinor;
        unsigned short Reserverd;
    }PreHeader;

    struct headerType
    {
        uchar LaserNum;
        uchar BlockNum;
        uchar Reserverd;
        uchar DisUnit;
        uchar ReturnNumber;
        uchar UDPSeq;
    }header;

    struct blockType
    {
        unsigned short azimuth;
        struct channelType {
            unsigned short distance;
            uchar reflectance;
            uchar reserved;
        }channel[32];

    }block[8];

    struct tailType {
        uchar reserved[10];
        uchar returnMode;
        unsigned short motorSpeed;
        uchar DateTime[6];
        unsigned int TimeStamp;
        uchar FactoryInfo;
    }tail;

    struct additionalInfoType {
        unsigned int udpSeq;
    }additionalInfo;

}LidarPayload[500];
#pragma pack(push, 1)

int scale_factor = 8; // 8x scaling factor for better visualization of the lidar data
cv::Mat lidarImage(32 * scale_factor, 4000, CV_8UC3, cv::Vec3b(0, 0, 0)); // 256 rows, 4000 colloums visualizes image of distances
cv::Mat reflectanceImage(32 * scale_factor, 4000, CV_8UC1, 1); // 256 rows, 4000 colloums visualizes image of reflectance
cv::Mat RGB_Img;
cv::VideoCapture camera(0);


#ifdef LEARN
//cv::Mat StorageImage(32 * scale_factor, 4000, CV_32FC2, cv::Vec2f(0, 0)); // 256 rows, 4000 colloums of both distance and reflectance for machine learning
cv::FileStorage fs_lidar("1505_lidar.bin", cv::FileStorage::WRITE);
cv::FileStorage fs_reflect("1505_reflect.bin", cv::FileStorage::WRITE);
int img_index = 0;
#endif // LEARN

void getcolor(int p, int np, float& r, float& g, float& b) {
    // This function is only used for visualization, it only gives color to distances. reflectance is a uchar and does not need color only grayscale
    float inc = 6.0 / np;
    float x = p * inc;
    r = 0.0f; g = 0.0f; b = 0.0f;
    if ((0 <= x && x <= 1) || (5 <= x && x <= 6)) r = 1.0f;
    else if (4 <= x && x <= 5) r = x - 4;
    else if (1 <= x && x <= 2) r = 1.0f - (x - 1);
    if (1 <= x && x <= 3) g = 1.0f;
    else if (0 <= x && x <= 1) g = x - 0;
    else if (3 <= x && x <= 4) g = 1.0f - (x - 3);
    if (3 <= x && x <= 5) b = 1.0f;
    else if (2 <= x && x <= 3) b = x - 2;
    else if (5 <= x && x <= 6) b = 1.0f - (x - 5);
}