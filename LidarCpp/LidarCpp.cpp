// LidarCpp.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#define _WINSOCK_DEPRECATED_NO_WARNINGS
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <stdio.h>
#include <iostream>
#pragma comment(lib, "Ws2_32.lib")

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

};
#pragma pack(push, 1)
recvBufferType LidarPayload[500];

int scale_factor = 8; // 8x scaling factor for better visualization of the lidar data
cv::Mat lidarImage(32 * scale_factor, 4000, CV_8UC3, cv::Vec3b(0, 0, 0)); // 256 rows, 4000 colloums visualizes image of distances
cv::Mat reflectanceImage(32 * scale_factor, 4000, CV_8UC1, 1); // 256 rows, 4000 colloums visualizes image of reflectance
#ifdef LEARN
    cv::Mat StorageImage(32 * scale_factor, 4000, CV_32FC2, cv::Vec2f(0, 0)); // 256 rows, 4000 colloums of both distance and reflectance for machine learning
    cv::FileStorage fs("12_05.bin", cv::FileStorage::WRITE); 
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

int main()
{
    // Initialize Winsock  
    WSADATA wsaData;
    int result = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (result != 0) {
        printf("WSAStartup failed: %d\n", result);
        return 1;
    }

    // Create a UDP socket
    SOCKET udp_socket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (udp_socket == INVALID_SOCKET) {
        printf("socket failed: %d\n", WSAGetLastError());
        WSACleanup();
        return 1;
    }

    // Enable broadcast on the socket
    BOOL broadcast_enabled = TRUE;
    result = setsockopt(udp_socket, SOL_SOCKET, SO_BROADCAST, (const char*)&broadcast_enabled, sizeof(broadcast_enabled));
    if (result == SOCKET_ERROR) {
        printf("setsockopt failed: %d\n", WSAGetLastError());
        closesocket(udp_socket);
        WSACleanup();
        return 1;
    }

    // Bind the socket to the local address and a random port
    SOCKADDR_IN local_address = { 0 };
    local_address.sin_family = AF_INET;
    local_address.sin_port = htons(2370); // Bind to a Lidar port
    local_address.sin_addr.s_addr = INADDR_ANY;
    result = bind(udp_socket, (SOCKADDR*)&local_address, sizeof(local_address));
    if (result == SOCKET_ERROR) {
        printf("bind failed: %d\n", WSAGetLastError());
        closesocket(udp_socket);
        WSACleanup();
        return 1;
    }

    // Receive data on the sockt
    SOCKADDR_IN sender_address = { 0 };
    int sender_address_size = sizeof(sender_address);


    //All the reciving and processing is happeneing here in a forever loop.
    while (true)
    {
        
        //read 500 udp packets each into a recvBuffetType struct that maches lidar output to make one 360deg image.
        for (int i = 0; i < 500; i++) {
            int bytes_received = recvfrom(udp_socket, receive_buffer, sizeof(receive_buffer), 0, (SOCKADDR*)&sender_address, &sender_address_size);
            if (bytes_received == SOCKET_ERROR) {
                printf("recvfrom failed: %d\n", WSAGetLastError());
                closesocket(udp_socket);
                WSACleanup();
                return 1;
            }
             recvBufferType* recvBuffer = (recvBufferType*)receive_buffer;
             LidarPayload[i] = *recvBuffer;
        }

        //after getting 500 payloads to make one 360deg image (360/0.09 = 4000 azimuth degree, 4000/8in each payload =500) you can process the data here.
        // this could possible be done in the previous for loop but I wanted to keep the code clean and easy to read and avoid sync issues since
        // the data is coming in as UDP and i don't want to implement time checkers.
        for (recvBufferType& singlePayLoad : LidarPayload) {
            for (int j = 0; j < 8; j++) {
                for (int i = 0; i < 32; i++) {
                    singlePayLoad.block[j].channel[i].distance *= singlePayLoad.header.DisUnit; //convert the distance to mm
                    singlePayLoad.block[j].channel[i].distance /= 10; //convert the distance to cm
                }
            }

        }

         
         // create a 32(*scale)by4000 color image where the index of the image is based on the azimuth and the value is the distance and a 32(*scale)x4000 
         //gray scale image where the index of the image is based on the azimuth and the value is the reflectance
         float red, green, blue; 
         for (recvBufferType& singlePayLoad : LidarPayload) {
             for (int j = 0; j < 8; j++) {
                 for (int i = 0; i < 32 * scale_factor; i+=scale_factor) {
                     //map azimuth into a value between 0-4000, min number is 0 and max number is 36000
                     int x = float(singlePayLoad.block[j].azimuth) / 9;
                     getcolor(singlePayLoad.block[j].channel[i/ scale_factor].distance, 1000, red, green, blue); //8meters technically. increase for more range
                     for (int add = 0; add < 7; add++) {
                         lidarImage.at<cv::Vec3b>(i+add, x) = cv::Vec3b(red * 255, green * 255, blue * 255);
                         reflectanceImage.at<uchar>(i+add, x) = singlePayLoad.block[j].channel[i / scale_factor].reflectance;
                         #ifdef LEARN
                            StorageImage.at<cv::Vec2f>(i + add, x) = cv::Vec2f(singlePayLoad.block[j].channel[i / scale_factor].distance, singlePayLoad.block[j].channel[i / scale_factor].reflectance);
                         #endif // LEARN
                         
                     }
                 }
             }
         }

        //show and save the images. 
        cv::imshow("Lidar Distance", lidarImage);
        cv::imshow("Lidar Reflectance", reflectanceImage);
        #ifdef LEARN
                fs << "test" << StorageImage;
        #endif // LEARN

        if (cv::waitKey(1) == 'q') {
            #ifdef LEARN
                fs.release();
            #endif // LEARN
			break;
		}
    }

    // Cleanup
    closesocket(udp_socket);
    WSACleanup();
}
