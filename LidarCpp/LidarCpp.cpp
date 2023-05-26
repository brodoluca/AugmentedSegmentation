//#include "helper.hpp"
//
//
//
//
//
//int main()
//{
//    if (!camera.isOpened()) {
//        printf("Camera failed \n");
//        return 1;
//    }
//
//    // Initialize Winsock  
//    WSADATA wsaData;
//    int result = WSAStartup(MAKEWORD(2, 2), &wsaData);
//    if (result != 0) {
//        printf("WSAStartup failed: %d\n", result);
//        return 2;
//    }
//
//    // Create a UDP socket
//    SOCKET udp_socket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
//    if (udp_socket == INVALID_SOCKET) {
//        printf("socket failed: %d\n", WSAGetLastError());
//        WSACleanup();
//        return 3;
//    }
//
//    // Enable broadcast on the socket
//    BOOL broadcast_enabled = TRUE;
//    result = setsockopt(udp_socket, SOL_SOCKET, SO_BROADCAST, (const char*)&broadcast_enabled, sizeof(broadcast_enabled));
//    if (result == SOCKET_ERROR) {
//        printf("setsockopt failed: %d\n", WSAGetLastError());
//        closesocket(udp_socket);
//        WSACleanup();
//        return 4;
//    }
//
//    // Bind the socket to the local address and a random port
//    SOCKADDR_IN local_address = { 0 };
//    local_address.sin_family = AF_INET;
//    local_address.sin_port = htons(2370); // Bind to a Lidar port
//    local_address.sin_addr.s_addr = INADDR_ANY;
//    result = bind(udp_socket, (SOCKADDR*)&local_address, sizeof(local_address));
//    if (result == SOCKET_ERROR) {
//        printf("bind failed: %d\n", WSAGetLastError());
//        closesocket(udp_socket);
//        WSACleanup();
//        return 5;
//    }
//
//    // Receive data on the sockt
//    SOCKADDR_IN sender_address = { 0 };
//    int sender_address_size = sizeof(sender_address);
//
//    
//    //All the reciving and processing is happeneing here in a forever loop.
//    while (true)
//    {
//        
//        //read 500 udp packets each into a recvBuffetType struct that maches lidar output to make one 360deg image.
//        for (int i = 0; i < 500; i++) {
//            int bytes_received = recvfrom(udp_socket, receive_buffer, sizeof(receive_buffer), 0, (SOCKADDR*)&sender_address, &sender_address_size);
//            if (bytes_received == SOCKET_ERROR) {
//                printf("recvfrom failed: %d\n", WSAGetLastError());
//                closesocket(udp_socket);
//                WSACleanup();
//                return 1;
//            }
//             recvBufferType* recvBuffer = (recvBufferType*)receive_buffer;
//             LidarPayload[i] = *recvBuffer;
//        }
//
//        //after getting 500 payloads to make one 360deg image (360/0.09 = 4000 azimuth degree, 4000/8in each payload =500) you can process the data here.
//        // this could possible be done in the previous for loop but I wanted to keep the code clean and easy to read and avoid sync issues since
//        // the data is coming in as UDP and i don't want to implement time checkers.
//
//        for (recvBufferType& singlePayLoad : LidarPayload) {
//            for (int j = 0; j < 8; j++) {
//                for (int i = 0; i < 32; i++) {
//                    singlePayLoad.block[j].channel[i].distance *= singlePayLoad.header.DisUnit; //convert the distance to mm
//                    singlePayLoad.block[j].channel[i].distance /= 10; //convert the distance to cm
//                }
//            }
//
//        }
//
//         
//         // create a 32(*scale)by4000 color image where the index of the image is based on the azimuth and the value is the distance and a 32(*scale)x4000 
//         //gray scale image where the index of the image is based on the azimuth and the value is the reflectance
//         float red, green, blue; 
//         for (recvBufferType& singlePayLoad : LidarPayload) {
//             for (int j = 0; j < 8; j++) {
//                 for (int i = 0; i < 32 * scale_factor; i+=scale_factor) {
//                     //map azimuth into a value between 0-4000, min number is 0 and max number is 36000
//                     int x = float(singlePayLoad.block[j].azimuth) / 9;
//                     getcolor(singlePayLoad.block[j].channel[i/ scale_factor].distance, 1500, red, green, blue); //more than 8meters technically. increase for more range
//                     for (int add = 0; add < 7; add++) {
//                         lidarImage.at<cv::Vec3b>(i+add, x) = cv::Vec3b(red * 255, green * 255, blue * 255);
//                         reflectanceImage.at<uchar>(i+add, x) = singlePayLoad.block[j].channel[i / scale_factor].reflectance;
//                         #ifdef LEARN
//                            //StorageImage.at<cv::Vec2f>(i + add, x) = cv::Vec2f(singlePayLoad.block[j].channel[i / scale_factor].distance, singlePayLoad.block[j].channel[i / scale_factor].reflectance);
//                         #endif // LEARN
//                         
//                     }
//                 }
//             }
//         }
//         camera >> RGB_Img;
//         int x = 1530, y = 0, height = (32 * scale_factor), width = RGB_Img.cols;
//         cv::Rect roi(x, y, width, height);
//         cv::Mat croppedLidarImage = lidarImage(roi);
//         cv::Mat croppedReflectanceImage = reflectanceImage(roi);
//         //cv::Mat croppedReflectanceImage = reflectanceImage(roi);
//        //show and save the images.
//
//        
//            cv::imshow("Camera Image", RGB_Img);
//            cv::imshow("Lidar Distance", croppedLidarImage);
//            cv::imshow("Lidar Reflectance", croppedReflectanceImage);
//        #ifdef LEARN
//            fs_lidar <<"img_" +std::to_string(img_index) << croppedLidarImage;
//            fs_reflect << "img_" + std::to_string(img_index) << croppedReflectanceImage;
//            cv::imwrite("C:\\Users\\z-3li\\OneDrive\\Documents\\LidarCameraSegmentation\\LidarCpp\\1505\\img_" + std::to_string(img_index)+".png", RGB_Img);
//            cv::imwrite("C:\\Users\\z-3li\\OneDrive\\Documents\\LidarCameraSegmentation\\LidarCpp\\1505_lidar\\img_" + std::to_string(img_index) + ".png", croppedLidarImage);
//            cv::imwrite("C:\\Users\\z-3li\\OneDrive\\Documents\\LidarCameraSegmentation\\LidarCpp\\1505_reflectance\\img_" + std::to_string(img_index) + ".png", croppedReflectanceImage);
//            img_index++;
//            std::cout << img_index << std::endl;   
//        #endif // LEARN
//
//        if (cv::waitKey(1) == 'q') {
//            #ifdef LEARN
//            fs_lidar.release();
//            fs_reflect.release();
//            #endif // LEARN
//			break;
//		}
//    }
//
//    // Cleanup
//    closesocket(udp_socket);
//    WSACleanup();
//}
