//
// Created by mry on 23-12-7.
//
#include <iostream>
using namespace std;
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <thread>

#include "hik_camera/HikCam.hpp"
#include "Detector/include/detector.h"
#include "visionSerial-main/include/serial.h"

HikCam cam;
cv::Mat img;

Detector::LightParams l{0.1, 0.5, 45.0};
Detector::ArmorParams a{0.7, 0.8, 3.2, 3.2, 5.5, 35.0};
Detector armor(230, l, a);

visionSerial Serial("/dev/ttyACM0",115200);
visionMsg msg1;
robotMsg msg2;

/*
 * 初始化HikCam
 */
void init_hik(){
    cam.StartDevice(0);               //打开0号相机
    cam.SetResolution(1280,1024);     //设置分辨率
    cam.SetPixelFormat(17301514);          //设置像素格式PixelType_Gvsp_BayerGR8
    cam.SetExposureTime(4000);                   //设置曝光时间
    cam.SetGAIN(10.0);                    //设置增益
    cam.SetFrameRate(120);                       //设置帧率上限
    cam.SetStreamOn();                           //开始取流
}

/*
 * 不断读取图像
 */
void img_thr(){
    while(true){
        cam.GetMat(img);
    }
}

void serial_thr(){
    while(true) {
        msg1 = {0xA5, armor.send_yaw, armor.send_pitch};
        msg2 = {0xA5, false, 0, 0, 0, 15};
        Serial.visionUpdate(msg1);     // 更新自瞄数据

        Serial.robotUpdate(msg2);      // 更新机器人数据
        armor.detect_color = msg2.foeColor;
        armor.speed = msg2.muzzleSpeed;
        usleep(10000);
    }
}

int main() {

    thread imgTHR(img_thr);
    imgTHR.detach();

    thread serialTHR(serial_thr);
    serialTHR.detach();

    while (true) {
//        auto start = std::chrono::high_resolution_clock::now();
        cv::namedWindow("image", CV_WINDOW_AUTOSIZE);

        //  Detect Start!
        armor.detect(img);
        armor.getAllNumbersImage();
        armor.drawResults(img);

        //  for debugger
        cout<<"send_yaw: "<< armor.send_yaw<<endl;
        cout<<"send_pitch: "<< armor.send_pitch<<endl;
        cout<<"reciece_color: "<< armor.detect_color<<endl;
        cout<<"recieve_speed: "<< armor.speed<<endl;

        imshow("image", img);
        if (cv::waitKey(10) == 27)        //按下Esc建结束
            break;

//        auto end = std::chrono::high_resolution_clock::now();
//        std::chrono::duration<double> total = end - start;
//        cout << "fps：" << 1/total.count() << endl;
//        std::string fps("fps : ");
//        cv::putText(img, fps+to_string(1/total.count()), cv::Point2i(20, 370), cv::FONT_HERSHEY_SIMPLEX, 0.8,
//                    cv::Scalar(0, 255, 255), 2);
    }
}





