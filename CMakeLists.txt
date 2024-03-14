cmake_minimum_required(VERSION 3.10)
project(RM_cv)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(CSerialPort REQUIRED)
include_directories(${CSerialPort_INCLUDE_DIR})

include_directories("/opt/MVS/include/")

add_executable(RM_cv
        hik_camera/HikCam.cpp
        hik_camera/HikCam.hpp

        Detector/include/detector.h
        Detector/src/detector.cpp
        Detector/include/classifier.h
        Detector/src/classifier.cpp
        Detector/include/pnp.h
        Detector/src/pnp.cpp

        visionSerial-main/include/packet.h
        visionSerial-main/include/serial.h

        debuger.cpp
        infantry.cpp

)


find_library(MVS_LIB libMvCameraControl.so /opt/MVS/lib/64/)
target_link_libraries(RM_cv ${OpenCV_LIBS} ${MVS_LIB} -lserialport ${CSerialPort_LIBRARY} -pthread )


