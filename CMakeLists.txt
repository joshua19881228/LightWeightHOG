project(examples)  
find_package(OpenCV REQUIRED)
AUX_SOURCE_DIRECTORY(. DIR_SRCS)

ADD_EXECUTABLE(LightWeightHOG main)
target_link_libraries(LightWeightHOG ${OpenCV_LIBS})

cmake_minimum_required(VERSION 2.8)  
INCLUDE_DIRECTORIES(/usr/include/opencv2)

