cmake_minimum_required(VERSION 3.2)  
set(CMAKE_CXX_STANDARD 11)

project (sobel)   
 

# 根据你的环境设置下或者使用 find()
set(OpenCV_LIBS "path\\to\\x64\\vc15\\lib\\opencv_world450.lib") 
set(OpenCV_INCLUDE_DIRS "path\\to\\include")

include_directories("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.0\\include")
link_libraries("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.0\\lib\\x64\\OpenCL.lib")

include_directories(${OpenCV_INCLUDE_DIRS})
link_libraries(${OpenCV_LIBS})

add_executable(sobel main.cpp)  