cmake_minimum_required(VERSION 3.0.2)
project(shield_perception)

add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  geometry_msgs
  sensor_msgs
  shield_planner_msgs
  std_msgs
  visualization_msgs
  rospy
)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

find_path(ZED_INCLUDE_DIRS sl/Camera.hpp
  HINTS /usr/local/zed/include
)
find_library(ZED_LIBRARIES
  NAMES sl_zed
  HINTS /usr/local/zed/lib
)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS
    /usr/include/x86_64-linux-gnu
    /usr/include
    /usr/local/cuda/include
    /usr/local/cuda/targets/x86_64-linux/include
    /usr/local/cuda-12.4/targets/x86_64-linux/include
    /usr/local/cuda-12.2/targets/x86_64-linux/include
    /usr/local/cuda-12.1/targets/x86_64-linux/include
)
find_library(TENSORRT_NVINFER_LIBRARY
  NAMES nvinfer
  HINTS
    /usr/lib/x86_64-linux-gnu
    /usr/local/cuda/lib64
    /usr/local/cuda/targets/x86_64-linux/lib
    /usr/local/cuda-12.4/targets/x86_64-linux/lib
    /usr/local/cuda-12.2/targets/x86_64-linux/lib
    /usr/local/cuda-12.1/targets/x86_64-linux/lib
)
find_library(TENSORRT_ONNX_PARSER_LIBRARY
  NAMES nvonnxparser
  HINTS
    /usr/lib/x86_64-linux-gnu
    /usr/local/cuda/lib64
    /usr/local/cuda/targets/x86_64-linux/lib
    /usr/local/cuda-12.4/targets/x86_64-linux/lib
    /usr/local/cuda-12.2/targets/x86_64-linux/lib
    /usr/local/cuda-12.1/targets/x86_64-linux/lib
)
find_library(CUDA_RUNTIME_LIBRARY
  NAMES cudart
  HINTS
    /usr/local/cuda/lib64
    /usr/local/cuda/targets/x86_64-linux/lib
    /usr/local/cuda-12.4/targets/x86_64-linux/lib
    /usr/local/cuda-12.2/targets/x86_64-linux/lib
    /usr/local/cuda-12.1/targets/x86_64-linux/lib
    /usr/lib/x86_64-linux-gnu
)
find_path(CUDA_INCLUDE_DIR cuda_runtime_api.h
  HINTS
    /usr/local/cuda/include
    /usr/local/cuda/targets/x86_64-linux/include
    /usr/local/cuda-12.4/targets/x86_64-linux/include
    /usr/local/cuda-12.2/targets/x86_64-linux/include
    /usr/local/cuda-12.1/targets/x86_64-linux/include
    /usr/include
)

if(NOT ZED_INCLUDE_DIRS OR NOT ZED_LIBRARIES)
  message(FATAL_ERROR "ZED SDK C++ headers/libraries not found. Install the ZED SDK or set ZED_INCLUDE_DIRS/ZED_LIBRARIES.")
endif()

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_NVINFER_LIBRARY OR NOT TENSORRT_ONNX_PARSER_LIBRARY OR NOT CUDA_RUNTIME_LIBRARY OR NOT CUDA_INCLUDE_DIR)
  message(FATAL_ERROR "TensorRT/CUDA headers/libraries not found. Install TensorRT/CUDA or set TENSORRT_INCLUDE_DIR, TENSORRT_NVINFER_LIBRARY, TENSORRT_ONNX_PARSER_LIBRARY, and CUDA_RUNTIME_LIBRARY.")
endif()

catkin_package(
  CATKIN_DEPENDS roscpp sensor_msgs shield_planner_msgs std_msgs visualization_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${ZED_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIR}
)

add_executable(track_ball_meancovariance_node
  src/track_ball_meancovariance_node.cpp
)
add_dependencies(track_ball_meancovariance_node
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(track_ball_meancovariance_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${ZED_LIBRARIES}
  ${TENSORRT_NVINFER_LIBRARY}
  ${CUDA_RUNTIME_LIBRARY}
)

add_executable(build_tensorrt_engine_trt8
  src/build_tensorrt_engine_trt8.cpp
)
target_link_libraries(build_tensorrt_engine_trt8
  ${TENSORRT_NVINFER_LIBRARY}
  ${TENSORRT_ONNX_PARSER_LIBRARY}
  ${CUDA_RUNTIME_LIBRARY}
)
