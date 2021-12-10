cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
message(STATUS "CMake version: ${CMAKE_VERSION}")

project(alex_net_gpu LANGUAGES CUDA CXX)
# Path to weithgs for deep neural networks
add_definitions(-DMW_DL_DATA_PATH=${PROJECT_SOURCE_DIR}/src/codegen)
## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  cv_bridge
)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)


################################################
## Declare ROS messages, services and actions ##
################################################
catkin_package(
        CATKIN_DEPENDS roscpp std_msgs sensor_msgs cv_bridge
        LIBRARIES {$PROJECT_NAME}
)

file(GLOB_RECURSE cpp_srcs src/codegen/*.cpp)
file(GLOB_RECURSE cuda_srcs src/codegen/*.cu)

set(CMAKE_CUDA_FLAGS "-arch sm_35 -rdc=true -Xcudafe \"--diag_suppress=unsigned_compare_with_zero\" -Xcompiler -MMD,-MP -O2")

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
        ${catkin_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${CUDNN_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
        src/codegen/
)

# Link directories for linking
link_directories(
        ${CUDA_LIBRARY_DIRS}
        ${CUDNN_PATH} ${CUDNN_PATH}/lib ${CUDNN_PATH}/lib64
)

add_library(${PROJECT_NAME} SHARED ${cpp_srcs} ${cuda_srcs})
set_target_properties(${PROJECT_NAME}
        PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(alex_net_gpu ${CUDA_LIBRARIES} ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} cudnn /usr/local/cuda-10.2/lib64/libcublas.so)

add_executable(alex_detector src/main.cpp)
target_link_libraries(alex_detector PUBLIC ${PROJECT_NAME} ${CUDA_LIBRARIES} ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
set_target_properties(alex_detector PROPERTIES CUDA_SEPARABLE_COMPILATION ON)