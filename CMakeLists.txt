project(rgbd_vio) # RGBD-VINS 

cmake_minimum_required(VERSION 2.8.3)

# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -std=c++11 -w -g -O3 -fpermissive -mno-avx -mno-avx2")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -w -g -O3 -fpermissive")
SET(CMAKE_BUILD_TYPE "RelWithDebInfo")

if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
#set(ROS_BUILD_TYPE RelWithDebInfo)

# rosbuild_init()

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# set(OpenCV_DIR "/home/davidz/work/3rdlibs/OpenCV-2.4.10/build")
find_package(OpenCV REQUIRED)
# message("OpenCV_INCLUDE_DIR: ${OpenCV_INCLUDE_DIRS}")
# include_directories("/home/hzhang8/work/3rdlib/opencv-3.3.1/install/include")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    cv_bridge
    sensor_msgs
    std_msgs
    pcl_conversions
    pcl_ros
    message_filters
    image_transport
    camera_model # this package in VINS-Mono or VINS-Fusion 
    tf
)

catkin_package(
    # DEPENDS Eigen
    # CATKIN_DEPENDS roscpp std_msgs sensor_msgs cv_bridge
    # INCLUDE_DIRS include
    # LIBRARIES
)

SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3 QUIET)
include_directories(${EIGEN3_INCLUDE_DIR} "/usr/include/eigen3")
include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories(${catkin_INCLUDE_DIRS})

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

## work on the rgbd inertial odometry 
add_subdirectory(rvio)

