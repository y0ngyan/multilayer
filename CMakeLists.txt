cmake_minimum_required(VERSION 3.0.2)
project(multilayer)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)
# ADD_COMPILE_OPTIONS(-std=c++11)s
ADD_COMPILE_OPTIONS(-std=c++14)
set(CMAKE_CXX_FLAGS "-std=c++17 ${CMAKE_CXX_FLAGS} -O3 -Wall -pthread") # -Wextra -Werror
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
# set(CMAKE_BUILD_TYPE "RELEASE")
set(CMAKE_BUILD_TYPE Debug)


## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
  pcl_ros
  pcl_conversions
)

# 添加消息定义
add_message_files(
  FILES
  VoxelGridMsg.msg
  VoxelGridMsgArray.msg
)

# 生成消息
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES multires
 CATKIN_DEPENDS message_runtime std_msgs geometry_msgs
#  CATKIN_DEPENDS rospcpp rospy std_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

find_package(OpenCV REQUIRED)
find_package(PCL 1.7 REQUIRED)
find_package(Eigen3 REQUIRED)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  SYSTEM
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})

add_executable(global_gridmap_node
  include/GlobalGridMap.hpp
  src/GlobalGridMap.cpp
  test/global_gridmap_node.cpp
)

target_link_libraries(global_gridmap_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

add_executable(multilayer_node
  include/MultiLayerSOGMMap.hpp
  include/RayCast.hpp
  src/MultiLayerSOGMMap.cpp
  src/RayCast.cpp
  test/multilayer_node.cpp
)

target_link_libraries(multilayer_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_executable(pointcloud_multilayer_node
  include/MultiLayerSOGMMap.hpp
  include/RayCast.hpp
  src/MultiLayerSOGMMap.cpp
  src/RayCast.cpp
  test/pointcloud_multilayer_node.cpp
)

target_link_libraries(pointcloud_multilayer_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
)