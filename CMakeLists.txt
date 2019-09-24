cmake_minimum_required(VERSION 2.8.3)
project(prob_grasp_planner)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  message_generation
  pcl_conversions
  pcl_ros
  cv_bridge
  point_cloud_segmentation
  image_transport
)

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
#catkin_python_setup()

 add_message_files(
   FILES
   HandConfig.msg
   VisualInfo.msg
 )

 add_service_files(
   FILES
   GenGraspVoxel.srv
   UpdatePalmPose.srv
   UpdatePalmPosesInObj.srv
   UpdatePriorPoses.srv
   UpdateObjectPose.srv
   GenInfVoxel.srv
   GraspPgmInf.srv
   GraspCnnInfer.srv
   GraspSampleInf.srv
   GraspActiveLearn.srv
   ActiveModelUpdate.srv
   ActiveDataUpdate.srv
   GraspVoxelInfer.srv
   GetPcdNormal.srv
 )


## Generate added messages and services with any dependencies listed here
 generate_messages(
   DEPENDENCIES
   std_msgs
   sensor_msgs
   geometry_msgs
   point_cloud_segmentation
 )

catkin_package(
  CATKIN_DEPENDS
  message_runtime
  roscpp
  rospy
  std_msgs
  pcl_conversions
  DEPENDS
  pcl_ros
  cv_bridge
  #pcl
  #opencv
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/include/grasp_common_library
)

add_executable(gen_voxel_from_pcd src/grasp_common_library/gen_voxel_from_pcd.cpp)
add_dependencies(gen_voxel_from_pcd prob_grasp_planner_generate_messages_cpp)
target_link_libraries(gen_voxel_from_pcd ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})

add_executable(gen_inference_voxel src/grasp_common_library/gen_inference_voxel.cpp)
add_dependencies(gen_inference_voxel prob_grasp_planner_generate_messages_cpp)
target_link_libraries(gen_inference_voxel ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})

add_executable(compute_pcd_normal src/grasp_common_library/compute_pcd_normal.cpp)
add_dependencies(compute_pcd_normal prob_grasp_planner_generate_messages_cpp)
target_link_libraries(compute_pcd_normal ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})

#link_directories(${OPENCV_LIBRARY_DIRS})
#add_definitions(${OPENCV_DEFINITIONS})
#link_directories(${PCL_LIBRARY_DIRS})
#add_definitions(${PCL_DEFINITIONS})

