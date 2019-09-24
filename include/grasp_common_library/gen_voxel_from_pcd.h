#ifndef GEN_VOXEL_FROM_PCD_H
#define GEN_VOXEL_FROM_PCD_H
#include <ros/ros.h>
#include <geometry_msgs/Pose.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl/common/common.h>
#include "prob_grasp_planner/GenGraspVoxel.h"

/**
* The class to generate voxel from the pointcloud.
*/
class GenVoxelFromPcd{
    // voxel_grid is represented by a vector of xyz coordinates of 
    // occupied voxels.
    typedef std::vector<cv::Point3i> voxel_grid;
    typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudXYZRGB;

    public:
        GenVoxelFromPcd(std::string cam_frame_id);

         /**
         * Generate the voxel in segmented object frame from the given pcd file in camera frame. 
         * The transformation of pointcloud is done by pcl_ros pointcloud transformation with
         * ROS tf. That means, the tf of the camera frame and object frame has to be published in ROS.
         * \param seg_obj_cloud
         *      The object point cloud.
         * \param voxel_dim
         *      The voxel grid dimension.
         * \param voxel_size
         *      The size of each voxel.
         * \return 
         *      The voxel grid.
         */
 
        voxel_grid genVoxelFromPcd(const sensor_msgs::PointCloud2& seg_obj_cloud,
                                    const cv::Point3i &voxel_dim,
                                    const cv::Point3f &voxel_size);

         /**
         * Generate the voxel in segmented object frame from the segmented object pointcloud in world frame. 
         * The transformation of pointcloud is done by pcl_ros pointcloud transformation with
         * ROS tf. That means, the tf of the object frame has to be published in ROS.
         * \param voxel_dim
         *      The voxel grid dimension.
         * \param voxel_size
         *      The size of each voxel.
         * \return 
         *      The voxel grid.
         */
 
        voxel_grid genVoxelFromPcd(const std::string &pcd_file,
                                    const cv::Point3i &voxel_dim,
                                    const cv::Point3f &voxel_size);

        /**
         * ROS service handler function to generate voxel grids in object frame.
         */
        bool genVoxel(prob_grasp_planner::GenGraspVoxel::Request& req,
                        prob_grasp_planner::GenGraspVoxel::Response& res); 

        /**
         * Publish one pointcloud for testing.
         *  \param pointcloud
         *      The pointcloud.
         *  \param topic_name
         *      The topic_name to publish the pointcloud.
         */
        void publishPointcloud(const PointCloudXYZRGB &pointcloud, const std::string &topic_name);

        /**
         * Publish two pointclouds for testing.
         *  \param pointcloud
         *      The 1st pointcloud.
         *  \param topic_name
         *      The topic_name to publish the 1st pointcloud.
         *  \param pointcloud
         *      The 2nd pointcloud.
         *  \param topic_name
         *      The topic_name to publish the 2nd pointcloud.

         */
        void publishPointcloud(const PointCloudXYZRGB &pointcloud, 
                               const std::string &topic_name, 
                               const PointCloudXYZRGB &pointcloud_2, 
                               const std::string &topic_2_name);

    private:
        std::string camera_frame_id;
};

#endif

