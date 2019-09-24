#ifndef GEN_INFERENCE_VOXEL_H
#define GEN_INFERENCE_VOXEL_H
#include <ros/ros.h>
#include <geometry_msgs/Pose.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl/common/common.h>
#include "prob_grasp_planner/GenInfVoxel.h"

/**
* The class to generate voxel from the pointcloud.
*/
class GenInferenceVoxel{
    // voxel_grid is represented by a vector of xyz coordinates of 
    // occupied voxels.
    typedef std::vector<cv::Point3i> voxel_grid;
    typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudXYZRGB;

    public:
        GenInferenceVoxel() {}
         /**
         * Generate the voxel in palm frame from the given pcd file in camera frame. 
         * The transformation of pointcloud is done by pcl_ros pointcloud transformation with
         * ROS tf. That means, the tf of the camera frame and palm frame has to be published in ROS.
         * \param pcd_file_path
         *      The pcd file path.
         * \param palm_voxel_dim
         *      The voxel grid dimension.
         * \param palm_voxel_size
         *      The size of each voxel.
         * \return 
         *      The palm voxel grid.
         */
 
        voxel_grid genVoxelFromPcd(const sensor_msgs::PointCloud2 &scene_cloud,
                                    const cv::Point3i &palm_voxel_dim,
                                    const cv::Point3f &palm_voxel_size,
                                    const std::string &object_frame_id);

        /**
         * ROS service handler function to generate voxel grids in palm frame.
         */
        bool genVoxel(prob_grasp_planner::GenInfVoxel::Request& req,
                        prob_grasp_planner::GenInfVoxel::Response& res); 

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
};

#endif

