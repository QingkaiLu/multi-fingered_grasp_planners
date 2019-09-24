#include <ros/ros.h>
#include <pcl/common/common.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.h>
#include "point_cloud_segmentation/SegmentGraspObjectBlensor.h"

void SegmentObjClient(const std::string &pcd_file)
{
    std::cout << pcd_file << std::endl;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::io::loadPCDFile(pcd_file, *cloud);

    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*cloud, cloud_msg);

    ros::NodeHandle n;
    ros::ServiceClient client = n.serviceClient<point_cloud_segmentation::
                                SegmentGraspObjectBlensor>("object_segmenter_blensor");
    point_cloud_segmentation::SegmentGraspObjectBlensor srv;
    srv.request.scene_cloud = cloud_msg;
    if (client.call(srv))
    {
        ROS_INFO("Object found: %d", srv.response.object_found);
    }
    else
    {
        ROS_ERROR("Failed to call service object_segmenter_blensor.");
    }
    
    return;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "object_segmenter_blensor");
    std::string pcd_file_path = "/home/kai/Workspace/test_data/object_13_coffee_mate_french_vanilla_grasp_200000.pcd";
    //std::string pcd_file_path = "/home/kai/Workspace/test_data/object_29_pringles_bbq_grasp_200000.pcd";
    SegmentObjClient(pcd_file_path);
    return 0;
}
