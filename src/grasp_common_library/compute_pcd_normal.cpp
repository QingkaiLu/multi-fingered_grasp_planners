#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <iostream>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>

#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/console/parse.h>

#include "ros/ros.h"
#include <sensor_msgs/Image.h>
#include <string>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>

#include <prob_grasp_planner/GetPcdNormal.h>

using namespace cv;
using namespace std;


bool computePcdNormal(prob_grasp_planner::GetPcdNormal::Request &req,
                prob_grasp_planner::GetPcdNormal::Response &res)
{
    std::cout << req.pcd_file_path << std::endl;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::io::loadPCDFile(req.pcd_file_path, *cloud);

    pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> ne;
    ne.setInputCloud(cloud);

    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
    ne.setSearchMethod(tree);

    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals_ptr(new pcl::PointCloud<pcl::Normal>);

    int normals_k_neighbors = 50;
    ne.setKSearch(normals_k_neighbors);

    ne.compute(*cloud_normals_ptr);

    pcl::toROSMsg(*cloud_normals_ptr, res.pcd_normal);
    pcl::toROSMsg(*cloud, res.point_cloud);
   
    return true;
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "compute_pcd_normal_server");
    ros::NodeHandle n;
    
    ros::ServiceServer service = n.advertiseService("/compute_pcd_normal", computePcdNormal);
    ROS_INFO("Service compute_pcd_normal_server:");
    ROS_INFO("Ready to compute pcd normal.");

    ros::spin();

    return 0;
}
