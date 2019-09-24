#include "gen_voxel_from_pcd.h"
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <limits> 
//#include <sensor_msg/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.h>


GenVoxelFromPcd::GenVoxelFromPcd(std::string cam_frame_id): camera_frame_id(cam_frame_id){}


GenVoxelFromPcd::voxel_grid GenVoxelFromPcd::genVoxelFromPcd(const std::string &pcd_file,
                                    const cv::Point3i &voxel_dim,
                                    const cv::Point3f &voxel_size)
{
    std::cout << pcd_file << std::endl;
    PointCloudXYZRGB::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::io::loadPCDFile(pcd_file, *cloud);
    cloud->header.frame_id = camera_frame_id;

    PointCloudXYZRGB::Ptr trans_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    tf::TransformListener listener;
   
    //listener.waitForTransform("object_pose", "blensor_camera", ros::Time(0), ros::Duration(10.0) );
    listener.waitForTransform("object_pose", camera_frame_id, ros::Time(0), ros::Duration(10.0) );
    pcl_ros::transformPointCloud("object_pose", *cloud, *trans_cloud, listener);
    trans_cloud->header.frame_id = "object_pose";
    //publishPointcloud(*cloud, "cloud", *trans_cloud, "cloud_object");

    voxel_grid voxel_grid;
    std::vector<std::vector<std::vector<int> > > voxel_grid_bi(voxel_dim.x, 
                                                std::vector<std::vector<int> >(voxel_dim.y, 
                                                std::vector<int>(voxel_dim.z, 0)));
    PointCloudXYZRGB::iterator iter_point;

    cv::Point3d voxel_min_loc;
    voxel_min_loc.x = - voxel_dim.x / 2 * voxel_size.x;
    voxel_min_loc.y = - voxel_dim.y / 2 * voxel_size.y;
    voxel_min_loc.z = - voxel_dim.z / 2 * voxel_size.z;
    for(iter_point = trans_cloud->points.begin(); iter_point < trans_cloud->points.end(); iter_point++)
    {
        if (std::isnan(iter_point->x) || std::isnan(iter_point->y) || std::isnan(iter_point->z))
            continue;
        cv::Point3i voxel_loc; 
        voxel_loc.x = int((iter_point->x - voxel_min_loc.x) / voxel_size.x);
        voxel_loc.y = int((iter_point->y - voxel_min_loc.y) / voxel_size.y);
        voxel_loc.z = int((iter_point->z - voxel_min_loc.z) / voxel_size.z);
        if(voxel_loc.x >= 0 && voxel_loc.x < voxel_dim.x &&
           voxel_loc.y >= 0 && voxel_loc.y < voxel_dim.y &&
           voxel_loc.z >= 0 && voxel_loc.z < voxel_dim.z)
            if(voxel_grid_bi[voxel_loc.x][voxel_loc.y][voxel_loc.z] != 1)
            {
                voxel_grid_bi[voxel_loc.x][voxel_loc.y][voxel_loc.z] = 1;
                voxel_grid.push_back(voxel_loc);
            }
    }
  
    return voxel_grid; 
}


GenVoxelFromPcd::voxel_grid GenVoxelFromPcd::genVoxelFromPcd(
                                    const sensor_msgs::PointCloud2& seg_obj_cloud,
                                    const cv::Point3i &voxel_dim,
                                    const cv::Point3f &voxel_size)
{
    PointCloudXYZRGB::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromROSMsg(seg_obj_cloud, *cloud);
    cloud->header.frame_id = "world";

    PointCloudXYZRGB::Ptr trans_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    tf::TransformListener listener;
   
    // std::cout << cloud->header.stamp << std::endl;
    // The transformPointCloud using tf has the error that the pointcloud stamp doesn't 
    // match the current ros time stamp. There are two ways to solve this problem: 1.
    // set the stamp of the point cloud to be 0 (I don't know why this solves the issue, 
    // but it works); 2. get the transformation matrix from lookupTransform first, 
    // then transformPointCloud using the trasformation matrix (gen_inference_voxel uses 
    // this method).
    cloud->header.stamp = 0;
    // std::cout << ros::Time::now() << std::endl;
    // std::cout << cloud->header.stamp << std::endl;
    listener.waitForTransform("object_pose", "world", ros::Time(0), ros::Duration(10.0) );
    pcl_ros::transformPointCloud("object_pose", *cloud, *trans_cloud, listener);
    trans_cloud->header.frame_id = "object_pose";
    publishPointcloud(*cloud, "cloud", *trans_cloud, "cloud_object");

    voxel_grid voxel_grid;
    std::vector<std::vector<std::vector<int> > > voxel_grid_bi(voxel_dim.x, 
                                                std::vector<std::vector<int> >(voxel_dim.y, 
                                                std::vector<int>(voxel_dim.z, 0)));
    PointCloudXYZRGB::iterator iter_point;

    cv::Point3d voxel_min_loc;
    voxel_min_loc.x = - voxel_dim.x / 2 * voxel_size.x;
    voxel_min_loc.y = - voxel_dim.y / 2 * voxel_size.y;
    voxel_min_loc.z = - voxel_dim.z / 2 * voxel_size.z;
    for(iter_point = trans_cloud->points.begin(); iter_point < trans_cloud->points.end(); iter_point++)
    {
        if (std::isnan(iter_point->x) || std::isnan(iter_point->y) || std::isnan(iter_point->z))
            continue;
        cv::Point3i voxel_loc; 
        voxel_loc.x = int((iter_point->x - voxel_min_loc.x) / voxel_size.x);
        voxel_loc.y = int((iter_point->y - voxel_min_loc.y) / voxel_size.y);
        voxel_loc.z = int((iter_point->z - voxel_min_loc.z) / voxel_size.z);
        if(voxel_loc.x >= 0 && voxel_loc.x < voxel_dim.x &&
           voxel_loc.y >= 0 && voxel_loc.y < voxel_dim.y &&
           voxel_loc.z >= 0 && voxel_loc.z < voxel_dim.z)
            if(voxel_grid_bi[voxel_loc.x][voxel_loc.y][voxel_loc.z] != 1)
            {
                voxel_grid_bi[voxel_loc.x][voxel_loc.y][voxel_loc.z] = 1;
                voxel_grid.push_back(voxel_loc);
            }
    }
  
    return voxel_grid; 
}


bool GenVoxelFromPcd::genVoxel(prob_grasp_planner::GenGraspVoxel::Request& req,
                prob_grasp_planner::GenGraspVoxel::Response& res)
{
    cv::Point3i voxel_dim(req.voxel_dim[0], req.voxel_dim[1], req.voxel_dim[2]);
    cv::Point3f voxel_size(req.voxel_size[0], req.voxel_size[1], req.voxel_size[2]);
    // voxel_grid voxel_grid = genVoxelFromPcd(req.pcd_file_path, voxel_dim, voxel_size);
    voxel_grid voxel_grid = genVoxelFromPcd(req.seg_obj_cloud, voxel_dim, voxel_size); 

    unsigned int voxels_num = voxel_grid.size();
    std::vector<int> voxel_grid_1d(3 * voxels_num, -1); 
    // Convert voxel_grid to 1d array.
    for (unsigned int i = 0; i < voxels_num; ++i)
    {
        // Translate the voxel grid from partial frame to full view frame
        voxel_grid_1d[3 * i]  = voxel_grid[i].x + req.voxel_trans_dim[0]; 
        voxel_grid_1d[3 * i + 1]  = voxel_grid[i].y + req.voxel_trans_dim[1]; 
        voxel_grid_1d[3 * i + 2]  = voxel_grid[i].z + req.voxel_trans_dim[2]; 
    }
    std::cout << "filled_voxel_num: " << voxels_num << std::endl;
    res.voxel_grid = voxel_grid_1d;
    return true;
}

void GenVoxelFromPcd::publishPointcloud(const PointCloudXYZRGB &pointcloud, 
                                        const std::string &topic_name)
{
    ros::NodeHandle nh;
    uint32_t queue_size = 1; 
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(pointcloud, cloud_msg);
    ros::Publisher pub = nh.advertise<sensor_msgs::PointCloud2>(topic_name, queue_size);
    ros::Rate loop_rate(10);
    while (ros::ok())
    //for(int i=0; i < 10; ++i)
    {
        pub.publish(cloud_msg);
        ros::spinOnce();
        loop_rate.sleep();
    }
}

void GenVoxelFromPcd::publishPointcloud(const PointCloudXYZRGB &pointcloud, 
                                        const std::string &topic_name, 
                                        const PointCloudXYZRGB &pointcloud_2, 
                                        const std::string &topic_2_name)
{
    ros::NodeHandle nh;
    uint32_t queue_size = 1; 
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(pointcloud, cloud_msg);
    ros::Publisher pub = nh.advertise<sensor_msgs::PointCloud2>(topic_name, queue_size);
    sensor_msgs::PointCloud2 cloud_msg_2;
    pcl::toROSMsg(pointcloud_2, cloud_msg_2);
    ros::Publisher pub2 = nh.advertise<sensor_msgs::PointCloud2>(topic_2_name, queue_size);
    ros::Rate loop_rate(10);
    //while (ros::ok())
    for(int i = 0; i < 20; ++i)
    {
        pub.publish(cloud_msg);
        pub2.publish(cloud_msg_2);
        ros::spinOnce();
        loop_rate.sleep();
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "gen_voxel_server");
    //ros::NodeHandle n;
    ros::NodeHandle n("~");

    std::string camera_frame_id;
    //n.param<std::string>("camera_frame_id", camera_frame_id, "test");
    n.getParam("camera_frame_id", camera_frame_id);

    GenVoxelFromPcd gen_voxel_from_pcd(camera_frame_id); 
    ros::ServiceServer service = n.advertiseService("/gen_voxel_from_pcd", &GenVoxelFromPcd::genVoxel, 
                                                    &gen_voxel_from_pcd);
    ROS_INFO("Service gen_voxel_from_pcd:");
    ROS_INFO("Ready to generate voxel from pcd.");

    ros::spin();

    return 0;
}

