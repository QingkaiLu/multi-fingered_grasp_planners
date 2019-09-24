#include "gen_inference_voxel.h"
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <limits> 
//#include <sensor_msg/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
//#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>

// TODO: update the voxel generation method.
GenInferenceVoxel::voxel_grid GenInferenceVoxel::genVoxelFromPcd(const sensor_msgs::PointCloud2 &scene_cloud,
                                    const cv::Point3i &palm_voxel_dim,
                                    const cv::Point3f &palm_voxel_size,
                                    const std::string &object_frame_id)
{
    tf::TransformListener listener;
    tf::StampedTransform transform;
    try
    {
        listener.waitForTransform(object_frame_id, scene_cloud.header.frame_id, ros::Time(0), ros::Duration(10.0));
        listener.lookupTransform(object_frame_id, scene_cloud.header.frame_id, ros::Time(0), transform);
        //tf_listener.lookupTransform (target_frame, in.header.frame_id, in.header.stamp, transform);
    }
    catch (tf::TransformException e){
        ROS_ERROR("%s", e.what());
    }
    catch (tf::LookupException &e)
    {
        ROS_ERROR ("%s", e.what ());
    }
    catch (tf::ExtrapolationException &e)
    {
        ROS_ERROR ("%s", e.what ());
    }

    sensor_msgs::PointCloud2 out_cloud_msg;
    pcl_ros::transformPointCloud(object_frame_id, transform, scene_cloud, out_cloud_msg);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr trans_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromROSMsg(out_cloud_msg, *trans_cloud);

    voxel_grid palm_voxel_grid;

    std::vector<std::vector<std::vector<int> > > palm_voxel_grid_bi(palm_voxel_dim.x, 
                                                std::vector<std::vector<int> >(palm_voxel_dim.y, 
                                                std::vector<int>(palm_voxel_dim.z, 0)));
    PointCloudXYZRGB::iterator iter_point;

    cv::Point3d palm_voxel_min_loc;
    palm_voxel_min_loc.x = - palm_voxel_dim.x / 2 * palm_voxel_size.x;
    palm_voxel_min_loc.y = - palm_voxel_dim.y / 2 * palm_voxel_size.y;
    palm_voxel_min_loc.z = - palm_voxel_dim.z / 2 * palm_voxel_size.z;
    for(iter_point = trans_cloud->points.begin(); iter_point < trans_cloud->points.end(); iter_point++)
    {
        if (std::isnan(iter_point->x) || std::isnan(iter_point->y) || std::isnan(iter_point->z))
            continue;
        cv::Point3i voxel_loc; 
        voxel_loc.x = int((iter_point->x - palm_voxel_min_loc.x) / palm_voxel_size.x);
        voxel_loc.y = int((iter_point->y - palm_voxel_min_loc.y) / palm_voxel_size.y);
        voxel_loc.z = int((iter_point->z - palm_voxel_min_loc.z) / palm_voxel_size.z);
        if(voxel_loc.x >= 0 && voxel_loc.x < palm_voxel_dim.x &&
           voxel_loc.y >= 0 && voxel_loc.y < palm_voxel_dim.y &&
           voxel_loc.z >= 0 && voxel_loc.z < palm_voxel_dim.z)
            if(palm_voxel_grid_bi[voxel_loc.x][voxel_loc.y][voxel_loc.z] != 1)
            {
                palm_voxel_grid_bi[voxel_loc.x][voxel_loc.y][voxel_loc.z] = 1;
                palm_voxel_grid.push_back(voxel_loc);
            }
    }
  
    return palm_voxel_grid; 
}

bool GenInferenceVoxel::genVoxel(prob_grasp_planner::GenInfVoxel::Request& req,
                prob_grasp_planner::GenInfVoxel::Response& res)
{
    cv::Point3i palm_voxel_dim(req.voxel_dim[0], req.voxel_dim[1], req.voxel_dim[2]);
    cv::Point3f palm_voxel_size(req.voxel_size[0], req.voxel_size[1], req.voxel_size[2]);
    voxel_grid palm_voxel_grid = genVoxelFromPcd(req.scene_cloud, palm_voxel_dim, palm_voxel_size, req.object_frame_id);

    unsigned int voxels_num = palm_voxel_grid.size();
    std::vector<int> palm_voxel_grid_1d(3 * voxels_num, -1); 
    // Convert voxel_grid to 1d array.
    for (unsigned int i = 0; i < voxels_num; ++i)
    {
        palm_voxel_grid_1d[3 * i]  = palm_voxel_grid[i].x; 
        palm_voxel_grid_1d[3 * i + 1]  = palm_voxel_grid[i].y; 
        palm_voxel_grid_1d[3 * i + 2]  = palm_voxel_grid[i].z; 
    }
    std::cout << "filled_voxel_num: " << voxels_num << std::endl;
    res.voxel_grid = palm_voxel_grid_1d;
    return true;
}

void GenInferenceVoxel::publishPointcloud(const PointCloudXYZRGB &pointcloud, 
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

void GenInferenceVoxel::publishPointcloud(const PointCloudXYZRGB &pointcloud, 
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
    ros::init(argc, argv, "gen_inf_voxel_server");
    ros::NodeHandle n;

    GenInferenceVoxel gen_inf_voxel;    
    ros::ServiceServer service = n.advertiseService("/gen_inference_voxel", &GenInferenceVoxel::genVoxel, 
                                                    &gen_inf_voxel);
    ROS_INFO("Service gen_inference_voxel:");
    ROS_INFO("Ready to generate voxel for inference.");

    ros::spin();

    return 0;
}

