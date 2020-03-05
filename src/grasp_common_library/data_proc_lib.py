#!/usr/bin/env python
import roslib
roslib.load_manifest('prob_grasp_planner')
import rospy
from geometry_msgs.msg import Pose, Quaternion, PoseStamped, PointStamped
from prob_grasp_planner.srv import *
from prob_grasp_planner.msg import *
import numpy as np
import time
import cv2
import h5py
import roslib.packages as rp
import sys
pkg_path = rp.get_pkg_dir('prob_grasp_planner')
sys.path.append(pkg_path + '/src/grasp_common_library')
import show_voxel
import tf
import os


class DataProcLib:
    '''
    Class to process active learning grasp queries. Grasp data includes grasp object voxel,
    grasp configuration and grasp success label.
    '''
    def __init__(self):
        # rospy.init_node('data_proc_lib') 

        #self.voxel_grid_dim = [20, 20, 20]
        #self.voxel_size = [0.01, 0.01, 0.01]

        self.voxel_grid_dim = np.array([26, 26, 26])
        self.voxel_grid_full_dim = np.array([32, 32, 32])
        # Compute the translation to move the partial voxel grid into 
        # the full voxel grid frame.
        self.voxel_trans_dim = (self.voxel_grid_full_dim - self.voxel_grid_dim) // 2

        # 3 dof location of the palm + 3 dof orientation of the palm (4 parameters for quaternion, 
        # 3 for Euler angles) + 1st two joints of the thumb + 1st joint of other three fingers
        # Other joint angles are fixed for grasp preshape inference.
        self.palm_loc_dof_dim = 3
        self.palm_dof_dim = 6

        self.object_frame_id = 'object_pose'
        self.world_frame_id = 'world'

        self.tf_br = tf.TransformBroadcaster()
        self.listener = tf.TransformListener()
 

    def trans_pose_to_obj_tf(self, pose_stamp, return_array=True):
        '''
        Transform a pose into object frame.

        Args:
            palm_pose_stamp
        Returns:
            palm pose array in object frame.
        '''
        self.listener.waitForTransform(pose_stamp.header.frame_id, self.object_frame_id, 
                                        rospy.Time(), rospy.Duration(4.0))
        try:
            self.listener.waitForTransform(pose_stamp.header.frame_id, 
                            self.object_frame_id, rospy.Time.now(), rospy.Duration(4.0))
            pose_object = self.listener.transformPose(self.object_frame_id, pose_stamp)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logerr('Could not find transformation!')
            return None

        vis_tf = False
        if vis_tf:
            j = 0
            rate = rospy.Rate(100)
            # while not rospy.is_shutdown():
            while j < 10:
                self.tf_br.sendTransform((pose_object.pose.position.x, pose_object.pose.position.y, 
                        pose_object.pose.position.z),
                        (pose_object.pose.orientation.x, pose_object.pose.orientation.y, 
                        pose_object.pose.orientation.z, pose_object.pose.orientation.w),
                        rospy.Time.now(), 'palm_pose_object', pose_object.header.frame_id)
                j += 1
                rate.sleep()
        
        if return_array:
            pose_quaternion = (pose_object.pose.orientation.x, pose_object.pose.orientation.y, 
                    pose_object.pose.orientation.z, pose_object.pose.orientation.w) 
            pose_euler = tf.transformations.euler_from_quaternion(pose_quaternion)
            pose_object_array = [pose_object.pose.position.x, pose_object.pose.position.y,
                    pose_object.pose.position.z, pose_euler[0], pose_euler[1], pose_euler[2]]
            return pose_object_array
        return pose_object


    def trans_pose_to_world_tf(self, pose_stamp, return_array=True):
        '''
        Transform a pose into world frame.

        Args:
            palm_pose_stamp
        Returns:
            palm pose array in world frame.
        '''
        self.listener.waitForTransform(pose_stamp.header.frame_id, self.world_frame_id, 
                                        rospy.Time(), rospy.Duration(4.0))
        try:
            self.listener.waitForTransform(pose_stamp.header.frame_id, 
                            self.world_frame_id, rospy.Time.now(), rospy.Duration(4.0))
            pose_world = self.listener.transformPose(self.world_frame_id, pose_stamp)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logerr('Could not find transformation!')
            return None

        vis_tf = False
        if vis_tf:
            j = 0
            rate = rospy.Rate(100)
            # while not rospy.is_shutdown():
            while j < 10:
                self.tf_br.sendTransform((pose_world.pose.position.x, pose_world.pose.position.y, 
                        pose_world.pose.position.z),
                        (pose_world.pose.orientation.x, pose_world.pose.orientation.y, 
                        pose_world.pose.orientation.z, pose_world.pose.orientation.w),
                        rospy.Time.now(), 'palm_pose_world', pose_world.header.frame_id)
                j += 1
                rate.sleep()
        
        if return_array:
            pose_quaternion = (pose_world.pose.orientation.x, pose_world.pose.orientation.y, 
                    pose_world.pose.orientation.z, pose_world.pose.orientation.w) 
            pose_euler = tf.transformations.euler_from_quaternion(pose_quaternion)
            pose_world_array = [pose_world.pose.position.x, pose_world.pose.position.y,
                    pose_world.pose.position.z, pose_euler[0], pose_euler[1], pose_euler[2]]
            return pose_world_array
        return pose_world


    def update_palm_pose_client(self, palm_pose):
        '''
        Client to update the palm pose tf.
        '''
        rospy.loginfo('Waiting for service update_grasp_palm_pose.')
        rospy.wait_for_service('update_grasp_palm_pose')
        rospy.loginfo('Calling service update_grasp_palm_pose.')
        try:
            update_palm_pose_proxy = rospy.ServiceProxy('update_grasp_palm_pose', UpdatePalmPose)
            update_palm_pose_request = UpdatePalmPoseRequest()
            update_palm_pose_request.palm_pose = palm_pose
            update_palm_pose_response = update_palm_pose_proxy(update_palm_pose_request) 
        except rospy.ServiceException, e:
            rospy.loginfo('Service update_grasp_palm_pose call failed: %s'%e)
        rospy.loginfo('Service update_grasp_palm_pose is executed.')


    def update_object_pose_client(self, object_pose):
        '''
        Client to update the object pose tf.
        '''
        rospy.loginfo('Waiting for service update_grasp_object_pose.')
        rospy.wait_for_service('update_grasp_object_pose')
        rospy.loginfo('Calling service update_grasp_object_pose.')
        try:
            update_object_pose_proxy = rospy.ServiceProxy('update_grasp_object_pose', UpdateObjectPose)
            update_object_pose_request = UpdateObjectPoseRequest()
            update_object_pose_request.object_pose_world = object_pose
            update_object_pose_response = update_object_pose_proxy(update_object_pose_request) 
        except rospy.ServiceException, e:
            rospy.loginfo('Service update_grasp_object_pose call failed: %s'%e)
        rospy.loginfo('Service update_grasp_object_pose is executed.')


    def voxel_gen_client_file(self, pcd_file_path):
        '''
        Service client to get voxel from pointcloud in palm frame.
        The palm and camera frame need to be publihsed as ROS tf 
        to call the service to generate the transformed pointcloud.
        
        Args:
            pcd_file_path.

        Returns:
            Voxelgrid.
        '''
        rospy.loginfo('Waiting for service gen_voxel_from_pcd.')
        rospy.wait_for_service('gen_voxel_from_pcd')
        rospy.loginfo('Calling service gen_voxel_from_pcd.')
        try:
            gen_voxel_proxy = rospy.ServiceProxy('gen_voxel_from_pcd', GenGraspVoxel)
            gen_voxel_request = GenGraspVoxelRequest()
            gen_voxel_request.pcd_file_path = pcd_file_path
            gen_voxel_request.voxel_dim = self.voxel_grid_dim
            gen_voxel_request.voxel_size = self.voxel_size

            gen_voxel_response = gen_voxel_proxy(gen_voxel_request) 
            return np.reshape(gen_voxel_response.voxel_grid, 
                                [len(gen_voxel_response.voxel_grid) / 3, 3])
        except rospy.ServiceException, e:
            rospy.loginfo('Service gen_voxel_from_pcd call failed: %s'%e)
        rospy.loginfo('Service gen_voxel_from_pcd is executed.')


    def voxel_gen_client(self, seg_object):
        '''
        Service client to get voxel from pointcloud in palm frame.
        The palm and camera frame need to be publihsed as ROS tf 
        to call the service to generate the transformed pointcloud.
        
        Args:
            object cloud segmentation.

        Returns:
            Voxelgrid.
        '''
        rospy.loginfo('Waiting for service gen_voxel_from_pcd.')
        rospy.wait_for_service('gen_voxel_from_pcd')
        rospy.loginfo('Calling service gen_voxel_from_pcd.')
        try:
            gen_voxel_proxy = rospy.ServiceProxy('gen_voxel_from_pcd', GenGraspVoxel)
            gen_voxel_request = GenGraspVoxelRequest()
            max_dim = np.max([seg_object.width, seg_object.height, seg_object.depth])
            voxel_size = max_dim / self.voxel_grid_dim

            gen_voxel_request.seg_obj_cloud = seg_object.cloud
            gen_voxel_request.voxel_dim = self.voxel_grid_dim
            gen_voxel_request.voxel_trans_dim = self.voxel_trans_dim
            gen_voxel_request.voxel_size = voxel_size

            gen_voxel_response = gen_voxel_proxy(gen_voxel_request) 
            return np.reshape(gen_voxel_response.voxel_grid, 
                        [len(gen_voxel_response.voxel_grid) / 3, 3]), \
                    voxel_size[0], self.voxel_grid_full_dim
        except rospy.ServiceException, e:
            rospy.loginfo('Service gen_voxel_from_pcd call failed: %s'%e)
        rospy.loginfo('Service gen_voxel_from_pcd is executed.')


    def lookup_transform(self, target_frame, source_frame):
        trans = None
        rot = None
        while rot is None: 
            try:
                # trans, rot = self.listener.lookupTransform(
                #                                 target_frame, source_frame, rospy.Time.now())

                common_time = self.listener.getLatestCommonTime(target_frame, source_frame)
                trans, rot = self.listener.lookupTransform(
                                                target_frame, source_frame, common_time)
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rospy.logerr('Could not find transformation from source to target frame!')
            
        rot_mat = tf.transformations.quaternion_matrix(rot)
        translation_mat = tf.transformations.translation_matrix(trans)
        transform_mat = np.copy(translation_mat)
        transform_mat[:3, :3] = rot_mat[:3, :3]

        return transform_mat




