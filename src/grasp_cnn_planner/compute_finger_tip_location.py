#!/usr/bin/env python

import rospy
import roslib.packages as rp
import sys
import hand_model
import numpy as np
import PyKDL
from geometry_msgs.msg import Pose, Quaternion, PoseStamped
from sensor_msgs.msg import JointState, CameraInfo
import tf

class ComputeFingerTipPose:
    def __init__(self):
        #self.listener = listener
        self.fingers_num = 4
        self.palm_pose = PoseStamped()
        self.hand_joint_state = JointState() 
        self.camera_info = CameraInfo()
        self.proj_mat = np.zeros((3, 3))
        self.kdlModel = hand_model.HandSimpleModel()
        self.finger_tip_camera_loc = np.zeros((self.fingers_num, 3))
        self.finger_tip_image_locs = np.zeros((self.fingers_num, 2))
        self.palm_image_loc = np.zeros(2)
  
    def set_up_input(self, palm_pose, hand_joint_state, camera_info=None, use_hd=True):
        '''
        Set up the input to do finger fk.
        Make sure the palm_pose_sd is in the correct point cloud frame 
        (e.g. kinect2 sd or hd).
        '''
        self.palm_pose = palm_pose
        self.palm_position = np.array([self.palm_pose.pose.position.x, self.palm_pose.pose.position.y, 
                                    self.palm_pose.pose.position.z])
        self.hand_joint_state = hand_joint_state
        if camera_info is not None:
            self.camera_info = camera_info
            camera_info_k_mat = self.camera_info.K
        else:
            # '/kinect2/hd/camera_info' relates to rgb image and hd point cloud.
            # '/kinect2/sd/camera_info' relates to depth image and sd point cloud.
            # The rgb info of sd cloud is downsampled from hd, the depth info of hd cloud id upsampled from sd.
            # camera info of '/kinect2/hd/camera_info'
            #D: [0.01279826536775383, 0.05700721513256569, -0.0025181724013412883, 0.001660633684394187, -0.09974551040196526]
            #K: [1063.6538241241747, 0.0, 954.2839107588829, 0.0, 1065.3220448775858, 529.5818992702947, 0.0, 0.0, 1.0]
            #R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            #P: [1063.6538241241747, 0.0, 954.2839107588829, 0.0, 0.0, 1065.3220448775858, 529.5818992702947, 0.0, 0.0, 0.0, 1.0, 0.0]
            # HD
            if use_hd:
                camera_info_k_mat = np.array([1063.6538241241747, 0.0, 954.2839107588829, 0.0, 1065.3220448775858, 
                                            529.5818992702947, 0.0, 0.0, 1.0])
            # SD
            else:
                camera_info_k_mat = np.array([368.18359375, 0.0, 252.43890380859375, 0.0, 368.18359375, 
                                            210.71270751953125, 0.0, 0.0, 1.0])
                
        self.proj_mat = np.reshape(camera_info_k_mat, (3, 3))
        self.joints_num_per_finger = len(hand_joint_state.position) / self.fingers_num
        #self.get_proj_matrix_from_camera_info()
        
    #def set_up_input_with_array(self, hand_config, camera_info=None):
    #    '''
    #    Set up the input using np array parameter hand_config to compute finger fk.
    #    hand_config: palm location + palm quaternion + finger joints angles 
    #    '''
    #    self.palm_pose = hand_config[:7]
    #    self.palm_position = hand_config[:3]
    #    self.hand_joint_state = hand_config[3:]
    #    if camera_info is not None:
    #        self.camera_info = camera_info
    #        camera_info_k_mat = self.camera_info.K
    #    else:
    #        # SD
    #        camera_info_k_mat = np.array([368.18359375, 0.0, 252.43890380859375, 0.0, 368.18359375, 
    #                                        210.71270751953125, 0.0, 0.0, 1.0])
    #    self.proj_mat = np.reshape(camera_info_k_mat, (3, 3))
    #    self.joints_num_per_finger = len(hand_joint_state.position) / self.fingers_num
    #    #self.get_proj_matrix_from_camera_info()


    def get_proj_matrix_from_camera_info(self):
        '''
            Get the project matrix from camera info.
            Use the k matrix since k and p matrices are the same.
        '''
        #for i in range(3):
        #    for k in range(3):
        #        self.proj_mat[i, k] = float(self.camera_info.K[i*3 + k])
        self.proj_mat = np.reshape(self.camera_info.K, (3, 3))

    def get_fingers_tip_pose(self):
        '''
        Get the pose of all finger tips.
        '''
        for i in xrange(self.fingers_num):
            joint_state_finger = self.hand_joint_state.position[self.joints_num_per_finger * i : 
                    self.joints_num_per_finger * (i + 1)]
            finger_tip_pose = self.compute_finger_tip_pose(np.array(joint_state_finger), i) 
            self.finger_tip_camera_loc[i, :] = np.array(finger_tip_pose)[:, 0]
            #self.finger_tip_camera_loc[i, :] = self.palm_position + finger_tip_pose[:3]
            #self.finger_tip_poses[i, :] = self.palm_position + finger_tip_pose 

    def compute_finger_tip_pose(self, u, EE_index):
        '''
        Get the pose of one finger tip in world frame.
        '''
        mat_finger_tip_to_palm = self.kdlModel.FK(u, EE_index)
        #listener = tf.TransformListener()
        #mat_palm_to_world = self.listener.fromTranslationRotation(
        #    (self.palm_pose.pose.position.x, self.palm_pose.pose.position.y, self.palm_pose.pose.position.z),
        #    (self.palm_pose.pose.orientation.x, self.palm_pose.pose.orientation.y,
        #     self.palm_pose.pose.orientation.z, self.palm_pose.pose.orientation.w)
        #)
        rot_palm_to_world = tf.transformations.quaternion_matrix((self.palm_pose.pose.orientation.x, self.palm_pose.pose.orientation.y,
             self.palm_pose.pose.orientation.z, self.palm_pose.pose.orientation.w))
        trans_palm_to_world = tf.transformations.translation_matrix((self.palm_pose.pose.position.x, self.palm_pose.pose.position.y, 
            self.palm_pose.pose.position.z))
        mat_palm_to_world = rot_palm_to_world + trans_palm_to_world
        mat_finger_tip_to_world = np.matmul(mat_palm_to_world, mat_finger_tip_to_palm)
        #pose_in_palm = np.zeros(6)
        #R = PyKDL.Rotation(T[0,0], T[0,1], T[0,2], T[1,0], T[1,1], 
        #                    T[1,2], T[2,0], T[2,1], T[2,2])
        #pose_in_palm[3:6] = R.GetRPY()
        #pose_in_palm[0:3] = T[0:3,3].ravel()
        #print T_finger_tip_to_palm
        #print T_palm_to_world
        #print T_finger_tip_to_world
        return mat_finger_tip_to_world[:3, 3] 

    def proj_finger_palm_locs_to_img(self):
        '''
        Project finger tips and palm center from 3d camera space to 2d image space.
        '''
        self.get_fingers_tip_pose()
        self.palm_image_loc = self.proj_point_from_camera_to_img(self.palm_position)
        for i in xrange(self.fingers_num):
            self.finger_tip_image_locs[i, :] = self.proj_point_from_camera_to_img(self.finger_tip_camera_loc[i, :])

    def proj_point_from_camera_to_img(self, point):
        '''
        Project one point from 3d camera space to 2d image space.
        '''
        #homo_point = self.proj_mat * point
        homo_point = np.matmul(self.proj_mat, point)
        # Convert projection matrix parameters from mm to meters.
        homo_point *= 0.001
        proj_x = int(homo_point[0] / homo_point[2])
        proj_y = int(homo_point[1] / homo_point[2])
        return np.array([proj_x, proj_y])


#if __name__ == '__main__':
#    compute_finger_tip_pose = ComputeFingerTipPose()


