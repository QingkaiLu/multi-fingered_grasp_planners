#!/usr/bin/env python

import roslib; roslib.load_manifest('prob_grasp_planner')
import rospy
from prob_grasp_planner.srv import *
from prob_grasp_planner.msg import *
#import tf
from sensor_msgs.msg import Image, JointState, CameraInfo
from geometry_msgs.msg import Pose, Quaternion, PoseStamped
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from gen_rgbd_images import GenRgbdImage

import roslib.packages as rp
from compute_finger_tip_location import ComputeFingerTipPose
from grasp_rgbd_patches_net import GraspRgbdPatchesNet
from grasp_rgbd_config_net import GraspRgbdConfigNet
from grasp_rgbd_inf import GraspInf
import h5py
import time

class RunGraspInference:
    def __init__(self):
        self.compute_finger_tip_loc = ComputeFingerTipPose()
        #self.rgbd_patches_save_path = rospy.get_param('~data_recording_path', 
        #        '/media/kai/multi_finger_grasp_data/')
        #rgbd_patches_save_path = '/data_space/data_kai/multi_finger_sim_data/v3/'
        rgbd_patches_save_path = '/media/kai/multi_finger_sim_data_complete_v3/'
        #self.rgbd_patches_file_name = rgbd_patches_save_path + 'grasps_rgbd_patches.h5'
        self.rgbd_file_name = rgbd_patches_save_path + 'grasp_rgbd.h5'
        self.grasp_patches_file_name = rgbd_patches_save_path + 'grasp_patches.h5'
        self.use_hd = True
        self.gen_rgbd = GenRgbdImage(rgbd_patches_save_path) 
        self.grasp_rgbd_inf = None

    def run_grasp_inference(self, grasp_id):
        '''
            Run grasp inference. 
        '''
        self.grasp_rgbd_inf = GraspInf(config_net=False, use_hd=self.use_hd)
        rgbd_file = h5py.File(self.rgbd_file_name, 'r')
        grasp_sample_id = 'grasp_' + str(grasp_id)
        rgbd_key = grasp_sample_id + '_rgbd'
        rgbd_image = np.copy(rgbd_file[rgbd_key])
        #print rgbd_image
        #palm_pcd_pose_key = grasp_sample_id + '_preshape_palm_pcd_pose'
        true_palm_pcd_pose_key = grasp_sample_id + '_true_preshape_palm_pcd_pose'
        true_palm_pcd_pose = PoseStamped()
        [true_palm_pcd_pose.pose.position.x, true_palm_pcd_pose.pose.position.y,
        true_palm_pcd_pose.pose.position.z, true_palm_pcd_pose.pose.orientation.x,
        true_palm_pcd_pose.pose.orientation.y, true_palm_pcd_pose.pose.orientation.z,
        true_palm_pcd_pose.pose.orientation.w] = rgbd_file[true_palm_pcd_pose_key]
        true_preshape_js_position_key = grasp_sample_id + '_true_preshape_js_position'
        true_preshape_js_position = JointState() 
        true_preshape_js_position.position = np.copy(rgbd_file[true_preshape_js_position_key])
        rgbd_file.close()

        #grasp_patches_file = h5py.File(self.grasp_patches_file_name, 'r')
        #grasp_patch_key = grasp_sample_id + '_grasp_patch'
        #rgbd_image = grasp_patches_file[grasp_patch_key]
        ##grasp_label_key = grasp_sample_id + '_grasp_label'
        #grasp_patches_file.close()

        print true_palm_pcd_pose
        print true_preshape_js_position

        #self.compute_finger_tip_loc.set_up_input(true_palm_pcd_pose, true_preshape_js_position)
        #self.compute_finger_tip_loc.proj_finger_palm_locs_to_img()
        #palm_image_loc = self.compute_finger_tip_loc.palm_image_loc
        #finger_tip_image_locs = self.compute_finger_tip_loc.finger_tip_image_locs 
        #rospy.loginfo(palm_image_loc)
        #rospy.loginfo(finger_tip_image_locs)
        #palm_patch, finger_tip_patches = self.gen_rgbd.get_finger_palm_patches(rgbd_image, palm_image_loc, finger_tip_image_locs, 
        #                                        object_id=req.object_id, grasp_id=req.grasp_id, 
        #                                        object_name=req.object_name, grasp_label=req.grasp_success_label,
        #                                        save=True)

        #grasp_rgbd_net = GraspRgbdNet()
        #grasp_rgbd_net.init_net_inf()
        #d_prob_d_palm, d_prob_d_fingers, suc_prob = \
        #        grasp_rgbd_net.get_rgbd_gradients(palm_patch, finger_tip_patches)
        #d_prob_d_fingers = np.array(d_prob_d_fingers)
        #print d_prob_d_palm.shape, d_prob_d_fingers.shape, suc_prob.shape
        #print np.mean(d_prob_d_palm), np.mean(d_prob_d_fingers), np.mean(suc_prob)

        init_hand_pcd_config = HandConfig()
        init_hand_pcd_config.palm_pose = true_palm_pcd_pose
        init_hand_pcd_config.hand_joint_state = true_preshape_js_position
        config_opt, suc_prob, suc_prob_init = \
                self.grasp_rgbd_inf.gradient_descent_inf(rgbd_image, init_hand_pcd_config, 
                                            save_grad_to_log=True, object_id=0, grasp_id=grasp_id) 
        
        #config_opt, suc_prob, suc_prob_init = \
        #        self.grasp_rgbd_inf.quasi_newton_lbfgs_inf(rgbd_image, req.init_hand_pcd_config) 

        print init_hand_pcd_config
        print config_opt, suc_prob, suc_prob_init 

        #To do: plot lines from palm center to finger tips and save the rgbd for the final inference 
        #result; do the same thing for the initialization.

        return

    def run_grasp_config_inference(self, grasp_id):
        '''
            Run grasp inference usig the grasp patch + configuration CNN. 
        '''
        self.grasp_rgbd_inf = GraspInf(config_net=True, use_hd=self.use_hd)
        rgbd_file = h5py.File(self.rgbd_file_name, 'r')
        grasp_sample_id = 'grasp_' + str(grasp_id)
        #rgbd_key = grasp_sample_id + '_rgbd'
        #rgbd_image = np.copy(rgbd_file[rgbd_key])
        #print rgbd_image
        #palm_pcd_pose_key = grasp_sample_id + '_preshape_palm_pcd_pose'
        true_palm_pcd_pose_key = grasp_sample_id + '_true_preshape_palm_pcd_pose'
        true_palm_pcd_pose = PoseStamped()
        [true_palm_pcd_pose.pose.position.x, true_palm_pcd_pose.pose.position.y,
        true_palm_pcd_pose.pose.position.z, true_palm_pcd_pose.pose.orientation.x,
        true_palm_pcd_pose.pose.orientation.y, true_palm_pcd_pose.pose.orientation.z,
        true_palm_pcd_pose.pose.orientation.w] = rgbd_file[true_palm_pcd_pose_key]
        true_preshape_js_position_key = grasp_sample_id + '_true_preshape_js_position'
        true_preshape_js_position = JointState() 
        true_preshape_js_position.position = np.copy(rgbd_file[true_preshape_js_position_key])
        rgbd_file.close()

        grasp_patches_file = h5py.File(self.grasp_patches_file_name, 'r')
        grasp_patch_key = grasp_sample_id + '_grasp_patch'
        grasp_patch = np.copy(grasp_patches_file[grasp_patch_key])
        #grasp_label_key = grasp_sample_id + '_grasp_label'
        #grasp_label = grasp_patches_file[grasp_label_key][()]
        grasp_config_key = grasp_sample_id + '_preshape_true_config'
        grasp_config = np.copy(grasp_patches_file[grasp_config_key])
        grasp_patches_file.close()

        print true_palm_pcd_pose
        print true_preshape_js_position

        #grasp_rgbd_config_net = GraspRgbdConfigNet()
        #grasp_rgbd_config_net.init_net_inf()
        #d_prob_d_config, suc_prob = \
        #        grasp_rgbd_config_net.get_config_gradients(grasp_patch, grasp_config)
        #d_prob_d_config = np.array(d_prob_d_config)
        #print d_prob_d_config
        #print suc_prob

        init_hand_pcd_config = HandConfig()
        init_hand_pcd_config.palm_pose = true_palm_pcd_pose
        init_hand_pcd_config.hand_joint_state = true_preshape_js_position
        config_opt, suc_prob, suc_prob_init = \
                self.grasp_rgbd_inf.gradient_descent_inf(grasp_patch, init_hand_pcd_config, 
                                            save_grad_to_log=True, object_id=0, grasp_id=grasp_id) 
        

        #print init_hand_pcd_config
        #print config_opt, suc_prob, suc_prob_init 

        #To do: plot lines from palm center to finger tips and save the rgbd for the final inference 
        #result; do the same thing for the initialization.

        return

if __name__ == '__main__':
    grasp_inference = RunGraspInference()
    #grasp_inference.run_grasp_inference(grasp_id=20)
    grasp_inference.run_grasp_config_inference(grasp_id=20)

