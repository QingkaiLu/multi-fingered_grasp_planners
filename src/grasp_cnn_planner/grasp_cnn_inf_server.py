#!/usr/bin/env python

import roslib; roslib.load_manifest('prob_grasp_planner')
import rospy
#from prob_grasp_planner.msg import visual_info, hand_config
from prob_grasp_planner.srv import *
from prob_grasp_planner.msg import *
import tf
from sensor_msgs.msg import Image, JointState, CameraInfo
#import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import Pose, Quaternion, PoseStamped
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from gen_rgbd_real_kinect2 import GenRgbdRealKinect2
from gen_rgbd_gazebo_kinect import GenRgbdGazeboKinect
import roslib.packages as rp
import sys
sys.path.append(rp.get_pkg_dir('grasp_pipeline') 
        + '/src')
from compute_finger_tip_location import ComputeFingerTipPose
#from grasp_rgbd_net import GraspRgbdNet
#from grasp_rgbd_inf import GraspRgbdInf
from grasp_rgbd_inf import GraspInf
import time
import copy

class GraspsCNNInferServer:
    def __init__(self):
        rospy.init_node('grasps_cnn_inf_server')
        self.real_kinect2 = rospy.get_param('~real_kinect2', True)
        self.compute_finger_tip_loc = ComputeFingerTipPose(self.real_kinect2)
        self.rgbd_patches_save_path = rospy.get_param('~data_recording_path', '')
        self.use_hd = rospy.get_param('~use_hd', True)
        if self.real_kinect2:
            self.gen_rgbd = GenRgbdRealKinect2(self.rgbd_patches_save_path, self.use_hd) 
        else:
            self.gen_rgbd = GenRgbdGazeboKinect(self.rgbd_patches_save_path) 
        self.grasp_rgbd_inf = GraspInf(config_net=True, use_hd=self.use_hd)
        #self.listener = tf.TransformListener()
        self.blensor_dummy_frame = 'blensor_camera'
        

    def handle_grasps_inference(self, req):
        '''
            Grasps inference service handler. 
        '''
        rgbd_image = self.gen_rgbd.get_rgbd_image(req.rgb_image_path, req.depth_image_path, 
                                                        req.rgbd_info.scene_cloud_normal) 

        #self.gen_rgbd.save_rgbd_into_h5(rgbd_image, req.init_hand_pcd_true_config, req.init_hand_pcd_goal_config, 
        #                                req.close_hand_pcd_config, object_id=req.object_id, grasp_id=req.grasp_id,
        #                                object_name=req.object_name)

        self.compute_finger_tip_loc.set_up_input(req.init_hand_pcd_goal_config.palm_pose, 
                                                    req.init_hand_pcd_goal_config.hand_joint_state, use_hd=self.use_hd)
        self.compute_finger_tip_loc.proj_finger_palm_locs_to_img()
        palm_image_loc = self.compute_finger_tip_loc.palm_image_loc

        #grasp_patch = self.gen_rgbd.extract_rgbd_patch(rgbd_image, tuple(palm_image_loc), patch_size=400)  
        grasp_patch = self.gen_rgbd.get_palm_patch(rgbd_image, palm_image_loc, patch_size=400)  


        init_config_blensor = copy.deepcopy(req.init_hand_pcd_goal_config)
        init_config_blensor.palm_pose = \
                self.grasp_rgbd_inf.trans_palm_pose(self.blensor_dummy_frame, init_config_blensor.palm_pose)

        response = GraspCnnInferResponse()
        config_opt_blensor, suc_prob, suc_prob_init = \
                self.grasp_rgbd_inf.gradient_descent_inf(grasp_patch, init_config_blensor, 
                                            save_grad_to_log=False, object_id=req.object_id, 
                                            grasp_id=req.grasp_id) 
        #print req.init_hand_pcd_goal_config
        #print config_opt_blensor, suc_prob, suc_prob_init 
        print 'top grasp:', req.is_top_grasp
        print suc_prob, suc_prob_init 

        config_opt = copy.deepcopy(config_opt_blensor) 
        config_opt.palm_pose = self.grasp_rgbd_inf.trans_palm_pose(
                               req.init_hand_pcd_goal_config.palm_pose.header.frame_id, 
                               config_opt_blensor.palm_pose)

        response.inf_hand_pcd_config = config_opt
        response.inf_suc_prob = suc_prob
        response.init_suc_prob = suc_prob_init
        response.success = True
        return response

    def create_grasps_inference_server(self):
        '''
            Create grasps inference server.
        '''
        rospy.Service('grasp_cnn_inference', GraspCnnInfer, self.handle_grasps_inference)
        #rospy.Service('grasps_inference', GraspDataRecording, self.handle_grasps_inference)
        rospy.loginfo('Service grasps_inference:')
        rospy.loginfo('Ready to infer grasps.')

    def handle_grasps_sample_inference(self, req):
        '''
            Grasps inference service handler. 
        '''
        rgbd_image = self.gen_rgbd.get_rgbd_image(req.rgb_image_path, req.depth_image_path, 
                                                        req.rgbd_info.scene_cloud_normal) 
        hand_config_dummy = HandConfig() 
        self.gen_rgbd.save_rgbd_into_h5(rgbd_image, hand_config_dummy, hand_config_dummy, 
                                        hand_config_dummy, object_id=req.object_id, grasp_id=req.grasp_id,
                                        object_name=req.object_name)
       
        t = time.time()
        max_suc_prob = -1.
        max_suc_prob_idx = -1
        for i in xrange(len(req.sample_hand_joint_state_list)):
            sample_hand_config = HandConfig()
            sample_hand_config.palm_pose = req.sample_palm_pose_list[i]
            sample_hand_config.hand_joint_state = req.sample_hand_joint_state_list[i]
            self.compute_finger_tip_loc.set_up_input(sample_hand_config.palm_pose, 
                                                        sample_hand_config.hand_joint_state, use_hd=self.use_hd)
            self.compute_finger_tip_loc.proj_finger_palm_locs_to_img()
            palm_image_loc = self.compute_finger_tip_loc.palm_image_loc
            grasp_patch = self.gen_rgbd.extract_rgbd_patch(rgbd_image, tuple(palm_image_loc), patch_size=400)  
            q = self.grasp_rgbd_inf.convert_full_to_preshape_config(sample_hand_config)
            suc_prob = self.grasp_rgbd_inf.get_config_suc_prob_config_net(q, grasp_patch)
            if suc_prob >= max_suc_prob:
                max_suc_prob = suc_prob
                max_suc_prob_idx = i

        elapased_time = time.time() - t
        print 'Total sample inference time: ', str(elapased_time)
        response = GraspSampleInfResponse()
        response.max_suc_prob = max_suc_prob
        response.max_suc_prob_sample_idx = max_suc_prob_idx
        response.success = True
        print response
        return response

    def create_grasps_sample_inf_server(self):
        '''
            Create grasps inference server.
        '''
        rospy.Service('grasps_sample_inference', GraspSampleInf, self.handle_grasps_sample_inference)
        rospy.loginfo('Service grasps_sample_inference:')
        rospy.loginfo('Ready to evaluated sampled grasps.')



if __name__ == '__main__':
    grasps_inference = GraspsCNNInferServer() 
    grasps_inference.create_grasps_inference_server()
    # grasps_inference.create_grasps_sample_inf_server()
    rospy.spin()

