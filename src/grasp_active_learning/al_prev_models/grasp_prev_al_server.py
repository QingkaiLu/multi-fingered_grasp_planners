#!/usr/bin/env python
import roslib
import rospy
from prob_grasp_planner.srv import *
from grasp_active_learner import GraspActiveLearner 
import numpy as np
import time
import roslib.packages as rp
import sys
pkg_path = rp.get_pkg_dir('prob_grasp_planner')
sys.path.append(pkg_path + '/src/grasp_cnn_planner')
#from gen_rgbd_images import GenRgbdImage
from compute_finger_tip_location import ComputeFingerTipPose
import copy

class GraspALServer:


    def __init__(self):
        rospy.init_node('grasp_al_server')
        self.grasp_planner_name = 'grasp_type_pgm'
        #self.grasp_planner_name = 'grasp_config_net'
        self.vis_preshape = rospy.get_param('~vis_preshape', False)
        virtual_hand_parent_tf = rospy.get_param('~virtual_hand_parent_tf', '')
        self.grasp_active_learn = GraspActiveLearner(self.grasp_planner_name, 
                                                     self.vis_preshape, 
                                                     virtual_hand_parent_tf=virtual_hand_parent_tf)
        
        if self.grasp_planner_name == 'grasp_type_pgm':
            #Create member variables for grasp type pgm planner
            self.voxel_grid_dim = [20, 20, 20]
            self.voxel_size = [0.01, 0.01, 0.01]
        elif self.grasp_planner_name == 'grasp_config_net':
            #Create member variables for grasp cnn planner
            self.rgbd_patches_save_path = rospy.get_param('~data_recording_path', '')
            self.use_hd = rospy.get_param('~use_hd', True)
            self.gen_rgbd = GenRgbdImage(self.rgbd_patches_save_path, self.use_hd) 
            self.compute_finger_tip_loc = ComputeFingerTipPose()
            self.blensor_dummy_frame = 'blensor_camera'
       
    
    def handle_grasp_active_learning(self, req):
        '''
            Grasps active learning service handler. 
        '''
        if req.grasp_planner_name == 'grasp_type_pgm':
            #Generate voxel grid from the point cloud
            sparse_voxel_grid = self.grasp_active_learn.grasp_model.voxel_gen_client(
                                req.scene_cloud, req.object_frame_id)
            #show_voxel.plot_voxel(sparse_voxel_grid, './voxel.png') 
            object_voxel_grid = np.zeros(tuple(self.voxel_grid_dim))
            voxel_grid_index = sparse_voxel_grid.astype(int)
            object_voxel_grid[voxel_grid_index[:, 0], voxel_grid_index[:, 1], voxel_grid_index[:, 2]] = 1
       
            #config_opt, entropy, entropy_init = \
            #    self.grasp_active_learn.active_max_entropy_lbfgs(req.grasp_type, object_voxel_grid,
            #                                req.init_hand_object_config, req.object_frame_id, bfgs=False)
            config_opt, uncertainty, uncertainty_init, init_config = \
                self.grasp_active_learn.active_max_uncertainty_lbfgs(req.grasp_type, object_voxel_grid,
                                            req.init_hand_object_config, req.object_frame_id, bfgs=False)

            #print config_opt, uncertainty, uncertainty_init, init_config
            #config_opt, entropy, entropy_init = \
            #    self.grasp_active_learn.gd_entropy(req.init_hand_object_config, grasp_type=req.grasp_type, 
            #                                       object_voxel=object_voxel_grid)
        elif self.grasp_planner_name == 'grasp_config_net':
            rgbd_image = self.gen_rgbd.get_rgbd_image(req.rgb_image_path, req.depth_image_path, 
                                                            req.rgbd_info.scene_cloud_normal) 

            self.compute_finger_tip_loc.set_up_input(req.init_hand_pcd_goal_config.palm_pose, 
                                                    req.init_hand_pcd_goal_config.hand_joint_state,
                                                    use_hd=self.use_hd)
            self.compute_finger_tip_loc.proj_finger_palm_locs_to_img()
            palm_image_loc = self.compute_finger_tip_loc.palm_image_loc

            #grasp_patch = self.gen_rgbd.extract_rgbd_patch(rgbd_image, tuple(palm_image_loc), patch_size=400)  
            grasp_patch = self.gen_rgbd.get_palm_patch(rgbd_image, palm_image_loc, patch_size=400)  

            init_config_blensor = copy.deepcopy(req.init_hand_pcd_goal_config)
            init_config_blensor.palm_pose = \
                    self.grasp_active_learn.grasp_model.trans_palm_pose(
                                self.blensor_dummy_frame, init_config_blensor.palm_pose)

            response = GraspCnnInferResponse()
            config_opt_blensor, entropy, entropy_init = \
                    self.grasp_active_learn.gd_entropy(init_config_blensor, grasp_rgbd_patch=grasp_patch,
                                                save_grad_to_log=False, object_id=req.object_id, 
                                                grasp_id=req.grasp_id) 
            print req.init_hand_pcd_goal_config
            print config_opt_blensor, entropy, entropy_init
            #print 'top grasp:', req.is_top_grasp
            #print entropy, entropy_init 

            config_opt = copy.deepcopy(config_opt_blensor) 
            config_opt.palm_pose = self.grasp_active_learn.grasp_model.trans_palm_pose(
                                   req.init_hand_pcd_goal_config.palm_pose.header.frame_id, 
                                   config_opt_blensor.palm_pose)


        response = GraspActiveLearnResponse()
        response.inf_hand_object_config = config_opt
        #response.inf_entropy = entropy
        #response.init_entropy = entropy_init
        response.inf_val = uncertainty
        response.init_val = uncertainty_init
        response.init_sample_config = init_config
        response.success = True
        return response


    def create_grasps_al_server(self):
        '''
            Create grasps active learning server.
        '''
        rospy.Service('grasp_active_learn', GraspActiveLearn, self.handle_grasp_active_learning)
        rospy.loginfo('Service grasp_active_learn:')
        rospy.loginfo('Ready to perform grasp active learning.')


if __name__ == '__main__':
    grasps_al = GraspALServer() 
    grasps_al.create_grasps_al_server()
    rospy.spin()

