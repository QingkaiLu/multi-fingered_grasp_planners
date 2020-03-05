#!/usr/bin/env python
import roslib
import rospy
from prob_grasp_planner.srv import *
# from grasp_active_learner import GraspActiveLearner 
from grasp_active_learner_fk import GraspActiveLearnerFK 
import numpy as np
import time
from geometry_msgs.msg import Pose, Quaternion, PoseStamped, PointStamped
import roslib.packages as rp
import sys
pkg_path = rp.get_pkg_dir('prob_grasp_planner')
sys.path.append(pkg_path + '/src/grasp_common_library')
import grasp_common_functions as gcf
import align_object_frame as align_obj
from data_proc_lib import DataProcLib 
import show_voxel
import tensorflow as tf


class ActivePlanServer:


    def __init__(self):
        rospy.init_node('active_plan_server')
        self.vis_preshape = rospy.get_param('~vis_preshape', False)
        virtual_hand_parent_tf = rospy.get_param('~virtual_hand_parent_tf', '')
        active_models_path = rospy.get_param('~active_models_path', '')
        
        #active_models_path = '/mnt/tars_data/multi_finger_sim_data/active_models/'

        # less_data_net_path = pkg_path + '/models/grasp_al_net/' + \
        #                    'grasp_net_freeze_enc_2_sets.ckpt'
        # less_data_prior_path = pkg_path + '/models/grasp_al_prior/gmm_2_sets'

        # more_data_net_path = pkg_path + '/models/grasp_al_net/' + \
        #                    'grasp_net_freeze_enc_10_sets.ckpt'
        # more_data_prior_path = pkg_path + '/models/grasp_al_prior/gmm_10_sets'

        less_data_net_path = pkg_path + '/models/grasp_al_net/' + \
                           'grasp_net_5_sets.ckpt'
        less_data_prior_path = pkg_path + '/models/grasp_al_prior/prior_net_5_sets.ckpt'

        more_data_net_path = pkg_path + '/models/grasp_al_net/' + \
                           'grasp_net_10_sets.ckpt'
        more_data_prior_path = pkg_path + '/models/grasp_al_prior/prior_net_10_sets.ckpt'


        self.data_proc_lib = DataProcLib()
        self.cvtf = gcf.ConfigConvertFunctions(self.data_proc_lib)
       
        #prior_name = 'GMM' 
        prior_name = 'MDN'

        g1 = tf.Graph()
        g2 = tf.Graph()
        g3 = tf.Graph()
        with g1.as_default():
            self.active_planner = GraspActiveLearnerFK(prior_name, 
                                        active_models_path=active_models_path,
                                        cvtf=self.cvtf) 
        with g2.as_default():
            self.more_data_spv_planner = GraspActiveLearnerFK(prior_name,
                                            grasp_net_model_path=more_data_net_path, 
                                            prior_model_path=more_data_prior_path,
                                            cvtf=self.cvtf) 
        with g3.as_default():
            self.less_data_spv_planner = GraspActiveLearnerFK(prior_name,
                                            grasp_net_model_path=less_data_net_path, 
                                            prior_model_path=less_data_prior_path,
                                            cvtf=self.cvtf) 
        self.action = 'grasp_suc'


    def handle_grasp_active_plan(self, req):
        response = GraspActiveLearnResponse()
        response.success = False

        seg_obj_resp = gcf.segment_object_client(self.data_proc_lib.listener)
        if not seg_obj_resp.object_found:
            print 'No object found for segmentation!'
            return response 

        obj_world_pose_stamp = PoseStamped()
        obj_world_pose_stamp.header.frame_id = seg_obj_resp.obj.header.frame_id
        obj_world_pose_stamp.pose = seg_obj_resp.obj.pose
        self.data_proc_lib.update_object_pose_client(obj_world_pose_stamp)

        sparse_voxel_grid, voxel_size, voxel_grid_full_dim = self.data_proc_lib.voxel_gen_client(seg_obj_resp.obj)
        
        # show_voxel.plot_voxel(sparse_voxel_grid) 

        voxel_grid = np.zeros(tuple(voxel_grid_full_dim))
        voxel_grid_index = sparse_voxel_grid.astype(int)
        voxel_grid[voxel_grid_index[:, 0], voxel_grid_index[:, 1], 
                    voxel_grid_index[:, 2]] = 1
        voxel_grid = np.expand_dims(voxel_grid, -1)

        obj_size = np.array([seg_obj_resp.obj.width, seg_obj_resp.obj.height, 
                                    seg_obj_resp.obj.depth])

        planner = None
        print req.planner_name
        if req.planner_name == 'active_planner':
            planner = self.active_planner
        elif req.planner_name == 'more_data_spv_planner':
            planner = self.more_data_spv_planner
        elif req.planner_name == 'less_data_spv_planner':
            planner = self.less_data_spv_planner

        reward, ik_config_inf, config_inf, obj_val_inf, \
            ik_config_init, config_init, obj_val_init, plan_info, inf_time = \
            planner.grasp_strategies(self.action, voxel_grid,
                                            obj_size)

        if reward is None:
            print 'Active planning failed!'
            return response

        full_config_init = self.cvtf.convert_preshape_to_full_config(
                                            config_init, 'object_pose')
        full_config_inf = self.cvtf.convert_preshape_to_full_config(
                                            config_inf, 'object_pose')

        self.data_proc_lib.update_palm_pose_client(full_config_inf.palm_pose)

        response.reward = reward
        response.action = plan_info['action']
        response.inf_ik_config_array = ik_config_inf
        response.inf_config_array = config_inf
        response.full_inf_config = full_config_inf
        response.inf_val = obj_val_inf
        response.inf_suc_prob = plan_info['inf_suc_prob']
        response.inf_log_prior = plan_info['inf_log_prior']
        response.inf_uct = plan_info['inf_uct']
        response.init_val = obj_val_init
        response.init_suc_prob = plan_info['init_suc_prob']
        response.init_log_prior = plan_info['init_log_prior']
        response.init_uct = plan_info['init_uct']
        response.init_ik_config_array = ik_config_init
        response.init_config_array = config_init
        response.full_init_config = full_config_init
        response.sparse_voxel_grid = sparse_voxel_grid.flatten()
        response.voxel_size = voxel_size
        response.voxel_grid_dim = voxel_grid_full_dim
        response.object_size = obj_size
        response.success = True
        return response


    def create_active_plan_server(self):
        '''
            Create grasps active learning server.
        '''
        rospy.Service('grasp_active_plan', GraspActiveLearn, 
                        self.handle_grasp_active_plan)
        rospy.loginfo('Service grasp_active_plan:')
        rospy.loginfo('Ready to perform grasp active planning.')


if __name__ == '__main__':
   active_planner = ActivePlanServer() 
   active_planner.create_active_plan_server()
   rospy.spin()

