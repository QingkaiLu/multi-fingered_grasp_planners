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


class GraspALServer:


    def __init__(self):
        rospy.init_node('grasp_al_server')
        self.vis_preshape = rospy.get_param('~vis_preshape', False)
        virtual_hand_parent_tf = rospy.get_param('~virtual_hand_parent_tf', '')

        # grasp_net_model_path = pkg_path + '/models/grasp_al_net/' + \
        #                    'grasp_net_freeze_enc_2_sets.ckpt'
        # gmm_model_path = pkg_path + '/models/grasp_al_prior/gmm_2_sets'
        # mdn_model_path = pkg_path + '/models/grasp_al_prior/' + \
        #                     'prior_net_freeze_enc_2_sets.ckpt'

        grasp_net_model_path = pkg_path + '/models/grasp_al_net/' + \
                           'grasp_net_5_sets.ckpt'
        gmm_model_path = pkg_path + '/models/grasp_al_prior/gmm_5_sets'
        suc_gmm_model_path = pkg_path + '/models/grasp_al_prior/suc_gmm_5_sets'
        mdn_model_path = pkg_path + '/models/grasp_al_prior/' + \
                            'prior_net_5_sets.ckpt'
        suc_mdn_model_path = pkg_path + '/models/grasp_al_prior/' + \
                            'prior_net_suc_5_sets.ckpt'

        active_models_path = '/mnt/tars_data/multi_finger_sim_data/active_models/'
        # spv_data_path = '/mnt/tars_data/gazebo_al_grasps/train/' + \
        #                         'merged_grasp_data_6_6_and_6_8.h5'
        # active_data_path = '/mnt/tars_data/multi_finger_sim_data/grasp_data.h5'
        spv_data_path = '/mnt/tars_data/gazebo_al_grasps/train/' + \
                        'merged_grasp_data_6_6_and_6_8_and_6_10_and_6_11_and_6_13.h5'
        suc_spv_data_path = '/mnt/tars_data/gazebo_al_grasps/train/' + \
                        'merged_suc_grasp_5_sets.h5'
        active_data_path = '/mnt/tars_data/multi_finger_sim_data/grasp_data.h5'
        suc_active_data_path = '/mnt/tars_data/multi_finger_sim_data/suc_grasp_data.h5'
        al_file_path = '/mnt/tars_data/multi_finger_sim_data/al_data.h5' 
        self.data_proc_lib = DataProcLib()
        self.cvtf = gcf.ConfigConvertFunctions(self.data_proc_lib)
        # self.gal = GraspActiveLearner(grasp_net_model_path, prior_model_path, 
        #                                 al_file_path, active_models_path, 
        #                                 spv_data_path, active_data_path) 
        # prior_name = 'GMM' 
        prior_name = 'MDN'

        prior_model_path = None
        if prior_name == 'MDN':
            prior_model_path = mdn_model_path
            suc_prior_path = suc_mdn_model_path
        elif prior_name == 'GMM':
            prior_model_path = gmm_model_path
            suc_prior_path = suc_gmm_model_path

        self.gal = GraspActiveLearnerFK(prior_name, grasp_net_model_path, 
                                        prior_model_path, suc_prior_path,
                                        al_file_path, active_models_path, 
                                        spv_data_path, suc_spv_data_path, 
                                        active_data_path, suc_active_data_path,
                                        self.cvtf) 
        # self.active_method = 'UCB' 
        self.active_method = 'ALTER_OPT'
        # self.active_method = 'ALTER_POOLING'
        # self.active_method = 'UCB_POOLING'


    def handle_grasp_active_learning(self, req):
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

        # sparse_voxel_grid = self.data_proc_lib.voxel_gen_client(seg_obj_resp.obj)
        sparse_voxel_grid, voxel_size, voxel_grid_dim = \
                                 self.data_proc_lib.voxel_gen_client(seg_obj_resp.obj)
        
        # show_voxel.plot_voxel(sparse_voxel_grid) 

        voxel_grid = np.zeros(tuple(voxel_grid_dim))
        voxel_grid_index = sparse_voxel_grid.astype(int)
        voxel_grid[voxel_grid_index[:, 0], voxel_grid_index[:, 1], 
                    voxel_grid_index[:, 2]] = 1
        voxel_grid = np.expand_dims(voxel_grid, -1)

        # voxel_grid = np.zeros(tuple(self.data_proc_lib.voxel_grid_full_dim))
        # print sparse_voxel_grid
        # voxel_grid_index = sparse_voxel_grid.astype(int)
        # voxel_grid[voxel_grid_index[:, 0], voxel_grid_index[:, 1], 
        #             voxel_grid_index[:, 2]] = 1
        # voxel_grid = np.expand_dims(voxel_grid, -1)

        obj_size = np.array([seg_obj_resp.obj.width, seg_obj_resp.obj.height, 
                                    seg_obj_resp.obj.depth])

        self.gal.cur_act_reward = None
        self.gal.cur_action_id = None
        if self.active_method == 'UCB':
            reward, ik_config_inf, config_inf, obj_val_inf, \
                    ik_config_init, config_init, obj_val_init, \
                        res_info = self.gal.ucb(voxel_grid, obj_size)
        elif self.active_method == 'ALTER_OPT':
            reward, ik_config_inf, config_inf, obj_val_inf, \
                    ik_config_init, config_init, obj_val_init, \
                        res_info = self.gal.alternation(voxel_grid, obj_size, 
                                                        pooling=False)
        elif self.active_method == 'ALTER_POOLING':
            reward, ik_config_inf, config_inf, obj_val_inf, \
                    ik_config_init, config_init, obj_val_init, \
                        res_info = self.gal.alternation(voxel_grid, obj_size, 
                                                        pooling=True)
        elif self.active_method == 'UCB_POOLING':
            reward, ik_config_inf, config_inf, obj_val_inf, \
                    ik_config_init, config_init, obj_val_init, \
                        res_info = self.gal.ucb(voxel_grid, obj_size, 
                                                        pooling=True)

        if reward is None:
            print 'Active learning failed!'
            return response

        full_config_init = self.cvtf.convert_preshape_to_full_config(
                                            config_init, 'object_pose')
        full_config_inf = self.cvtf.convert_preshape_to_full_config(
                                            config_inf, 'object_pose')

        self.data_proc_lib.update_palm_pose_client(full_config_inf.palm_pose)

        response.reward = reward
        response.action = res_info['action']
        response.inf_ik_config_array = ik_config_inf
        response.inf_config_array = config_inf
        response.full_inf_config = full_config_inf
        response.inf_val = obj_val_inf
        response.inf_suc_prob = res_info['inf_suc_prob']
        response.inf_log_prior = res_info['inf_log_prior']
        response.inf_uct = res_info['inf_uct']
        response.init_val = obj_val_init
        response.init_suc_prob = res_info['init_suc_prob']
        response.init_log_prior = res_info['init_log_prior']
        response.init_uct = res_info['init_uct']
        response.init_ik_config_array = ik_config_init
        response.init_config_array = config_init
        response.full_init_config = full_config_init
        response.sparse_voxel_grid = sparse_voxel_grid.flatten()
        response.object_size = obj_size
        response.success = True
        return response


    def create_grasps_al_server(self):
        '''
            Create grasps active learning server.
        '''
        rospy.Service('grasp_active_learn', GraspActiveLearn, 
                        self.handle_grasp_active_learning)
        rospy.loginfo('Service grasp_active_learn:')
        rospy.loginfo('Ready to perform grasp active learning.')


    def handle_active_model_update(self, req):
        response = ActiveModelUpdateResponse() 
        self.gal.active_model_update(req.batch_id)
        response.success = True
        return response


    def create_al_model_update_server(self):
        '''
            Create grasps active learning model update server.
        '''
        rospy.Service('active_model_update', ActiveModelUpdate, 
                        self.handle_active_model_update)
        rospy.loginfo('Service active_model_update:')
        rospy.loginfo('Ready to perform grasp active learning model update.')


    def handle_active_data_update(self, req):
        response = ActiveDataUpdateResponse() 
        if req.grasp_has_plan:
            self.gal.active_data_update()
        response.success = True
        return response


    def create_al_data_update_server(self):
        '''
            Create grasps active learning model update server.
        '''
        rospy.Service('active_data_update', ActiveDataUpdate, 
                        self.handle_active_data_update)
        rospy.loginfo('Service active_data_update:')
        rospy.loginfo('Ready to perform grasp active learning data update.')


if __name__ == '__main__':
    grasps_al = GraspALServer() 
    grasps_al.create_grasps_al_server()
    grasps_al.create_al_model_update_server()
    grasps_al.create_al_data_update_server()
    rospy.spin()

