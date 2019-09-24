#!/usr/bin/env python
import roslib
#roslib.load_manifest('prob_grasp_planner')
import rospy
from prob_grasp_planner.srv import *
from prob_grasp_planner.msg import VisualInfo, HandConfig
from sensor_msgs.msg import Image, JointState, CameraInfo
from geometry_msgs.msg import Pose, Quaternion, PoseStamped
from cv_bridge import CvBridge, CvBridgeError
import tf
import cv2
import numpy as np
import time
from grasp_pgm_inference import GraspPgmInfer 
import show_voxel

class GraspPgmInferServer:
    def __init__(self):
        rospy.init_node('grasps_pgm_inf_server')
        #self.rgbd_patches_save_path = rospy.get_param('~data_recording_path', 
        #        '/media/kai/multi_finger_sim_data/')
        #self.use_hd = rospy.get_param('~use_hd', True)
        #self.gen_rgbd = GenRgbdImage(self.rgbd_patches_save_path) 
        #self.grasp_rgbd_inf = GraspInf(config_net=True, use_hd=self.use_hd)
        self.voxel_grid_dim = [20, 20, 20]
        self.voxel_size = [0.01, 0.01, 0.01]
        self.grasp_pgm_inf_type = GraspPgmInfer(pgm_grasp_type=True) 
        self.grasp_pgm_inf_no_type = GraspPgmInfer(pgm_grasp_type=False) 
        
    #def voxel_gen_client(self, scene_cloud, object_frame_id):
    #    '''
    #    Service client to get voxel from pointcloud in object frame.
    #    
    #    Args:
    #        scene_cloud.

    #    Returns:
    #        Voxelgrid.
    #    '''
    #    rospy.loginfo('Waiting for service gen_inference_voxel.')
    #    rospy.wait_for_service('gen_inference_voxel')
    #    rospy.loginfo('Calling service gen_inference_voxel.')
    #    try:
    #        gen_voxel_proxy = rospy.ServiceProxy('gen_inference_voxel', GenInfVoxel)
    #        gen_voxel_request = GenInfVoxelRequest()
    #        gen_voxel_request.scene_cloud = scene_cloud
    #        gen_voxel_request.voxel_dim = self.voxel_grid_dim
    #        gen_voxel_request.voxel_size = self.voxel_size
    #        gen_voxel_request.object_frame_id = object_frame_id

    #        gen_voxel_response = gen_voxel_proxy(gen_voxel_request) 
    #        sparse_voxel_grid = np.reshape(gen_voxel_response.voxel_grid, 
    #                            [len(gen_voxel_response.voxel_grid) / 3, 3])
    #        return sparse_voxel_grid
    #    except rospy.ServiceException, e:
    #        rospy.loginfo('Service gen_inference_voxel call failed: %s'%e)
    #    rospy.loginfo('Service gen_inference_voxel is executed.')

    def handle_grasp_inference(self, req):
        '''
            Grasps inference service handler. 
        '''
        if req.grasp_type == 'all':
            grasp_pgm_inf = self.grasp_pgm_inf_no_type
        elif req.grasp_type == 'prec' or req.grasp_type == 'power':
            grasp_pgm_inf = self.grasp_pgm_inf_type
        else:
            rospy.logerr('Wrong grasp type for inference!')

        #Generate voxel grid from the point cloud
        sparse_voxel_grid = grasp_pgm_inf.voxel_gen_client(req.scene_cloud, req.object_frame_id)
        #show_voxel.plot_voxel(sparse_voxel_grid, './voxel.png') 
        object_voxel_grid = np.zeros(tuple(self.voxel_grid_dim))
        voxel_grid_index = sparse_voxel_grid.astype(int)
        object_voxel_grid[voxel_grid_index[:, 0], voxel_grid_index[:, 1], voxel_grid_index[:, 2]] = 1

        response = GraspPgmInfResponse()
        #config_opt, suc_prob, suc_prob_init = \
        #        grasp_pgm_inf.gd_inf_one_type(req.grasp_type, object_voxel_grid, 
        #                            req.init_hand_object_config, save_grad_to_log=True, 
        #                            object_id=req.object_id, grasp_id=req.grasp_id)

        config_opt, suc_prob, suc_prob_init = \
                            grasp_pgm_inf.quasi_newton_lbfgs_inf(req.grasp_type, object_voxel_grid,
                                    req.init_hand_object_config, req.object_frame_id, bfgs=False)

        #config_opt, suc_prob, suc_prob_init = \
        #                    grasp_pgm_inf.quasi_newton_lbfgs_inf(req.grasp_type, object_voxel_grid,
        #                            req.init_config_array, req.object_frame_id, bfgs=False)
        #config_opt, suc_prob, suc_prob_init = \
        #        grasp_pgm_inf.gd_inf_one_type(req.grasp_type, object_voxel_grid, 
        #                            req.init_config_array, save_grad_to_log=False, 
        #                            object_id=req.object_id, grasp_id=req.grasp_id)

        #if req.grasp_type == 'prec':
        #    config_opt.palm_pose.pose.position.x += 0.035

        #Offset for robot modelling error.
        #Can't add the offset if it's too close to the object, since 
        #the planner can't find the plan if the inaccurate robot model
        #is too close to the object. This might not be the best way to solve 
        #the collision checking problem. I need to dig this deeper to figure out 
        #a better way.
        #if config_opt.palm_pose.pose.position.x < -0.05:
        #    config_opt.palm_pose.pose.position.x += 0.03

        #config_opt.palm_pose.pose.position.x += 0.03
        #config_opt.palm_pose.pose.position.x += 0.04


        response.inf_hand_object_config = config_opt
        response.inf_suc_prob = suc_prob
        response.init_suc_prob = suc_prob_init
        response.success = True
        return response

    def create_grasps_inference_server(self):
        '''
            Create grasps inference server.
        '''
        rospy.Service('grasp_pgm_inference', GraspPgmInf, self.handle_grasp_inference)
        rospy.loginfo('Service grasp_pgm_inference:')
        rospy.loginfo('Ready to infer pgm grasps.')

if __name__ == '__main__':
    grasps_inference = GraspPgmInferServer() 
    grasps_inference.create_grasps_inference_server()
    rospy.spin()

