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
import show_voxel
import tf
import os
import align_object_frame as align_obj_frame

class ProcGraspData:
    '''
    Class to process grasp data. Grasp data includes grasp object voxel,
    grasp configuration and grasp success label.
    '''
    # TODO: replace voxel generation functions with these functions of 
    # data_proc_lib in grasp_common_library
    def __init__(self, data_path):
        rospy.init_node('proc_grasp_data') 
        #self.data_path = '/media/kai/multi_finger_sim_data_complete_v4/'
        #self.data_path = '/media/kai/multi_finger_sim_data_precision_1/'
        self.data_path = data_path
        self.grasp_patches_file_path = self.data_path + 'grasp_patches.h5'
        self.grasp_data_file_path = self.data_path + 'grasp_data.h5'
        #self.grasp_voxel_file_path = self.data_path + 'grasp_voxel_data.h5'
        self.grasp_voxel_file_path = self.data_path + 'grasp_voxel_obj_tf_data.h5'
        #self.grasp_voxel_file_path = self.data_path + 'grasp_voxel_non_top_data.h5'
        #self.grasp_voxel_file_path = self.data_path + 'power_failure_grasp_voxel_data.h5'
        #self.grasp_voxel_file_path = self.data_path + 'power_align_failure_grasps.h5'
        #self.grasp_voxel_file_path = self.data_path + 'prec_failure_grasp_voxel_data.h5'
        #self.grasp_voxel_file_path = self.data_path + 'prec_align_failure_grasps.h5'
        #self.intialize_grasp_voxel_file(self.grasp_voxel_file_path)

        self.voxel_grid_dim = [20, 20, 20]
        self.voxel_size = [0.01, 0.01, 0.01]

        #self.voxel_grid_dim = [2, 2, 2]
        #self.voxel_size = [0.02, 0.02, 0.02]

        self.config_dim = 14
        self.classes_num = 1

        # 3 dof location of the palm + 3 dof orientation of the palm (4 parameters for quaternion, 
        # 3 for Euler angles) + 1st two joints of the thumb + 1st joint of other three fingers
        # Other joint angles are fixed for grasp preshape inference.
        self.palm_loc_dof_dim = 3
        self.palm_dof_dim = 6
        #self.finger_joints_dof_dim = 8
        #self.theta_dim = self.palm_dof_dim + self.finger_joints_dof_dim 

        self.hand_config_frame_id = 'blensor_camera'
        self.object_frame_id = 'object_pose'

        self.tf_br = tf.TransformBroadcaster()
        self.listener = tf.TransformListener()
        self.align_object_tf = True

        self.gen_failure_data = False #True
        self.gen_top_grasp = True #False 

    def intialize_grasp_voxel_file(self, file_name):
        '''
        Initialize the grasp voxel h5 file.
        '''
        grasp_voxel_file = h5py.File(file_name, 'w')
        grasps_number_key = 'grasps_number'
        if grasps_number_key not in grasp_voxel_file:
            grasp_voxel_file.create_dataset(grasps_number_key, data=0)
        grasp_voxel_file.close()

    def voxel_gen_client(self, pcd_file_path, trans_camera_to_palm):
        '''
        Service client to get voxel from pointcloud.
        
        Args:
            pcd_file_path.
            trans_camera_to_palm: transformation from camera to palm frame.

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
            gen_voxel_request.trans_camera_to_palm = np.reshape(trans_camera_to_palm, 16)
            gen_voxel_request.voxel_dim = self.voxel_grid_dim
            gen_voxel_request.voxel_size = self.voxel_size

            gen_voxel_response = gen_voxel_proxy(gen_voxel_request) 
            return np.reshape(gen_voxel_response.voxel_grid, 
                                [len(gen_voxel_response.voxel_grid) / 3, 3])
        except rospy.ServiceException, e:
            rospy.loginfo('Service gen_voxel_from_pcd call failed: %s'%e)
        rospy.loginfo('Service gen_voxel_from_pcd is executed.')

    def voxel_gen_client(self, pcd_file_path):
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

    def convert_array_to_pose(self, pose_array):
        '''
        Convert pose Quaternion array to ROS PoseStamped.

        Args:
            pose_array

        Returns:
            ROS pose.
        '''
        pose_stamp = PoseStamped()
        pose_stamp.header.frame_id = 'world'
        pose_stamp.pose.position.x, pose_stamp.pose.position.y, \
                pose_stamp.pose.position.z = pose_array[:self.palm_loc_dof_dim]    
    
        pose_stamp.pose.orientation.x, pose_stamp.pose.orientation.y, \
                pose_stamp.pose.orientation.z, pose_stamp.pose.orientation.w = pose_array[self.palm_loc_dof_dim:]    

        return pose_stamp

    def convert_preshape_to_full_config(self, preshape_config):
        '''
        Convert preshape grasp configuration to full grasp configuration by filling zeros for 
        uninferred finger joints.

        Args:
            Preshape configuration.
        
        Returns:
            The full grasp configuration.

        '''
        hand_config = HandConfig()
        hand_config.palm_pose.header.frame_id = self.hand_config_frame_id

        hand_config.palm_pose.pose.position.x, hand_config.palm_pose.pose.position.y, \
                hand_config.palm_pose.pose.position.z = preshape_config[:self.palm_loc_dof_dim]    
    
        palm_euler = preshape_config[self.palm_loc_dof_dim:self.palm_dof_dim] 
        palm_quaternion = tf.transformations.quaternion_from_euler(palm_euler[0], palm_euler[1], palm_euler[2])
        hand_config.palm_pose.pose.orientation.x, hand_config.palm_pose.pose.orientation.y, \
                hand_config.palm_pose.pose.orientation.z, hand_config.palm_pose.pose.orientation.w = palm_quaternion 

        hand_config.hand_joint_state.name = ['index_joint_0','index_joint_1','index_joint_2', 'index_joint_3',
                   'middle_joint_0','middle_joint_1','middle_joint_2', 'middle_joint_3',
                   'ring_joint_0','ring_joint_1','ring_joint_2', 'ring_joint_3',
                   'thumb_joint_0','thumb_joint_1','thumb_joint_2', 'thumb_joint_3']
        hand_config.hand_joint_state.position = [preshape_config[self.palm_dof_dim], preshape_config[self.palm_dof_dim + 1], 0., 0.,
                                                preshape_config[self.palm_dof_dim + 2], preshape_config[self.palm_dof_dim + 3], 0., 0.,
                                                preshape_config[self.palm_dof_dim + 4], preshape_config[self.palm_dof_dim + 5], 0., 0.,
                                                preshape_config[self.palm_dof_dim + 6], preshape_config[self.palm_dof_dim + 7], 0., 0.]
        return hand_config

    def gen_grasp_pcd_map(self):
        '''
        Generate the map from one grasp to its pcd file.

        Notice:
            There are around 20 grasps whose pcd file paths are None here.
            Among these grasps, there are two cases:
            a. grasps with object_grasp_id 20. 
            b. grasps with object_grasp_id that's not 20. There are less than 
            5 these grasps. I havn't figured out why this happens yet. There is no
            pcd file saved for these grasps. So the way to get the point cloud for
            these grasps would be to convert the RGBD image to point cloud with the 
            Blensor camera intrinsic parameters. 
        '''
        grasp_patches_file = h5py.File(self.grasp_patches_file_path, 'r')
        grasp_data_file = h5py.File(self.grasp_data_file_path, 'r')
        grasp_to_pcd_file = h5py.File(self.data_path + 'grasp_to_pcd.h5', 'w')

        grasps_num = grasp_patches_file['grasps_number'][()] 

        for i in xrange(grasps_num):
        #for i in xrange(10):
            #print 'sample_id: ', i
            grasp_sample_id = 'grasp_' + str(i)
            print grasp_sample_id
            object_grasp_id_key = grasp_sample_id + '_object_grasp_id'
            object_grasp_id = grasp_patches_file[object_grasp_id_key][()]
            object_grasp_id_split = object_grasp_id.split('_')
            object_id = object_grasp_id_split[1] 
            grasp_id = object_grasp_id_split[3] 
            grasp_id = int(grasp_id)
            grasp_object_name_key = 'object_' + object_id + '_name'
            object_name = grasp_data_file[grasp_object_name_key][()] 
            
            pcd_file_name = 'object_' + object_id + '_' + object_name + \
                    '_grasp_' + str(grasp_id) + '00000.pcd' 
            pcd_file_path = self.data_path + 'pcd/' + pcd_file_name          
            print pcd_file_path

            grasp_pose_key = 'object_' + object_id + '_grasp_' + str(grasp_id)  + '_object_world_pose'
            if grasp_pose_key in grasp_data_file:
                grasp_pose = grasp_data_file[grasp_pose_key][()]
                print grasp_pose
            else:
                pcd_file_name = None
                continue

            if not os.path.isfile(pcd_file_path):
                pcd_file_name = None
                for i in xrange(1, 3):
                    prev_pcd_file_name = 'object_' + object_id + '_' + object_name + \
                            '_grasp_' + str(grasp_id - i) + '00000.pcd' 
                    prev_pcd_file_path = self.data_path + 'pcd/' + prev_pcd_file_name          
                    print prev_pcd_file_name

                    if os.path.isfile(prev_pcd_file_path):
                        grasp_pose_key = 'object_' + object_id + '_grasp_' + str(grasp_id - i)  + '_object_world_pose'
                        grasp_pose_prev = grasp_data_file[grasp_pose_key][()]
                        print grasp_pose_prev
                        #if np.array_equal(grasp_pose_prev, grasp_pose):
                        if np.allclose(grasp_pose_prev, grasp_pose):
                            print 'object pose equal'
                            pcd_file_name = prev_pcd_file_name
                            break
                        print 'object pose not equal'
                        print grasp_pose - grasp_pose_prev

            print pcd_file_name
            if pcd_file_name is not None:
                grasp_to_pcd_file.create_dataset(grasp_sample_id, data=pcd_file_name)
            print grasp_sample_id in grasp_to_pcd_file
            print '##############'

        grasp_to_pcd_file.close()

    def trans_pose_to_obj_tf(self, pose_stamp):
        '''
        Transform a pose into object frame.

        Args:
            palm_pose_stamp
        Returns:
            palm pose array in object frame.
        '''
        pose_object = None
        rate = rospy.Rate(10.0)
        i = 0
        while (not rospy.is_shutdown()) and i < 50:
            try:
                if pose_object is None:
                    pose_object = self.listener.transformPose(self.object_frame_id, pose_stamp)
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
            i += 1
            rate.sleep()

        pose_quaternion = (pose_object.pose.orientation.x, pose_object.pose.orientation.y, 
                pose_object.pose.orientation.z, pose_object.pose.orientation.w) 
        pose_euler = tf.transformations.euler_from_quaternion(pose_quaternion)
        pose_object_array = [pose_object.pose.position.x, pose_object.pose.position.y,
                pose_object.pose.position.z, pose_euler[0], pose_euler[1], pose_euler[2]]

        vis_tf = False#True
        if vis_tf:
            j = 0
            #while not rospy.is_shutdown():
            while j < 10:
                self.tf_br.sendTransform((pose_object.pose.position.x, pose_object.pose.position.y, 
                        pose_object.pose.position.z),
                        (pose_object.pose.orientation.x, pose_object.pose.orientation.y, 
                        pose_object.pose.orientation.z, pose_object.pose.orientation.w),
                        rospy.Time.now(), 'palm_pose_object', pose_object.header.frame_id)
            j += 1
            rate.sleep(1)
        
        return pose_object_array

    def proc_data(self):
        '''
        Process grasp data to get grasp voxel data.
        '''
        grasp_patches_file = h5py.File(self.grasp_patches_file_path, 'r')
        grasp_data_file = h5py.File(self.grasp_data_file_path, 'r')
        #grasp_voxel_file = h5py.File(self.grasp_voxel_file_path, 'w')
        grasp_voxel_file = h5py.File(self.grasp_voxel_file_path, 'a')
        grasp_to_pcd_file = h5py.File(self.data_path + 'grasp_to_pcd.h5', 'r')

        grasps_num = grasp_patches_file['grasps_number'][()] 
        #grasp_configs = np.zeros((grasps_num, self.config_dim))       
        #grasp_labels = np.zeros((grasps_num, self.classes_num))       
        #grasp_voxel_grids = np.zeros((grasps_num,) + tuple(self.voxel_grid_dim)) 
        grasp_configs = []
        grasp_configs_obj = []
        grasp_labels = []
        grasp_voxel_grids = []

        voxel_grasps_id = 0
        # pcd files with loading error: vector::_M_range_check 
        # To do: figure out why.
        problematic_pcd_files = {'object_135_mom_to_mom_sweet_potato_corn_apple_grasp_300000.pcd'}
        #samples without object world pose
        #KeyError: "Unable to open object (Object 'object_154_grasp_5_object_world_pose' doesn't exist)"
        samples_without_obj_pose = {1292}
        for i in xrange(grasps_num):
        #for i in xrange(50):
            #if i in samples_without_obj_pose:
            #    continue
            print 'sample_id: ', i
            grasp_sample_id = 'grasp_' + str(i)
            grasp_config_key = grasp_sample_id + '_preshape_true_config'
            #grasp_configs[i] = grasp_patches_file[grasp_config_key]
            grasp_config = grasp_patches_file[grasp_config_key]
            object_grasp_id_key = grasp_sample_id + '_object_grasp_id'
            object_grasp_id = grasp_patches_file[object_grasp_id_key][()]
            object_pose_key = object_grasp_id + '_object_world_pose'
            #Skip grasps without object world poses.
            if object_pose_key in grasp_data_file:
                object_pose_array = grasp_data_file[object_pose_key]
            else:
                continue

            grasp_label_key = grasp_sample_id + '_grasp_label'
            grasp_label = grasp_patches_file[grasp_label_key][()]

            if not self.gen_top_grasp:
                #Skip top precision and successful grasps to collect failure power grasps
                top_grasp_key = object_grasp_id + '_top_grasp'
                top_grasp = grasp_data_file[top_grasp_key][()]
                if top_grasp:
                    continue

            if self.gen_failure_data:
                if grasp_label == 1:
                    continue

            voxel_grasp_id_key = 'voxel_grasp_' + str(voxel_grasps_id)
            print voxel_grasp_id_key
            sparse_voxel_key = voxel_grasp_id_key + '_sparse_voxel'
            if voxel_grasp_id_key not in grasp_voxel_file:
            #if sparse_voxel_key not in grasp_voxel_file:
                # Resume from previous voxel grids generation.
                if grasp_sample_id in grasp_to_pcd_file:
                    pcd_file_name = grasp_to_pcd_file[grasp_sample_id][()]
                    print pcd_file_name
                else:
                    print '***pcd_file does not exist!'
                    continue
                if pcd_file_name in problematic_pcd_files:
                    print '***Bad pcd_file!!!'
                    continue
                pcd_file_path = self.data_path + 'pcd/' + pcd_file_name          
   
                hand_config = self.convert_preshape_to_full_config(grasp_config) 
                print hand_config.palm_pose
                self.update_palm_pose_client(hand_config.palm_pose)

                object_pose = self.convert_array_to_pose(object_pose_array)
                if self.align_object_tf:
                    object_pose = align_obj_frame.align_obj_ort(object_pose, self.listener)
                print object_pose
                self.update_object_pose_client(object_pose)

                sparse_voxel_grid = self.voxel_gen_client(pcd_file_path)
                #print sparse_voxel_grid
                
                #if i % 10 == 0:
                #    show_voxel.plot_voxel(sparse_voxel_grid, './voxel.png') 
                #show_voxel.plot_voxel(sparse_voxel_grid, './voxel.png') 

                #if voxel_grasp_id_key not in grasp_voxel_file:
                #    grasp_voxel_file.create_dataset(voxel_grasp_id_key, data=grasp_sample_id)
                grasp_voxel_file.create_dataset(voxel_grasp_id_key, data=grasp_sample_id)
                grasp_voxel_file.create_dataset(sparse_voxel_key, data=sparse_voxel_grid)
            else:
                sparse_voxel_grid = grasp_voxel_file[sparse_voxel_key][()]

                hand_config = self.convert_preshape_to_full_config(grasp_config) 
                print hand_config.palm_pose
                self.update_palm_pose_client(hand_config.palm_pose)

                object_pose = self.convert_array_to_pose(object_pose_array)
                if self.align_object_tf:
                    object_pose = align_obj_frame.align_obj_ort(object_pose, self.listener)
                print object_pose
                self.update_object_pose_client(object_pose)

            voxel_grid = np.zeros(tuple(self.voxel_grid_dim))
            voxel_grid_index = sparse_voxel_grid.astype(int)
            voxel_grid[voxel_grid_index[:, 0], voxel_grid_index[:, 1], voxel_grid_index[:, 2]] = 1
            print sparse_voxel_grid.shape
            print np.sum(voxel_grid)
            #print sparse_voxel_grid
            #print np.array(np.where(voxel_grid == 1.))
            #print voxel_grid
            grasp_voxel_grids.append(voxel_grid)
            grasp_configs.append(grasp_config)

            palm_pose_object = self.trans_pose_to_obj_tf(hand_config.palm_pose) 
            #print palm_pose_object
            #print grasp_config[self.palm_dof_dim:]
            grasp_config_obj = np.concatenate((palm_pose_object, grasp_config[self.palm_dof_dim:]), axis=0)
            print 'grasp_config_obj:', grasp_config_obj
            grasp_configs_obj.append(grasp_config_obj)

            #grasp_label_key = grasp_sample_id + '_grasp_label'
            #grasp_label = grasp_patches_file[grasp_label_key][()]
            grasp_labels.append(grasp_label)
            voxel_grasps_id += 1
            print '######'

        grasp_voxel_file.create_dataset('grasp_voxel_grids', data=grasp_voxel_grids)
        grasp_voxel_file.create_dataset('grasp_configs', data=grasp_configs)
        grasp_voxel_file.create_dataset('grasp_configs_obj', data=grasp_configs_obj)
        grasp_voxel_file.create_dataset('grasp_labels', data=grasp_labels)
        #print np.sum(grasp_voxel_file['grasp_voxel_grids'][()])

        print 'Grasp data processing is done.'
        grasp_voxel_file.close()
        grasp_patches_file.close() 
        grasp_data_file.close()
        grasp_to_pcd_file.close()

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
            #update_palm_pose_request.palm_pose_in_pcd.header.frame_id = 'blensor_camera'
            #update_palm_pose_request.palm_pose_in_pcd.pose = palm_pose.pose 
            update_palm_pose_request.palm_pose_in_pcd = palm_pose
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
            #update_object_pose_request.object_pose_world.header.frame_id = 'world'
            #update_object_pose_request.object_pose_world.pose = object_pose.pose 
            update_object_pose_request.object_pose_world = object_pose
            update_object_pose_response = update_object_pose_proxy(update_object_pose_request) 
        except rospy.ServiceException, e:
            rospy.loginfo('Service update_grasp_object_pose call failed: %s'%e)
        rospy.loginfo('Service update_grasp_object_pose is executed.')

if __name__ == "__main__":
    data_path = '/media/kai/multi_finger_sim_data_complete_v4/'
    proc_grasp_data = ProcGraspData(data_path)
    proc_grasp_data.gen_grasp_pcd_map()
    proc_grasp_data.proc_data()
    #for i in xrange(5):
    #    data_path = '/media/kai/multi_finger_sim_data_precision_' + str(i + 1) + '/'
    #    proc_grasp_data = ProcGraspData(data_path)
    #    proc_grasp_data.gen_grasp_pcd_map()
    #    proc_grasp_data.proc_data()

