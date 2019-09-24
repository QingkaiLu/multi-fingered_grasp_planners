#!/usr/bin/env python
import roslib
roslib.load_manifest('prob_grasp_planner')
import rospy
from geometry_msgs.msg import Pose, Quaternion, PoseStamped, PointStamped
from prob_grasp_planner.srv import *
import numpy as np
import time
import cv2
import h5py
import show_voxel
import tf
import os
import proc_grasp_data as pgd
import align_object_frame as align_obj_frame

class GenPrecPowData:
    def __init__(self, grasp_type, data_path):
        #rospy.init_node('gen_prec_pow_data') 
        #self.grasp_type = 'prec'
        #self.grasp_type = 'power'
        self.grasp_type = grasp_type
        #self.data_path = '/media/kai/multi_finger_sim_data_complete_v4/'
        #self.data_path = '/media/kai/multi_finger_sim_data_precision_1/'
        self.data_path = data_path
        self.grasp_patches_file_path = self.data_path + 'grasp_patches.h5'
        self.grasp_data_file_path = self.data_path + 'grasp_data.h5'
        self.align_object_tf = True
        if self.align_object_tf:
            self.grasp_voxel_file_path = self.data_path + self.grasp_type + \
                                        '_grasps/' + self.grasp_type + '_align_suc_grasps.h5'
        else:
            self.grasp_voxel_file_path = self.data_path + self.grasp_type + \
                                        '_grasps/grasp_voxel_obj_tf_data.h5'
        self.prec_power_grasp_path = self.data_path + self.grasp_type + \
                                    '_grasps/' + self.grasp_type + '_grasp_ids'
        self.proc_grasp = pgd.ProcGraspData('')

        self.voxel_grid_dim = [20, 20, 20]
        self.voxel_size = [0.01, 0.01, 0.01]
        self.config_dim = 14
        self.classes_num = 1
        self.palm_loc_dof_dim = 3
        self.palm_dof_dim = 6
        self.hand_config_frame_id = 'blensor_camera'
        self.object_frame_id = 'object_pose'
        self.tf_br = tf.TransformBroadcaster()
        self.listener = tf.TransformListener()

    def read_grasp_ids(self):
        grasp_ids = [line.rstrip('\n') for line in open(self.prec_power_grasp_path)]
        return grasp_ids
    
    def map_grasp_obj_info_to_grasp_ids(self):
        grasp_obj_info = set([line.rstrip('\n') for line in open(self.prec_power_grasp_path)])
        #object_13_coffee_mate_french_vanilla_grasp_3

        grasp_patches_file = h5py.File(self.grasp_patches_file_path, 'r')
        grasp_data_file = h5py.File(self.grasp_data_file_path, 'r')
        pcd_to_grasp_file = h5py.File(self.data_path + self.grasp_type + '_grasps/pcd_to_grasp.h5', 'w')

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
            
            #pcd_file_name = 'object_' + object_id + '_' + object_name + \
            #        '_grasp_' + str(grasp_id) + '00000.pcd' 
            grasp_obj = 'object_' + object_id + '_' + object_name + \
                    '_grasp_' + str(grasp_id) 
            if grasp_obj not in grasp_obj_info:
                continue

            print grasp_obj
            print grasp_obj_info

            pcd_file_name = grasp_obj + '00000.pcd' 
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
                        if np.allclose(grasp_pose_prev, grasp_pose):
                            print 'object pose equal'
                            pcd_file_name = prev_pcd_file_name
                            break
                        print 'object pose not equal'
                        print grasp_pose - grasp_pose_prev

            #print pcd_file_name
            if pcd_file_name is not None:
                pcd_to_grasp_file.create_dataset(grasp_obj + '_gid', data=grasp_sample_id)
                pcd_to_grasp_file.create_dataset(grasp_obj + '_pcd', data=pcd_file_name)
                #ISSUE: pcd_file_name could be the same with previous grasps' pcds
                print pcd_file_name
            print grasp_sample_id in pcd_to_grasp_file
            print '##############'

        pcd_to_grasp_file.close()
 
    def proc_data(self):
        '''
        Process grasp data to get grasp voxel data.
        '''
        grasp_obj_info = [line.rstrip('\n') for line in open(self.prec_power_grasp_path)]

        grasp_patches_file = h5py.File(self.grasp_patches_file_path, 'r')
        grasp_data_file = h5py.File(self.grasp_data_file_path, 'r')
        grasp_voxel_file = h5py.File(self.grasp_voxel_file_path, 'a')
        pcd_to_grasp_file = h5py.File(self.data_path + self.grasp_type + '_grasps/pcd_to_grasp.h5', 'r')

        grasps_num = grasp_patches_file['grasps_number'][()] 
        grasp_configs = []
        grasp_configs_obj = []
        grasp_labels = []
        grasp_voxel_grids = []

        voxel_grasps_id = 0
        for grasp_obj in grasp_obj_info:
            pcd_file_name = pcd_to_grasp_file[grasp_obj + '_pcd'][()] 
            grasp_sample_id = pcd_to_grasp_file[grasp_obj + '_gid'][()]
            grasp_config_key = grasp_sample_id + '_preshape_true_config'
            grasp_config = grasp_patches_file[grasp_config_key]
            object_grasp_id_key = grasp_sample_id + '_object_grasp_id'
            object_grasp_id = grasp_patches_file[object_grasp_id_key][()]
            object_pose_key = object_grasp_id + '_object_world_pose'
            #Skip grasps without object world poses.
            if object_pose_key in grasp_data_file:
                object_pose_array = grasp_data_file[object_pose_key]
            else:
                continue

            voxel_grasp_id_key = 'voxel_grasp_' + str(voxel_grasps_id)
            print voxel_grasp_id_key
            sparse_voxel_key = voxel_grasp_id_key + '_sparse_voxel'
            if voxel_grasp_id_key not in grasp_voxel_file:
                ## Resume from previous voxel grids generation.
                #if grasp_sample_id in pcd_to_grasp_file:
                #    pcd_file_name = pcd_to_grasp_file[grasp_sample_id][()]
                #    print pcd_file_name
                #else:
                #    print '***pcd_file does not exist!'
                #    continue
                pcd_file_path = self.data_path + 'pcd/' + pcd_file_name          
   
                hand_config = self.proc_grasp.convert_preshape_to_full_config(grasp_config) 
                self.proc_grasp.update_palm_pose_client(hand_config.palm_pose)

                print 'object_pose_array:', object_pose_array[()]
                object_pose = self.proc_grasp.convert_array_to_pose(object_pose_array)
                print 'object_pose:', object_pose
                #Update the object pose without alignment for debugging 
                #self.proc_grasp.update_object_pose_client(object_pose)
                #raw_input('Continue to aligned object pose.')

                if self.align_object_tf:
                    object_pose = align_obj_frame.align_obj_ort(object_pose, self.listener)
                self.proc_grasp.update_object_pose_client(object_pose)

                sparse_voxel_grid = self.proc_grasp.voxel_gen_client(pcd_file_path)
                
                #if i % 10 == 0:
                #    show_voxel.plot_voxel(sparse_voxel_grid, './voxel.png') 
                #show_voxel.plot_voxel(sparse_voxel_grid, './voxel.png') 

                grasp_voxel_file.create_dataset(voxel_grasp_id_key, data=grasp_sample_id)
                grasp_voxel_file.create_dataset(sparse_voxel_key, data=sparse_voxel_grid)
            else:
                sparse_voxel_grid = grasp_voxel_file[sparse_voxel_key][()]

                hand_config = self.proc_grasp.convert_preshape_to_full_config(grasp_config) 
                print hand_config.palm_pose
                self.proc_grasp.update_palm_pose_client(hand_config.palm_pose)

                object_pose = self.proc_grasp.convert_array_to_pose(object_pose_array)
                if self.align_object_tf:
                    object_pose = align_obj_frame.align_obj_ort(object_pose, self.listener)
                self.proc_grasp.update_object_pose_client(object_pose)

            voxel_grid = np.zeros(tuple(self.voxel_grid_dim))
            voxel_grid_index = sparse_voxel_grid.astype(int)
            voxel_grid[voxel_grid_index[:, 0], voxel_grid_index[:, 1], voxel_grid_index[:, 2]] = 1

            print sparse_voxel_grid.shape
            print np.sum(voxel_grid)

            grasp_voxel_grids.append(voxel_grid)
            grasp_configs.append(grasp_config)

            palm_pose_object = self.proc_grasp.trans_pose_to_obj_tf(hand_config.palm_pose) 
            grasp_config_obj = np.concatenate((palm_pose_object, grasp_config[self.palm_dof_dim:]), axis=0)
            print 'grasp_config_obj:', grasp_config_obj
            grasp_configs_obj.append(grasp_config_obj)

            grasp_label_key = grasp_sample_id + '_grasp_label'
            grasp_label = grasp_patches_file[grasp_label_key][()]
            grasp_labels.append(grasp_label)
            voxel_grasps_id += 1
            print '######'

        grasp_voxel_file.create_dataset('grasp_voxel_grids', data=grasp_voxel_grids)
        grasp_voxel_file.create_dataset('grasp_configs', data=grasp_configs)
        grasp_voxel_file.create_dataset('grasp_configs_obj', data=grasp_configs_obj)
        grasp_voxel_file.create_dataset('grasp_labels', data=grasp_labels)

        print 'Grasp data processing is done.'
        grasp_voxel_file.close()
        grasp_patches_file.close() 
        grasp_data_file.close()
        pcd_to_grasp_file.close()


if __name__ == '__main__':
    #data_path = '/media/kai/multi_finger_sim_data_complete_v4/'
    #grasp_type = 'power'
    #gen_data = GenPrecPowData(grasp_type, data_path)
    #gen_data.map_grasp_obj_info_to_grasp_ids()
    #gen_data.proc_data()

    for i in xrange(5):
        if i != 2:
            continue
        data_path = '/media/kai/multi_finger_sim_data_precision_' + str(i + 1) + '/'
        grasp_type = 'prec'
        gen_data = GenPrecPowData(grasp_type, data_path)
        gen_data.map_grasp_obj_info_to_grasp_ids()
        gen_data.proc_data()

