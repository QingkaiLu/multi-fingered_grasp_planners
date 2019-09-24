#!/usr/bin/env python
import roslib
roslib.load_manifest('prob_grasp_planner')
import rospy
import numpy as np
import h5py
import roslib.packages as rp
import sys
pkg_path = rp.get_pkg_dir('prob_grasp_planner')
sys.path.append(pkg_path + '/src/grasp_common_library')
import grasp_common_functions as gcf
import show_voxel
import tf
import glob
from data_proc_lib import DataProcLib 
import os
from geometry_msgs.msg import Pose, Quaternion, PoseStamped, PointStamped
from point_cloud_segmentation.srv import *


def find_pcd_file(data_path, grasp_data_file, object_id, grasp_id_cur_obj, object_name):
    pcd_file_name = object_id + '_' + object_name + '_grasp_' + str(grasp_id_cur_obj) + '.pcd'
    pcd_file_path = data_path + 'pcd/' + pcd_file_name
    object_world_seg_pose_key = object_id + '_grasp_' + str(grasp_id_cur_obj) + '_object_world_seg_pose'
    object_world_seg_pose = grasp_data_file[object_world_seg_pose_key][()]
    
    if not os.path.isfile(pcd_file_path):
        pcd_file_name = None
        for i in xrange(1, 3):
            prev_pcd_file_name = object_id + '_' + object_name + \
                                '_grasp_' + str(grasp_id_cur_obj - i) + '.pcd' 
            prev_pcd_file_path = data_path + 'pcd/' + prev_pcd_file_name          

            if os.path.isfile(prev_pcd_file_path):
                object_world_seg_pose_key = object_id + '_grasp_' + str(grasp_id_cur_obj - i) + '_object_world_seg_pose'
                object_world_pose_prev = grasp_data_file[object_world_seg_pose_key][()]
                if np.allclose(object_world_pose_prev, object_world_seg_pose):
                    print 'object pose equal'
                    pcd_file_name = prev_pcd_file_name
                    break
                print 'object pose not equal'
    return pcd_file_name


def proc_grasp_data(raw_data_path, proc_data_path):
    grasp_file_path = raw_data_path + 'grasp_data.h5'
    grasp_data_file = h5py.File(grasp_file_path, 'r')
    proc_grasp_file_path = proc_data_path + 'grasp_rgbd_data.h5'
    #proc_grasp_file_path = proc_data_path + 'proc_grasp_data_2.h5'
    proc_grasp_file = h5py.File(proc_grasp_file_path, 'a')
    grasps_number_key = 'grasps_number'
    if grasps_number_key not in proc_grasp_file:
        proc_grasp_file.create_dataset(grasps_number_key, data=0)
    grasp_no_obj_num_key = 'grasp_no_obj_num'
    if grasp_no_obj_num_key not in proc_grasp_file:
        proc_grasp_file.create_dataset(grasp_no_obj_num_key, data=0)

    cvtf = gcf.ConfigConvertFunctions()

    grasps_num = grasp_data_file['total_grasps_num'][()]
    grasps_no_seg_num = proc_grasp_file[grasp_no_obj_num_key][()]
    max_object_id = grasp_data_file['max_object_id'][()]
    print grasps_num, max_object_id

    data_proc_lib = DataProcLib()
    # Notice: the max grasp id of one object could be 19 (0-19) + 3 = 22, because we try 
    # 3 grasp heuristic grasp trials for each object pose.
    max_grasps_per_obj = 23
    cur_grasps_num = 0
    cur_grasps_no_seg_num = 0
    #suc_grasps_num = 0
    #top_grasps_num = 0
    #top_suc_grasps_num = 0
    for obj_id in xrange(max_object_id + 1):
        object_id = 'object_' + str(obj_id)
        object_name = grasp_data_file[object_id + '_name'][()]
        #print object_name
        for grasp_id_cur_obj in xrange(max_grasps_per_obj):
            #print grasp_id_cur_obj
            print cur_grasps_num
            object_grasp_id = object_id + '_grasp_' + str(grasp_id_cur_obj)
            print object_grasp_id

            grasp_label_key = object_grasp_id + '_grasp_label'
            if grasp_label_key in grasp_data_file:
                # Only check the grasp already exists if it's a valid grasp.
                grasp_source_info_key = 'grasp_' + str(cur_grasps_num) + '_source_info' 
                if grasp_source_info_key in proc_grasp_file:
                # Skip processed grasps from previous runs.
                    if cur_grasps_no_seg_num < grasps_no_seg_num:
                        cur_grasps_no_seg_num += 1
                    else:
                        cur_grasps_num += 1
                    continue

                grasp_label = grasp_data_file[grasp_label_key][()]
                # Relabel the grasp labels to overcome grasp control errors

                object_world_seg_pose_key = object_grasp_id + '_object_world_seg_pose'
                object_world_seg_pose_array = grasp_data_file[object_world_seg_pose_key][()]
                preshape_js_position_key = object_grasp_id + '_preshape_joint_state_position'
                preshape_js = grasp_data_file[preshape_js_position_key][()]
                palm_world_pose_key = object_grasp_id + '_preshape_palm_world_pose'
                palm_world_pose_array = grasp_data_file[palm_world_pose_key][()]
                top_grasp_key = object_grasp_id + '_top_grasp'
                top_grasp = grasp_data_file[top_grasp_key][()]
                pcd_file_name = find_pcd_file(data_path, grasp_data_file,
                                                    object_id, grasp_id_cur_obj, object_name)
                pcd_file_path = data_path + 'pcd/' + pcd_file_name
                print pcd_file_path
                seg_obj_resp = gcf.seg_obj_from_file_client(pcd_file_path, data_proc_lib.listener)
                if not seg_obj_resp.object_found:
                    print 'No object found for segmentation!'
                    cur_grasps_no_seg_num += 1
                    proc_grasp_file[grasp_no_obj_num_key][()] = cur_grasps_no_seg_num
                    continue
                
                palm_world_pose = cvtf.convert_array_to_pose(palm_world_pose_array, 'world')
                data_proc_lib.update_palm_pose_client(palm_world_pose)

                #object_world_pose = cvtf.convert_array_to_pose(object_world_seg_pose_array, 'world')
                #data_proc_lib.update_object_pose_client(object_world_pose)

                obj_world_pose_stamp = PoseStamped()
                obj_world_pose_stamp.header.frame_id = seg_obj_resp.obj.header.frame_id
                obj_world_pose_stamp.pose = seg_obj_resp.obj.pose
                data_proc_lib.update_object_pose_client(obj_world_pose_stamp)

                sparse_voxel_grid = data_proc_lib.voxel_gen_client(seg_obj_resp.obj)

                #show_voxel.plot_voxel(sparse_voxel_grid) 

                #voxel_grid = np.zeros(tuple(data_proc_lib.voxel_grid_full_dim))
                #voxel_grid_index = sparse_voxel_grid.astype(int)
                #voxel_grid[voxel_grid_index[:, 0], voxel_grid_index[:, 1], voxel_grid_index[:, 2]] = 1

                # print sparse_voxel_grid.shape
                # print np.sum(voxel_grid)
                # print sparse_voxel_grid
                # print np.array(np.where(voxel_grid == 1.))
                # print voxel_grid
 
                palm_object_pose_array = data_proc_lib.trans_pose_to_obj_tf(palm_world_pose) 
                grasp_config_obj = np.concatenate((palm_object_pose_array, preshape_js), axis=0)
                #print 'grasp_config_obj:', grasp_config_obj

                grasp_source_info = object_grasp_id + '_' + object_name
                proc_grasp_file.create_dataset(grasp_source_info_key, data=grasp_source_info)
                grasp_config_obj_key = 'grasp_' + str(cur_grasps_num) + '_config_obj'
                proc_grasp_file.create_dataset(grasp_config_obj_key, data=grasp_config_obj)
                grasp_sparse_voxel_key = 'grasp_' + str(cur_grasps_num) + '_sparse_voxel'
                proc_grasp_file.create_dataset(grasp_sparse_voxel_key, data=sparse_voxel_grid)
                obj_dim_w_h_d = np.array([seg_obj_resp.obj.width, seg_obj_resp.obj.height, 
                                            seg_obj_resp.obj.depth])
                obj_dim_key = 'grasp_' + str(cur_grasps_num) + '_dim_w_h_d'
                proc_grasp_file.create_dataset(obj_dim_key, data=obj_dim_w_h_d)
                grasp_label_key = 'grasp_' + str(cur_grasps_num) + '_label'
                proc_grasp_file.create_dataset(grasp_label_key, data=grasp_label)
                grasp_top_grasp_key = 'grasp_' + str(cur_grasps_num) + '_top_grasp'
                proc_grasp_file.create_dataset(grasp_top_grasp_key, data=top_grasp)

                cur_grasps_num += 1
                proc_grasp_file[grasps_number_key][()] = cur_grasps_num

            else:
                print 'Grasp label does not exist!'

                #proc_grasp_file.close()
                #raw_input('Hold')
                
                #suc_grasps_num += grasp_label
                #top_grasps_num += top_grasp
                #if grasp_label == 1 and top_grasp:
                #    top_suc_grasps_num += 1

    #print cur_grasps_num, suc_grasps_num, top_grasps_num, top_suc_grasps_num
    grasp_data_file.close()
    proc_grasp_file.close()


def merge_grasp_datasets(datasets_path, data_folders, 
                        merged_file_name, only_success=False):
    merged_file_path = datasets_path + merged_file_name 
    merged_data_file = h5py.File(merged_file_path, 'w')
    grasps_number_key = 'grasps_number'
    if grasps_number_key not in merged_data_file:
        merged_data_file.create_dataset(grasps_number_key, data=0)

    cur_grasps_num = 0
    for data_folder in data_folders:
        proc_grasp_file_path = datasets_path + data_folder + '/proc_grasp_data.h5'
        proc_grasp_file = h5py.File(proc_grasp_file_path, 'r')
        grasps_num = proc_grasp_file['grasps_number'][()]
        print data_folder

        for g_num in xrange(grasps_num):
            print g_num
            local_label_key = 'grasp_' + str(g_num) + '_label'
            global_label_key = 'grasp_' + str(cur_grasps_num) + '_label'
            if only_success:
                print proc_grasp_file[local_label_key][()]
                if proc_grasp_file[local_label_key][()] == 0:
                    continue
            merged_data_file.create_dataset(global_label_key, 
                                    data=proc_grasp_file[local_label_key][()])
            local_source_info_key = 'grasp_' + str(g_num) + '_source_info' 
            global_source_info_key = 'grasp_' + str(cur_grasps_num) + '_source_info' 
            merged_data_file.create_dataset(global_source_info_key, 
                                    data=proc_grasp_file[local_source_info_key][()])
            local_config_obj_key = 'grasp_' + str(g_num) + '_config_obj'
            global_config_obj_key = 'grasp_' + str(cur_grasps_num) + '_config_obj'
            merged_data_file.create_dataset(global_config_obj_key, 
                                    data=proc_grasp_file[local_config_obj_key][()])
            local_voxel_key = 'grasp_' + str(g_num) + '_sparse_voxel'
            global_voxel_key = 'grasp_' + str(cur_grasps_num) + '_sparse_voxel'
            merged_data_file.create_dataset(global_voxel_key, 
                                    data=proc_grasp_file[local_voxel_key][()])
            local_obj_dim_key = 'grasp_' + str(g_num) + '_dim_w_h_d'
            global_obj_dim_key = 'grasp_' + str(cur_grasps_num) + '_dim_w_h_d'
            merged_data_file.create_dataset(global_obj_dim_key, 
                                    data=proc_grasp_file[local_obj_dim_key][()])
            local_top_grasp_key = 'grasp_' + str(g_num) + '_top_grasp'
            global_top_grasp_key = 'grasp_' + str(cur_grasps_num) + '_top_grasp'
            merged_data_file.create_dataset(global_top_grasp_key, 
                                    data=proc_grasp_file[local_top_grasp_key][()])
            data_folder_key = 'grasp_' + str(cur_grasps_num) + '_data_folder'
            merged_data_file.create_dataset(data_folder_key, data=data_folder)

            cur_grasps_num += 1
            merged_data_file[grasps_number_key][()] = cur_grasps_num

        proc_grasp_file.close()

    merged_data_file.close()


if __name__ == '__main__':
    # datasets_path = '/mnt/tars_data/gazebo_al_grasps/train/'
    datasets_path = '/mnt/tars_data/gazebo_al_grasps/test/'
    data_folders = os.listdir(datasets_path)
    for data_folder in data_folders:
        data_path = datasets_path + data_folder + '/'
        print data_path
        # proc_grasp_data(data_path)

    # data_path = '/mnt/tars_data/gazebo_train_grasps/multi_finger_sim_data_6_6/'
    # data_path = '/mnt/tars_data/gazebo_al_grasps/multi_finger_sim_data_6_26/'
    # proc_grasp_data(data_path)

    # merged_file_name = 'merged_grasp_data.h5'
    # data_folders = ['multi_finger_sim_data_6_6', 'multi_finger_sim_data_6_8']
    # merged_file_name = 'merged_grasp_data_6_6_and_6_8.h5'
    # data_folders = ['multi_finger_sim_data_6_6', 'multi_finger_sim_data_6_8',
    #                 'multi_finger_sim_data_6_10', 'multi_finger_sim_data_6_11']
    # merged_file_name = 'merged_grasp_data_6_6_and_6_8_and_6_10_and_6_11.h5'
    # data_folders = ['multi_finger_sim_data_6_14', 'multi_finger_sim_data_6_16']
    # merged_file_name = 'merged_grasp_data_6_14_and_6_16.h5'
    #print data_folders
    # data_folders = ['multi_finger_sim_data_6_6', 'multi_finger_sim_data_6_8',
    #                 'multi_finger_sim_data_6_10', 'multi_finger_sim_data_6_11',
    #                 'multi_finger_sim_data_6_13']
    # merged_file_name = 'merged_grasp_data_6_6_and_6_8_and_6_10_and_6_11_and_6_13.h5'

    # merged_file_name = 'merged_grasp_data.h5'
    # data_folders = ['multi_finger_sim_data_6_16', 'multi_finger_sim_data_6_18']
    # merged_file_name = 'merged_grasp_data_6_16_and_6_18.h5'
    
    # merged_file_name = 'merged_grasp_data_10_sets.h5'
    # merge_grasp_datasets(datasets_path, data_folders, merged_file_name)

    # merged_file_name = 'merged_suc_grasp_10_sets.h5'
    # merged_file_name = 'merged_suc_grasp_6_14_and_6_16.h5'
    # merge_grasp_datasets(datasets_path, data_folders, merged_file_name, only_success=True)



