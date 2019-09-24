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
# from data_proc_lib import DataProcLib 
import os
from geometry_msgs.msg import Pose, Quaternion, PoseStamped, PointStamped
from gen_rgbd_gazebo_kinect import GenRgbdGazeboKinect
from prob_grasp_planner.srv import *


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
            print prev_pcd_file_name

            if os.path.isfile(prev_pcd_file_path):
                object_world_seg_pose_key = object_id + '_grasp_' + str(grasp_id_cur_obj - i) + '_object_world_seg_pose'
                object_world_pose_prev = grasp_data_file[object_world_seg_pose_key][()]
                if np.allclose(object_world_pose_prev, object_world_seg_pose):
                    print 'object pose equal'
                    pcd_file_name = prev_pcd_file_name
                    break
                print 'object pose not equal'
    return pcd_file_name


def get_pcd_normal_client(pcd_file_path):
    rospy.loginfo('Waiting for service compute_pcd_normal.')
    rospy.wait_for_service('compute_pcd_normal')
    rospy.loginfo('Calling service compute_pcd_normal.')
    try:
        compute_normal_proxy = rospy.ServiceProxy('compute_pcd_normal', GetPcdNormal)
        compute_normal_request = GetPcdNormalRequest()
        compute_normal_request.pcd_file_path = pcd_file_path

        compute_normal_response = compute_normal_proxy(compute_normal_request) 
        return compute_normal_response
    except rospy.ServiceException, e:
        rospy.loginfo('Service compute_pcd_normal call failed: %s'%e)
    rospy.loginfo('Service compute_pcd_normal is executed.')


def proc_grasp_rgbd_data(raw_data_path, grasp_rgbd_file_path):
     grasp_file_path = raw_data_path + 'grasp_data.h5'
     grasp_data_file = h5py.File(grasp_file_path, 'r')
 
     gen_rgbd = GenRgbdGazeboKinect() 
 
     grasp_rgbd_file = h5py.File(grasp_rgbd_file_path, 'a')
     rgbd_number_key = 'rgbd_number'
     if rgbd_number_key not in grasp_rgbd_file:
         grasp_rgbd_file.create_dataset(rgbd_number_key, data=0)

     grasp_rgbd_file.close()

     cvtf = gcf.ConfigConvertFunctions()
 
     grasps_num = grasp_data_file['total_grasps_num'][()]
     max_object_id = grasp_data_file['max_object_id'][()]
     print grasps_num, max_object_id

     # Notice: the max grasp id of one object could be 19 (0-19) + 3 = 22, because we try 
     # 3 grasp heuristic grasp trials for each object pose.
     max_grasps_per_obj = 23
     cur_grasps_num = 0
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
                 grasp_rgbd_file = h5py.File(grasp_rgbd_file_path, 'a')

                 # Recover from break
                 if cur_grasps_num < grasp_rgbd_file[rgbd_number_key][()]:
                    print 'Skip ', cur_grasps_num
                    cur_grasps_num += 1
                    continue
                 grasp_label = grasp_data_file[grasp_label_key][()]
                 pcd_file_name = find_pcd_file(raw_data_path, grasp_data_file,
                                                     object_id, grasp_id_cur_obj, object_name)
                 pcd_file_path = raw_data_path + 'pcd/' + pcd_file_name
                 print pcd_file_path
                 image_name = pcd_file_name[:-3] + 'png'
                 rgb_path = raw_data_path + 'rgb_image/' + image_name
                 normal_resp = get_pcd_normal_client(pcd_file_path) 

                 # print normal_resp
                 # rgbd = gen_rgbd.get_rgbd_image(rgb_path, depth_path, normal_resp.pcd_normal)
                 rgbd = gen_rgbd.get_rgbd_image(rgb_path, normal_resp.point_cloud, normal_resp.pcd_normal)
                 grasp_rgbd_patch = gen_rgbd.extract_rgbd_patch(rgbd, (320.0, 240.0), patch_size=400)  
                 # gen_rgbd.save_rgbd_image(grasp_rgbd_patch, '/home/qingkai/Workspace/grasp_rgbd')
                 # raw_input('HOLD on')
                 grasp_rgbd_patch_key = object_grasp_id + '_rgbd_patch'
                 grasp_rgbd_file.create_dataset(grasp_rgbd_patch_key, data=grasp_rgbd_patch)

                 cur_grasps_num += 1
                 grasp_rgbd_file[rgbd_number_key][()] = cur_grasps_num
                 grasp_rgbd_file.close() 
             else:
                 print 'Grasp label does not exist!'
 
     grasp_data_file.close()
     #grasp_rgbd_file.close()


# def mp_grasp_rgbd_data(raw_data_path, grasp_rgbd_file_path):
#     grasp_file_path = raw_data_path + 'grasp_data.h5'
#     grasp_data_file = h5py.File(grasp_file_path, 'r')
# 
#     voxel_file_path = raw_data_path + 'proc_grasp_data.h5'
#     voxel_data_file = h5py.File(voxel_file_path, 'r')
# 
#     gen_rgbd = GenRgbdGazeboKinect() 
#     cvtf = gcf.ConfigConvertFunctions()
# 
#     grasp_rgbd_file = h5py.File(grasp_rgbd_file_path, 'a')
#     grasps_number_key = 'grasps_number'
#     grasps_num = voxel_data_file['grasps_number'][()]
# 
#     for g_num in xrange(grasps_num):
#         print g_num
#         source_info_key = 'grasp_' + str(g_num) + '_source_info' 
#         source_info = voxel_data_file[source_info_key][()]
#         info_split = source_info.split('_')
#         print info_split
#         object_id = info_split[0] + '_' + info_split[1]
#         grasp_id = info_split[3]
#         object_grasp_id = object_id + '_grasp_' + grasp_id
#         print object_grasp_id
#         object_name = source_info[len(object_grasp_id) - 1:]
#         print object_name
#         raw_input('Hold')
# 
#         # pcd_file_name = object_id + '_' + object_name + \
#         #                 '_grasp_' + str(grasp_id) + '.pcd' 
# 
#         pcd_file_name = find_pcd_file(raw_data_path, grasp_data_file,
#                                             object_id, int(grasp_id), object_name)
# 
#         pcd_file_path = raw_data_path + 'pcd/' + pcd_file_name
#         print pcd_file_path
# 
#         image_name = pcd_file_name[:-3] + 'png'
#         rgb_path = raw_data_path + 'rgb_image/' + image_name
# 
# 
#     grasp_data_file.close()
#     voxel_data_file.close()
#     grasp_rgbd_file.close()


def merge_grasp_rgbd_datasets(datasets_path, data_folders, 
                        merged_file_name, rgbd_datasets_path, only_success=False):
    merged_file_path = rgbd_datasets_path + merged_file_name 
    merged_data_file = h5py.File(merged_file_path, 'w')
    grasps_number_key = 'grasps_number'
    if grasps_number_key not in merged_data_file:
        merged_data_file.create_dataset(grasps_number_key, data=0)

    cur_grasps_num = 0
    for data_folder in data_folders:
        grasp_voxel_file_path = datasets_path + data_folder + '/proc_grasp_data.h5'
        grasp_voxel_file = h5py.File(grasp_voxel_file_path, 'r')
        rgbd_file_path = rgbd_datasets_path + data_folder + '_rgbd.h5' 
        grasp_rgbd_file = h5py.File(rgbd_file_path, 'r')

        grasps_num = grasp_voxel_file['grasps_number'][()]
        print data_folder

        for g_num in xrange(grasps_num):
            print g_num
            local_label_key = 'grasp_' + str(g_num) + '_label'
            global_label_key = 'grasp_' + str(cur_grasps_num) + '_label'
            if only_success:
                print grasp_voxel_file[local_label_key][()]
                if grasp_voxel_file[local_label_key][()] == 0:
                    continue
            merged_data_file.create_dataset(global_label_key, 
                                    data=grasp_voxel_file[local_label_key][()])
            local_source_info_key = 'grasp_' + str(g_num) + '_source_info' 
            global_source_info_key = 'grasp_' + str(cur_grasps_num) + '_source_info' 
            merged_data_file.create_dataset(global_source_info_key, 
                                    data=grasp_voxel_file[local_source_info_key][()])
            local_config_obj_key = 'grasp_' + str(g_num) + '_config_obj'
            global_config_obj_key = 'grasp_' + str(cur_grasps_num) + '_config_obj'
            merged_data_file.create_dataset(global_config_obj_key, 
                                    data=grasp_voxel_file[local_config_obj_key][()])

            source_info_key = 'grasp_' + str(g_num) + '_source_info' 
            source_info = grasp_voxel_file[source_info_key][()]
            info_split = source_info.split('_')
            #print info_split
            object_id = info_split[0] + '_' + info_split[1]
            grasp_id = info_split[3]
            object_grasp_id = object_id + '_grasp_' + grasp_id
            print object_grasp_id

            # local_rgbd_patch_key = 'grasp_' + str(g_num) + '_grasp_rgbd_patch'
            local_rgbd_patch_key = object_grasp_id + '_rgbd_patch'
            global_rgbd_patch_key = 'grasp_' + str(cur_grasps_num) + '_rgbd_patch'
            merged_data_file.create_dataset(global_rgbd_patch_key, 
                                        data=grasp_rgbd_file[local_rgbd_patch_key][()])
            # local_voxel_key = 'grasp_' + str(g_num) + '_sparse_voxel'
            # global_voxel_key = 'grasp_' + str(cur_grasps_num) + '_sparse_voxel'
            # merged_data_file.create_dataset(global_voxel_key, 
            #                         data=grasp_voxel_file[local_voxel_key][()])
            # local_obj_dim_key = 'grasp_' + str(g_num) + '_dim_w_h_d'
            # global_obj_dim_key = 'grasp_' + str(cur_grasps_num) + '_dim_w_h_d'
            # merged_data_file.create_dataset(global_obj_dim_key, 
            #                         data=grasp_voxel_file[local_obj_dim_key][()])
            local_top_grasp_key = 'grasp_' + str(g_num) + '_top_grasp'
            global_top_grasp_key = 'grasp_' + str(cur_grasps_num) + '_top_grasp'
            merged_data_file.create_dataset(global_top_grasp_key, 
                                    data=grasp_voxel_file[local_top_grasp_key][()])
            data_folder_key = 'grasp_' + str(cur_grasps_num) + '_data_folder'
            merged_data_file.create_dataset(data_folder_key, data=data_folder)

            cur_grasps_num += 1
            merged_data_file[grasps_number_key][()] = cur_grasps_num

        grasp_voxel_file.close()
        grasp_rgbd_file.close()

    merged_data_file.close()


if __name__ == '__main__':
    # datasets_path = '/mnt/tars_data/gazebo_al_grasps/train/'
    # rgbd_datasets_path = '/mnt/tars_data/gazebo_al_grasps/train_isrr/'
    datasets_path = '/mnt/tars_data/gazebo_al_grasps/test/'
    rgbd_datasets_path = '/mnt/tars_data/gazebo_al_grasps/test_isrr/'
    is_proc = False #True
    if is_proc:
        data_folders = os.listdir(datasets_path)
        for i, data_folder in enumerate(data_folders):
            raw_data_path = datasets_path + data_folder + '/'
            proc_file_path = rgbd_datasets_path + data_folder + '_rgbd.h5' 
            print raw_data_path
            print proc_file_path
            proc_grasp_rgbd_data(raw_data_path, proc_file_path)
    else:
        # Merge
        data_folders = os.listdir(datasets_path)
        # data_folders = data_folders[:2]
        print data_folders
        #merged_file_name = 'merged_grasp_rgbd_10_sets.h5'
        merged_file_name = 'merged_grasp_rgbd_test_sets.h5'
        merge_grasp_rgbd_datasets(datasets_path, data_folders, merged_file_name, rgbd_datasets_path)





