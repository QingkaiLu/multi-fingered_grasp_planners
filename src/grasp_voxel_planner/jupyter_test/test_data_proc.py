#!/usr/bin/env python
# coding: utf-8

# In[1]:


import h5py 
import numpy as np
import os


# In[2]:


grasp_file_path = '/mnt/tars_data/multi_finger_sim_data_6_6/grasp_data.h5'
grasp_data_file = h5py.File(grasp_file_path, 'r')


# In[3]:


grasps_num = grasp_data_file['total_grasps_num'][()]
max_object_id = grasp_data_file['max_object_id'][()]
print grasps_num, max_object_id


# In[30]:


def find_pcd_file(data_path, grasp_data_file, object_id, grasp_id, object_name):
    pcd_file_name = object_id + '_' + object_name +     '_grasp_' + str(grasp_id) + '.pcd'
    pcd_file_path = data_path + 'pcd/' + pcd_file_name
    object_world_seg_pose_key = object_id + '_grasp_' + str(grasp_id) + '_object_world_seg_pose'
    object_world_seg_pose = grasp_data_file[object_world_seg_pose_key][()]
    
    #print pcd_file_path
    #print os.path.isfile(pcd_file_path)
    if not os.path.isfile(pcd_file_path):
        #print 'HAHA'
        pcd_file_name = None
        for i in xrange(1, 3):
            prev_pcd_file_name = object_id + '_' + object_name +                     '_grasp_' + str(grasp_id - i) + '.pcd' 
            prev_pcd_file_path = data_path + 'pcd/' + prev_pcd_file_name          
            #print prev_pcd_file_name

            if os.path.isfile(prev_pcd_file_path):
#                 grasp_pose_key = object_id + '_grasp_' + str(grasp_id - i)  + '_object_world_pose'
#                 grasp_pose_prev = grasp_data_file[grasp_pose_key][()]
                object_world_seg_pose_key = object_id + '_grasp_' + str(grasp_id - i) + '_object_world_seg_pose'
                object_world_pose_prev = grasp_data_file[object_world_seg_pose_key][()]
                #print grasp_pose_prev
                #if np.array_equal(grasp_pose_prev, grasp_pose):
                if np.allclose(object_world_pose_prev, object_world_seg_pose):
                    print 'object pose equal'
                    pcd_file_name = prev_pcd_file_name
                    break
                print 'object pose not equal'
                #print grasp_pose - grasp_pose_prev
    return pcd_file_name


# In[31]:


# For each grasp, extract the grasp config, grasp label, and generate voxel
# Also need to record the object id, grasp id, and object_name of each grasp.

# Notice: the max grasp id of one object could be 19 (0-19) + 3 = 22, because we try 
# 3 grasp heuristic grasp trials for each object pose.
max_grasps_per_obj = 23
total_grasps_num = 0
suc_grasps_num = 0
top_grasps_num = 0
top_suc_grasps_num = 0
save_visual_data_pre_path = '/mnt/tars_data/multi_finger_sim_data_6_6/'
for obj_id in xrange(max_object_id + 1):
    object_id = 'object_' + str(obj_id)
    object_name = grasp_data_file[object_id + '_name'][()]
    #print object_name
    for grasp_id in xrange(max_grasps_per_obj):
        #print grasp_id
        object_grasp_id = object_id + '_grasp_' + str(grasp_id)
        grasp_label_key = object_grasp_id + '_grasp_label'
        if grasp_label_key in grasp_data_file:
            grasp_label = grasp_data_file[grasp_label_key][()]
            #print grasp_label
            total_grasps_num += 1
            suc_grasps_num += grasp_label
            object_world_seg_pose_key = object_grasp_id + '_object_world_seg_pose'
            object_world_seg_pose = grasp_data_file[object_world_seg_pose_key][()]
            preshape_js_position_key = object_grasp_id + '_preshape_joint_state_position'
            preshape_js = grasp_data_file[preshape_js_position_key][()]
            palm_world_pose_key = object_grasp_id + '_preshape_palm_world_pose'
            palm_world_pose = grasp_data_file[palm_world_pose_key][()]
            top_grasp_key = object_grasp_id + '_top_grasp'
            top_grasp = grasp_data_file[top_grasp_key][()]
            top_grasps_num += top_grasp
            if grasp_label == 1 and top_grasp:
                top_suc_grasps_num += 1
            #print object_world_seg_pose, preshape_js, palm_world_pose
            
#             pcd_file_path = save_visual_data_pre_path + \
#             'pcd/' + 'object_' + str(obj_id) + '_' + str(object_name) + \
#             '_grasp_' + str(grasp_id) + '.pcd'
#             print pcd_file_path
#             print os.path.isfile(pcd_file_path)
            pcd_file_path_found = find_pcd_file(save_visual_data_pre_path, grasp_data_file,
                                                object_id, grasp_id, object_name)
            print pcd_file_path_found
            #print grasp_id
#             if pcd_file_path_found is None:
#                 break
    #break
#     if pcd_file_path_found is None:
#         break
            

print total_grasps_num, suc_grasps_num, top_grasps_num, top_suc_grasps_num


# In[ ]:




