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
import show_voxel
import tf
import align_object_frame as align_obj_frame
import glob
from data_proc_lib import DataProcLib 


def proc_grasp_queries(grasp_data_path):
    '''
    Process grasp queries to get grasp voxel data.
    '''
    # TODO: testing.
    data_proc_lib = DataProcLib()

    objects_per_batch = 2
    grasps_per_object = 2

    voxel_grid_dim = [20, 20, 20]
    object_frame_id = 'object_pose'
    #listener = tf.TransformListener()

    #grasp_id_file = h5py.File(grasp_data_path + '/grasp_data/grasp_obj_id.h5', 'r')
    #objects_num = grasp_id_file['cur_object_id'][()] + 1
    ##objects_num = 4
    #grasp_id_file.close()

    grasp_voxel_file_path = grasp_data_path + '/grasp_voxel_data.h5'
    grasp_voxel_file = h5py.File(grasp_voxel_file_path, 'a')

    grasp_configs_obj = []
    grasp_labels = []
    grasp_voxel_grids = []
    voxel_grasps_id = 0
    for obj_id in xrange(objects_per_batch):
        for grasp_id in xrange(grasps_per_object):
            obj_grasp_data_path = glob.glob(grasp_data_path + '/grasp_data/object_' + 
                                            str(obj_id) + '*grasp_' + str(grasp_id) + '*.h5')[0]
            obj_grasp_data_file = h5py.File(obj_grasp_data_path, 'r')
            obj_id_grasp_id = 'object_' + str(obj_id) + '_grasp_' + str(grasp_id)
            grasp_label = obj_grasp_data_file[obj_id_grasp_id + '_grasp_label'][()]
            js_position = obj_grasp_data_file[obj_id_grasp_id + 
                                                '_true_preshape_js_position'][()]
            palm_pose_array = obj_grasp_data_file[obj_id_grasp_id + 
                                                    '_true_preshape_palm_world_pose'][()]
            #object_pose_array = obj_grasp_data_file[obj_id_grasp_id + 
            #                                        '_object_pcd_pose'][()]
            object_pose_array = obj_grasp_data_file['object_' + str(obj_id) + '_grasp_' + 
                                str(grasp_id) + '_object_pose'][()]

            obj_grasp_data_file.close()

            pcd_file_path = glob.glob(grasp_data_path + '/pcd/object_' +  str(obj_id)
                                 + '*_grasp_' + str(grasp_id) + '.pcd')[0]

            palm_pose_world = data_proc_lib.convert_array_to_pose(palm_pose_array, 'world')
            data_proc_lib.update_palm_pose_client(palm_pose_world)

            # Object pose is already aligned in the new version data collection.
            object_pose_world = data_proc_lib.convert_array_to_pose(object_pose_array, 'world')
            #object_aligned_pose = align_obj_frame.align_obj_ort(object_pose_world, 
            #                                                    data_proc_lib.listener)
            #data_proc_lib.update_object_pose_client(object_aligned_pose)
            data_proc_lib.update_object_pose_client(object_aligned_pose)
            voxel_grasp_id = 'voxel_grasp_' + str(voxel_grasps_id)
            print voxel_grasp_id
            sparse_voxel_key = voxel_grasp_id + '_sparse_voxel'
            if sparse_voxel_key not in grasp_voxel_file:
            # Resume from previous voxel grids generation.
                print 'gen voxel'
                sparse_voxel_grid = data_proc_lib.voxel_gen_client(pcd_file_path)
                grasp_voxel_file.create_dataset(voxel_grasp_id + '_obj_id', data=obj_id)
                grasp_voxel_file.create_dataset(voxel_grasp_id + '_grasp_id', data=grasp_id)
                grasp_voxel_file.create_dataset(sparse_voxel_key, data=sparse_voxel_grid)
            else:
                print 'read voxel'
                sparse_voxel_grid = grasp_voxel_file[sparse_voxel_key][()]

            #show_voxel.plot_voxel(sparse_voxel_grid, './voxel.png') 

            voxel_grid = np.zeros(tuple(voxel_grid_dim))
            voxel_grid_index = sparse_voxel_grid.astype(int)
            voxel_grid[voxel_grid_index[:, 0], voxel_grid_index[:, 1], voxel_grid_index[:, 2]] = 1
            print sparse_voxel_grid.shape
            print np.sum(voxel_grid)
            #print sparse_voxel_grid
            #print np.array(np.where(voxel_grid == 1.))
            #print voxel_grid
            grasp_voxel_grids.append(voxel_grid)

            palm_pose_object_array = data_proc_lib.trans_pose_to_obj_tf(palm_pose_world) 
            #print palm_pose_object
            grasp_config_obj = np.concatenate((palm_pose_object_array, js_position), axis=0)
            print 'grasp_config_obj:', grasp_config_obj
            grasp_configs_obj.append(grasp_config_obj)

            grasp_labels.append(grasp_label)
            voxel_grasps_id += 1
            print '######'

    grasp_voxel_file.create_dataset('grasp_voxel_grids', data=grasp_voxel_grids)
    print grasp_configs_obj
    grasp_voxel_file.create_dataset('grasp_configs_obj', data=grasp_configs_obj)
    grasp_voxel_file.create_dataset('grasp_labels', data=grasp_labels)
    print np.sum(grasp_voxel_file['grasp_voxel_grids'][()])

    print 'Grasp data processing is done.'

    grasp_voxel_file.close()


if __name__ == '__main__':
    #data_path = '/mnt/tars_data/multi_finger_exp_data'
    data_path = '/dataspace/data_kai/al_grasp_queries/queries_batch_0'
    #proc_grasp_queries = ProcGraspQueries(data_path)
    #proc_grasp_queries.proc_grasp_queries()
    proc_grasp_queries(data_path)


