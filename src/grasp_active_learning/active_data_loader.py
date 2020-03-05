import numpy as np
import h5py
import copy


class ActiveDataLoader():


    def __init__(self, supervised_data_path, active_data_path):
        self.supervised_data_path = supervised_data_path 
        self.active_data_path = active_data_path

        self.voxel_grid_full_dim = [32, 32, 32]
        self.preshape_config_idx = list(xrange(8)) + [10, 11] + \
                                    [14, 15] + [18, 19]
        self.num_spv_samples = None
        self.num_act_samples = None
        self.num_samples = None

        # self.count_samples()

        # self.shuffle(is_shuffle=False)
        # self.shuffle(is_shuffle=True)

        self.epochs_completed = 0
        self.index_in_epoch = 0
        self.starts_new_epoch = True


    def count_samples(self):
        supervised_data_file = h5py.File(self.supervised_data_path, 'r')
        self.num_spv_samples = supervised_data_file['grasps_number'][()]
        supervised_data_file.close()

        active_data_file = h5py.File(self.active_data_path, 'r')
        self.num_act_samples = active_data_file['total_grasps_num'][()]
        active_data_file.close()

        self.num_samples = self.num_spv_samples + self.num_act_samples


    def active_batch(self, start, end):
        grasp_configs = []
        grasp_voxel_grids = []
        grasp_obj_sizes = []
        grasp_labels = []

        active_data_file = h5py.File(self.active_data_path, 'r') 
        for i in xrange(start, end): 
            print 'active sample:', i
            grasp_preshape_config, voxel_grid, \
                obj_size, grasp_label = \
                    self.load_active_sample(active_data_file, i)
            grasp_configs.append(grasp_preshape_config)
            grasp_voxel_grids.append(voxel_grid)
            grasp_obj_sizes.append(obj_size)
            grasp_labels.append(grasp_label)

        grasp_labels = np.expand_dims(grasp_labels, -1)
        grasp_voxel_grids = np.expand_dims(grasp_voxel_grids, -1)

        active_data_file.close()

        return grasp_configs, grasp_voxel_grids, \
                grasp_obj_sizes, grasp_labels


    def load_active_sample(self, data_file, grasp_id):
        obj_grasp_id_key = 'grasp_' + str(grasp_id) + '_obj_grasp_id'
        object_grasp_id = data_file[obj_grasp_id_key][()]

        voxel_grid_key = object_grasp_id + '_sparse_voxel_grid'
        sparse_voxel_grid = data_file[voxel_grid_key][()]
        object_size_key = object_grasp_id + '_object_size'
        obj_size = data_file[object_size_key][()]
        inf_config_array_key = object_grasp_id + '_inf_config_array'
        grasp_preshape_config = data_file[inf_config_array_key][()]
        grasp_label_key = object_grasp_id + '_grasp_label' 
        grasp_label = data_file[grasp_label_key][()]

        # Change the labels of grasps without plans from -1 to be 0 
        if grasp_label != 1:
            grasp_label = 0

        voxel_grid = np.zeros(tuple(self.voxel_grid_full_dim))
        voxel_grid_index = sparse_voxel_grid.astype(int)
        voxel_grid[voxel_grid_index[:, 0], voxel_grid_index[:, 1],
                    voxel_grid_index[:, 2]] = 1

        return grasp_preshape_config, voxel_grid, \
                obj_size, grasp_label
        

    def shuffle(self, is_shuffle=True):
        if is_shuffle:
            self.randperm = np.random.permutation(self.num_samples)
        else:
            self.randperm = list(xrange(self.num_samples))
 

    def next_batch(self, batch_size, 
                    is_shuffle=True):
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        if self.index_in_epoch > self.num_samples:
            # self.count_act_samples()
            # Finished epoch
            self.starts_new_epoch = True
            self.epochs_completed += 1            
            self.shuffle(is_shuffle)
            # Start next epoch
            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= self.num_samples
        else:
            self.starts_new_epoch = False

        end = self.index_in_epoch        
        
        return self.load_batches(start, end)


    def load_batches(self, start, end):
        supervised_data_file = h5py.File(self.supervised_data_path, 'r')
        active_data_file = h5py.File(self.active_data_path, 'r')
        grasp_configs = []
        grasp_voxel_grids = []
        grasp_obj_sizes = []
        grasp_labels = []
        for i in xrange(start, end):
            if i < self.num_spv_samples:
                grasp_preshape_config, voxel_grid, \
                    obj_size, grasp_label = \
                        self.load_supervised_sample(supervised_data_file, i)
            else:
                grasp_preshape_config, voxel_grid, \
                    obj_size, grasp_label = \
                        self.load_active_sample(active_data_file, 
                                                i - self.num_spv_samples)

            grasp_configs.append(grasp_preshape_config)
            grasp_voxel_grids.append(voxel_grid)
            grasp_obj_sizes.append(obj_size)
            grasp_labels.append(grasp_label)

        grasp_labels = np.expand_dims(grasp_labels, -1)
        grasp_voxel_grids = np.expand_dims(grasp_voxel_grids, -1)
        supervised_data_file.close()
        active_data_file.close()

        return grasp_configs, grasp_voxel_grids, \
                grasp_obj_sizes, grasp_labels


    def load_supervised_sample(self, data_file, grasp_id):
        grasp_config_obj_key = 'grasp_' + str(grasp_id) + '_config_obj'
        grasp_full_config = data_file[grasp_config_obj_key][()] 
        grasp_preshape_config = grasp_full_config[self.preshape_config_idx]
        grasp_sparse_voxel_key = 'grasp_' + str(grasp_id) + '_sparse_voxel'
        sparse_voxel_grid = data_file[grasp_sparse_voxel_key][()]
        obj_dim_key = 'grasp_' + str(grasp_id) + '_dim_w_h_d'
        obj_size = data_file[obj_dim_key][()]
        grasp_label_key = 'grasp_' + str(grasp_id) + '_label'
        grasp_label = data_file[grasp_label_key][()]

        voxel_grid = np.zeros(tuple(self.voxel_grid_full_dim))
        voxel_grid_index = sparse_voxel_grid.astype(int)
        voxel_grid[voxel_grid_index[:, 0], voxel_grid_index[:, 1],
                    voxel_grid_index[:, 2]] = 1

        return grasp_preshape_config, voxel_grid, \
                obj_size, grasp_label


    def load_grasp_configs(self):
        grasp_configs = []

        supervised_data_file = h5py.File(self.supervised_data_path, 'r')
        for grasp_id in xrange(self.num_spv_samples):
            grasp_config_obj_key = 'grasp_' + str(grasp_id) + '_config_obj'
            grasp_full_config = supervised_data_file[grasp_config_obj_key][()] 
            grasp_preshape_config = grasp_full_config[self.preshape_config_idx]
            grasp_configs.append(grasp_preshape_config)
        supervised_data_file.close()

        active_data_file = h5py.File(self.active_data_path, 'r')
        for grasp_id in xrange(self.num_act_samples):
            obj_grasp_id_key = 'grasp_' + str(grasp_id) + '_obj_grasp_id'
            object_grasp_id = active_data_file[obj_grasp_id_key][()]
            inf_config_array_key = object_grasp_id + '_inf_config_array'
            grasp_preshape_config = active_data_file[inf_config_array_key][()]
            grasp_configs.append(grasp_preshape_config)
        active_data_file.close()

        return grasp_configs


if __name__ == '__main__':
    supervised_data_path = '/mnt/tars_data/gazebo_al_grasps/train/' + \
            'merged_grasp_data_6_6_and_6_8_and_6_10_and_6_11_and_6_13.h5'
    active_data_path = '/mnt/tars_data/multi_finger_sim_data/grasp_data.h5'
    adl = ActiveDataLoader(supervised_data_path, active_data_path)
    
    # adl.count_samples()
    # print adl.num_act_samples

    # act_batch_size = 8
    # batch_id = 0
    # grasp_configs, grasp_voxel_grids, grasp_obj_sizes, \
    #         grasp_labels = adl.active_batch(act_batch_size * batch_id, 
    #                                         act_batch_size * (batch_id + 1))
    # print grasp_configs

    # batch_size = 1
    # # Don't forget to count the number of samples and shuffle the data
    # # before refitting the model using the supervised and active data.
    # adl.count_samples()
    # adl.shuffle(is_shuffle=True)
    # while adl.epochs_completed < 1:
    #     grasp_configs, grasp_voxel_grids, grasp_obj_sizes, \
    #             grasp_labels = adl.next_batch(batch_size)
    #     print grasp_configs

    adl.count_samples()
    grasp_configs = adl.load_grasp_configs()
    print grasp_configs

