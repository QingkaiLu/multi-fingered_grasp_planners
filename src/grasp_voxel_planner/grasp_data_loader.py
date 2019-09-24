import numpy as np
import h5py
import copy


class GraspDataLoader():
    # Reference: http://www.cvc.uab.es/people/joans/
    # slides_tensorflow/tensorflow_html/feeding_and_queues.html

    def __init__(self, data_path):
        self.data_path = data_path 
        self.voxel_grid_full_dim = [32, 32, 32]
        self.preshape_config_idx = list(xrange(8)) + [10, 11] + \
                                    [14, 15] + [18, 19]
        self.open_data_file()
        self.epochs_completed = 0
        self.shuffle(is_shuffle=False)
        self.index_in_epoch = 0
        self.starts_new_epoch = True


    def open_data_file(self):
        data_file = h5py.File(self.data_path, 'r')
        self.num_samples = data_file['grasps_number'][()]
        data_file.close()


    def shuffle(self, is_shuffle=True):
        if is_shuffle:
            self.randperm = np.random.permutation(self.num_samples)
        else:
            self.randperm = list(xrange(self.num_samples))
 

    def next_batch(self, batch_size, 
                    is_shuffle=True, top_label=False):
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        if self.index_in_epoch > self.num_samples:
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
        if not top_label:
            return self.load_batches(start, end)
        else:
            return self.load_batches(start, end), \
                     self.load_batch_top_labels(start, end)


    def load_batches(self, start, end):
        data_file = h5py.File(self.data_path, 'r')
        grasp_configs = []
        grasp_voxel_grids = []
        grasp_obj_sizes = []
        grasp_labels = []
        for i in xrange(start, end):
            grasp_id = self.randperm[i]
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

            grasp_configs.append(grasp_preshape_config)
            grasp_voxel_grids.append(voxel_grid)
            grasp_obj_sizes.append(obj_size)
            grasp_labels.append(grasp_label)

        grasp_labels = np.expand_dims(grasp_labels, -1)
        grasp_voxel_grids = np.expand_dims(grasp_voxel_grids, -1)
        data_file.close()

        return grasp_configs, grasp_voxel_grids, \
                grasp_obj_sizes, grasp_labels


    def load_batch_top_labels(self, start, end):
        data_file = h5py.File(self.data_path, 'r')
        top_labels = []
        for i in xrange(start, end):
            grasp_id = self.randperm[i]
            top_label_key = 'grasp_' + str(grasp_id) + '_top_grasp'
            top_label = data_file[top_label_key][()]
            top_labels.append(top_label)

        top_labels = np.expand_dims(top_labels, -1)
        data_file.close()

        return top_labels


    def load_grasp_configs(self):
        data_file = h5py.File(self.data_path, 'r')
        grasp_configs = []
        for grasp_id in xrange(self.num_samples):
            grasp_config_obj_key = 'grasp_' + str(grasp_id) + '_config_obj'
            grasp_full_config = data_file[grasp_config_obj_key][()] 
            grasp_preshape_config = grasp_full_config[self.preshape_config_idx]
            grasp_configs.append(grasp_preshape_config)

        data_file.close()

        return grasp_configs

