import numpy as np
import h5py
import copy
import random


class GraspRgbdLoader():
    # Reference: http://www.cvc.uab.es/people/joans/
    # slides_tensorflow/tensorflow_html/feeding_and_queues.html

    def __init__(self, data_path):
        self.data_path = data_path 
        self.preshape_config_idx = list(xrange(8)) + [10, 11] + \
                                    [14, 15] + [18, 19]
        self.open_data_file()
        self.epochs_completed = 0
        # self.shuffle(is_shuffle=False)
        self.shuffle(is_shuffle=True)
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
                    is_shuffle=True, top_label=False, oversample_suc_num=None):
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
            # return self.load_batches(start, end)
            return self.load_batches_oversample_suc(start, end, oversample_suc_num)
        else:
            return self.load_batches(start, end), \
                     self.load_batch_top_labels(start, end)


    def load_batches_oversample_suc(self, start, end, oversample_suc_num=None):
        if oversample_suc_num is None:
            return self.load_batches(start, end)

        suc_configs, suc_rgbd_patches, suc_labels = \
                self.load_suc_batches(oversample_suc_num)

        grasp_configs, grasp_rgbd_patches, grasp_labels = \
                self.load_batches(start, end - oversample_suc_num)

        grasp_configs = grasp_configs + suc_configs
        grasp_rgbd_patches = grasp_rgbd_patches + suc_rgbd_patches
        grasp_labels = list(grasp_labels) + list(suc_labels)

        return grasp_configs, grasp_rgbd_patches, grasp_labels


    def load_batches(self, start, end):
        data_file = h5py.File(self.data_path, 'r')
        grasp_configs = []
        grasp_rgbd_patches = []
        grasp_labels = []
        for i in xrange(start, end):
            grasp_id = self.randperm[i]
            grasp_config_obj_key = 'grasp_' + str(grasp_id) + '_config_obj'
            grasp_full_config = data_file[grasp_config_obj_key][()] 
            grasp_preshape_config = grasp_full_config[self.preshape_config_idx]
            grasp_rgbd_patch_key = 'grasp_' + str(grasp_id) + '_rgbd_patch'
            rgbd_patch = data_file[grasp_rgbd_patch_key][()]
            grasp_label_key = 'grasp_' + str(grasp_id) + '_label'
            grasp_label = data_file[grasp_label_key][()]

            grasp_configs.append(grasp_preshape_config)
            grasp_rgbd_patches.append(rgbd_patch)
            grasp_labels.append(grasp_label)

        grasp_labels = np.expand_dims(grasp_labels, -1)
        data_file.close()

        return grasp_configs, grasp_rgbd_patches, grasp_labels


    def load_suc_batches(self, grasps_num):
        data_file = h5py.File(self.data_path, 'r')
        grasp_configs = []
        grasp_rgbd_patches = []
        grasp_labels = []
        while len(grasp_labels) < grasps_num:
            grasp_id = random.randint(0, self.num_samples - 1)
            grasp_label_key = 'grasp_' + str(grasp_id) + '_label'
            grasp_label = data_file[grasp_label_key][()]
            if not grasp_label:
                continue
            grasp_config_obj_key = 'grasp_' + str(grasp_id) + '_config_obj'
            grasp_full_config = data_file[grasp_config_obj_key][()] 
            grasp_preshape_config = grasp_full_config[self.preshape_config_idx]
            grasp_rgbd_patch_key = 'grasp_' + str(grasp_id) + '_rgbd_patch'
            rgbd_patch = data_file[grasp_rgbd_patch_key][()]

            grasp_configs.append(grasp_preshape_config)
            grasp_rgbd_patches.append(rgbd_patch)
            grasp_labels.append(grasp_label)

        grasp_labels = np.expand_dims(grasp_labels, -1)
        data_file.close()

        return grasp_configs, grasp_rgbd_patches, grasp_labels


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

