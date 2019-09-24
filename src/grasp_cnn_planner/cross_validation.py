import numpy as np
from sklearn.model_selection import StratifiedKFold
import random
import h5py
#from grasp_rgbd_config_net import GraspRgbdConfigNet
import os

def cross_val_seen_obj(k, labels, train_file_name, test_file_name):
    '''
    Seperate data into k folds for cross validation of seen objects.
    '''
    cross_val = StratifiedKFold(n_splits=k)
    #Note that providing y is sufficient to generate the splits and hence np.zeros(n_samples) 
    #may be used as a placeholder for X instead of actual training data.
    n_samples = np.shape(labels)[0]
    X = np.zeros(n_samples)
    print labels
    print X
    #train_indices = []
    #test_indices = []
    for i, (train, test) in enumerate(cross_val.split(X, labels)):
        #print("%s %s" % (train, test))
        #print test
        #train_indices.append(train)
        #test_indices.append(test)
        #print test_indices
        np.savetxt(train_file_name + '_fold_' + str(i) + '.txt', np.array(train))
        np.savetxt(test_file_name + '_fold_' + str(i) + '.txt', np.array(test))

    #train_indices, test_indices = cross_val.split(X, labels)
    #print train_indices[0]
    #print np.array(train_indices).shape
    #np.savetxt(train_file_name, np.array(train_indices))
    #np.savetxt(test_file_name, np.array(test_indices))

def get_k_fold_train_test_indices(k, train_file_name, test_file_name):
    '''
    Get the train and test indices of kth fold for cross validation.
    '''
    train_indices = np.loadtxt(train_file_name + '_fold_' + str(k) + '.txt')
    test_indices = np.loadtxt(test_file_name + '_fold_' + str(k) + '.txt')
    return train_indices.astype(int), test_indices.astype(int)

def read_data_labels(data_file_name):
    '''
    Read the grasp labels for stratified cross validation of seen objects.
    '''
    data_file = h5py.File(data_file_name, 'r')
    grasps_num = data_file['grasps_number'][()]
    #classes_num = 1
    grasp_labels = np.zeros(grasps_num)       

    for i in xrange(grasps_num):
        #print 'reading ', i
        grasp_sample_id = 'grasp_' + str(i)
        grasp_label_key = grasp_sample_id + '_grasp_label'
        grasp_labels[i] = data_file[grasp_label_key][()]

    data_file.close()

    return grasp_labels 

def cross_val_unseen_obj(k, grasp_ids_dict, grasp_suc_num_dict, 
                        train_file_name, test_file_name):
    '''
    Seperate data into k folds for cross validation of unseen objects.
    '''
    grasp_suc_num_list = []
    for key, value in grasp_suc_num_dict.items():
        grasp_suc_num_list.append((key, value))
   
    random.shuffle(grasp_suc_num_list)
    objects_by_suc_num = sorted(grasp_suc_num_list, key=lambda tup: tup[1])
    k_folds_indices_list = [[] for i in range(k)]
    for i, (object_name, _) in enumerate(objects_by_suc_num): 
        j = i % k
        #Reverse for even rounds to keep successful grasps number 
        #balanced among different folds.
        if (i / k) % 2 == 1:
            j = k - 1 - j 
        k_folds_indices_list[j] += grasp_ids_dict[object_name] 

    test_indices = k_folds_indices_list
    train_indices = [[] for i in range(k)]
    for i in xrange(k):
        for j in xrange(k):
            if i != j:
                train_indices[i] += test_indices[j] 
        np.savetxt(train_file_name + '_fold_' + str(i) + '.txt', np.array(train_indices[i]))
        np.savetxt(test_file_name + '_fold_' + str(i) + '.txt', np.array(test_indices[i]))

    #np.savetxt(train_file_name, np.array(train_indices))
    #np.savetxt(test_file_name, np.array(test_indices))

def read_grasp_info_per_obj(grasp_patches_file_name, grasp_info_file_name):
    '''
    Read the grasp information for stratified cross validation of unseen objects.
    '''
    grasp_patches_file = h5py.File(grasp_patches_file_name, 'r')
    grasp_info_file = h5py.File(grasp_info_file_name, 'r')
    grasps_num = grasp_patches_file['grasps_number'][()] 
    grasp_ids_dict = {}
    grasp_suc_num_dict = {}
    for i in xrange(grasps_num):
        #print 'reading ', i
        grasp_sample_id = 'grasp_' + str(i)
        object_grasp_id_key = grasp_sample_id + '_object_grasp_id'
        object_grasp_id = grasp_patches_file[object_grasp_id_key][()]
        object_id = object_grasp_id.split('_grasp')[0]
        object_name_key = object_id + '_name'
        object_name = grasp_info_file[object_name_key][()]
        grasp_label_key = grasp_sample_id + '_grasp_label'
        grasp_label = grasp_patches_file[grasp_label_key][()]
        if object_name not in grasp_ids_dict:
            grasp_ids_dict[object_name] = [i]
            if grasp_label:
                grasp_suc_num_dict[object_name] = 1
            else:
                grasp_suc_num_dict[object_name] = 0
        else:
            grasp_ids_dict[object_name].append(i)
            if grasp_label:
                grasp_suc_num_dict[object_name] += 1

    grasp_patches_file.close()
    grasp_info_file.close()

    return grasp_ids_dict, grasp_suc_num_dict

def run_cross_val_seen_obj():
    rgbd_patches_save_path = '/data_space/data_kai/multi_finger_sim_data/v4/'
    data_file_name = rgbd_patches_save_path + 'grasp_patches.h5'
    labels = read_data_labels(data_file_name)
    k = 5
    train_file_name = '../cross_val/cross_val_train_seen'
    test_file_name = '../cross_val/cross_val_test_seen'
    #cross_val_seen_obj(k, labels, train_file_name, test_file_name)
    for i in xrange(k):
        train_indices, test_indices = get_k_fold_train_test_indices(i, train_file_name, test_file_name)
        #print train_indices
        #print test_indices
        train_samples_num = np.shape(train_indices)[0]
        train_pos_num = np.sum(labels[train_indices.astype(int)])
        test_samples_num = np.shape(test_indices)[0]
        test_pos_num = np.sum(labels[test_indices.astype(int)])
        print 'train samples num:', train_samples_num, \
              'train positive samples num:', train_pos_num
        print 'test samples num:', test_samples_num, \
              'test positive samples num:', test_pos_num

        cmd = 'python grasp_rgbd_config_net.py train seen ' + str(i)
        os.system(cmd)
        cmd = 'python grasp_rgbd_config_net.py test seen ' + str(i)
        os.system(cmd)

def run_cross_val_unseen_obj():
    rgbd_patches_save_path = '/data_space/data_kai/multi_finger_sim_data/v4/'
    grasp_patches_file_name = rgbd_patches_save_path + 'grasp_patches.h5'
    grasp_info_file_name = rgbd_patches_save_path + 'grasp_data.h5'
    grasp_ids_dict, grasp_suc_num_dict = \
            read_grasp_info_per_obj(grasp_patches_file_name, grasp_info_file_name)
    k = 5
    train_file_name = '../cross_val/cross_val_train_unseen'
    test_file_name = '../cross_val/cross_val_test_unseen'
    #cross_val_unseen_obj(k, grasp_ids_dict, grasp_suc_num_dict, train_file_name, test_file_name)
    labels = read_data_labels(grasp_patches_file_name)
    for i in xrange(k):
        train_indices, test_indices = get_k_fold_train_test_indices(i, train_file_name, test_file_name)
        #print train_indices
        #print test_indices
        train_samples_num = np.shape(train_indices)[0]
        train_pos_num = np.sum(labels[train_indices.astype(int)])
        test_samples_num = np.shape(test_indices)[0]
        test_pos_num = np.sum(labels[test_indices.astype(int)])
        print 'train samples num:', train_samples_num, \
              'train positive samples num:', train_pos_num
        print 'test samples num:', test_samples_num, \
              'test positive samples num:', test_pos_num

        cmd = 'python grasp_rgbd_config_net.py train unseen ' + str(i)
        os.system(cmd)
        cmd = 'python grasp_rgbd_config_net.py test unseen ' + str(i)
        os.system(cmd)

def main():
    k = 5
    run_cross_val_seen_obj()
    run_cross_val_unseen_obj()

if __name__ == '__main__':
    main()
