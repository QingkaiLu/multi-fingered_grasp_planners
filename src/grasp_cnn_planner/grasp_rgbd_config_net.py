import tensorflow as tf
import random
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tsne_vis
import h5py
import numpy as np
import time
#import gen_rgbd_patches as gp
import cv2
import plot_roc_pr_curve
from sklearn.metrics import precision_recall_fscore_support
import sys
import cross_validation
import roslib.packages as rp
from sklearn import mixture
from sklearn.cluster import KMeans
import pickle
import sys
pkg_path = rp.get_pkg_dir('prob_grasp_planner')
# sys.path.append(pkg_path + '/src/grasp_type_planner')
#import proc_grasp_data as pgd
import rospy
from prob_grasp_planner.srv import *


# One network that directly takes the large subimage and 
# predicate the grasp preshape configuration as a regression problem.
# Notice: the word "epoch" is not used correctly here, it should be "iteration".
class GraspRgbdConfigNet():
    def __init__(self):
        # Parameters
        self.grasp_patch_rows = 400
        self.grasp_patch_cols = 400
        self.rgbd_channels = 8
        self.rate_dec_epochs = 1000#400
        self.training_epochs = 3000#1200
        self.batch_size = 8#16
        self.display_step = 10#0
        self.dropout_prob = 0.75#0.9
        self.config_dim = 14#11
        #self.read_data_ratio = 0.25
        #self.read_batch_epochs = 5000
        self.read_data_ratio = 0.5
        self.read_batch_epochs = 5000
        self.classes_num = 1

        #rgbd_patches_save_path = '/data_space/data_kai/multi_finger_exp_data'
        rgbd_patches_save_path = '/mnt/tars_data/multi_finger_sim_data_complete_v4/'
        self.grasp_patches_file_path = rgbd_patches_save_path + 'grasp_patches.h5'
        self.grasp_data_file_path = rgbd_patches_save_path + 'grasp_data.h5'
        self.logs_path = '/home/qingkai/tf_logs/multi_finger_sim_data'
        pkg_path = rp.get_pkg_dir('prob_grasp_planner') 

        self.cnn_model_path = pkg_path + '/models/grasp_cnn_planner/' + \
                              'models/cnn_rgbd_config.ckpt'       
        self.prior_path = pkg_path + '/models/grasp_cnn_planner/priors/'

        # tf Graph input
        self.holder_grasp_patch = tf.placeholder(tf.float32, 
                [None, self.grasp_patch_rows, self.grasp_patch_cols, self.rgbd_channels], name='holder_grasp_patch')
        self.holder_config = tf.placeholder(tf.float32, [None, self.config_dim], name='holder_config')
        self.holder_labels = tf.placeholder(tf.float32, [None, self.classes_num], name = 'holder_labels')
        self.keep_prob = tf.placeholder(tf.float32, name='holder_keep_prob')
        self.learning_rate = tf.placeholder(tf.float32, name='holder_learn_rate')
               
        self.train_samples_pct = 0.8
        self.train_samples_num = -1 
        self.testing_samples_num = -1 
        self.training_labels = None
        self.training_grasp_samples = None 
        self.training_samples_fingers = None 
        self.testing_labels = None
        self.testing_grasp_samples = None 
        self.testing_samples_fingers = None 
        self.read_data_batch = False
        self.alpha_ridge = 0.5
        #Need to change this to false for cross validation.
        self.use_all_data_train = True

        #self.proc_grasp = pgd.ProcGraspData('')


    #Create some wrappers for simplicity
    def conv2d(self, x, W, b, stride=1):
        # Conv2D wrapper, with relu activation
        x = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)
    
    def max_pool2d(self, x, k=2, stride=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, stride, stride, 1], 
                              padding='SAME')
    
    def conv_net_grasp(self, x_grasp, x_config, w_grasp, b_grasp, strides_grasp):
        with tf.name_scope('Grasp'):
            #Convolution 1
            conv1 = self.conv2d(x_grasp, w_grasp['conv1'], b_grasp['conv1'], strides_grasp['conv1'])
            #Convolution 2
            conv2 = self.conv2d(conv1, w_grasp['conv2'], b_grasp['conv2'], strides_grasp['conv2'])
            #Max pool layer
            max_pool_1 = self.max_pool2d(conv2, strides_grasp['pool1'], strides_grasp['pool1'])
            #Configuration FC layer
            fc_config = tf.add(tf.matmul(x_config, w_grasp['fc_config']), b_grasp['fc_config'])
            fc_config = tf.nn.relu(fc_config)
            fc_config = tf.nn.dropout(fc_config, self.keep_prob)
            #Tile and reshape configuration FC outputs to the same dimension of conv2 outputs
            strides_prod_rgbd = self.strides_grasp['conv1'] * self.strides_grasp['conv2'] * \
                                self.strides_grasp['pool1']
            rgbd_pool_dim_row = self.grasp_patch_rows / strides_prod_rgbd 
            rgbd_pool_dim_col = self.grasp_patch_cols / strides_prod_rgbd
            fc_config = tf.expand_dims(fc_config, axis=1)
            fc_config = tf.expand_dims(fc_config, axis=2)
            fc_config = tf.tile(fc_config, [1, rgbd_pool_dim_row, rgbd_pool_dim_col, 1])
            #fc_config = tf.tile(fc_config, [self.conv2_w_par_grasp[-1], rgbd_pool_dim_row * rgbd_pool_dim_col])
            #fc_config = tf.reshape(fc_config, [self.conv2_w_par_grasp[-1], rgbd_pool_dim_row, rgbd_pool_dim_col])
            #Concatenate tiled configuration FC outputs and the rgbd pooling layer outputs.
            rgbd_config_concat = tf.concat(axis=3, values=[max_pool_1, fc_config])
            #Convluation 3
            conv3 = self.conv2d(rgbd_config_concat, w_grasp['conv3'], b_grasp['conv3'], strides_grasp['conv3'])
            #Max pool layer
            max_pool_2 = self.max_pool2d(conv3, strides_grasp['pool2'], strides_grasp['pool2'])
            #FC layer 1
            fc1 = tf.add(tf.matmul(tf.reshape(max_pool_2, [-1, int(np.prod(max_pool_2.get_shape()[1:]))]), 
                                   w_grasp['fc1']), b_grasp['fc1'])
            fc1 = tf.nn.relu(fc1)
            fc1 = tf.nn.dropout(fc1, self.keep_prob)
            #FC layer 2
            fc2 = tf.add(tf.matmul(fc1, w_grasp['fc2']), b_grasp['fc2'])
            fc2 = tf.nn.relu(fc2)
            fc2 = tf.nn.dropout(fc2, self.keep_prob)
            out_layer = tf.matmul(fc2, w_grasp['out']) + b_grasp['out']

        return (fc1, fc2, out_layer)

    def create_net_var(self):
        # Store layers weight & bias
        self.strides_grasp = {
                'conv1': 2,
                'conv2': 2,
                'pool1': 2,
                'conv3': 2,
                'pool2': 2,
                }

        # conv shapes for the palm.
        # 12x12x12 conv, 8 input channels, 32 outputs
        self.conv1_w_par_grasp = [12, 12, self.rgbd_channels, 32] 
        self.conv1_b_par_grasp = [32]
        # 6x6x6 conv, 32 input channels, 8 outputs
        self.conv2_w_par_grasp = [6, 6, 32, 8]
        self.conv2_b_par_grasp = [8]
        fc_config_w_par_grasp = [self.config_dim, self.conv2_w_par_grasp[-1]]
        # 6x6x6 conv, 32 input channels, 8 outputs
        self.conv3_w_par_grasp = [3, 3, 8 * 2, 8]
        self.conv3_b_par_grasp = [8]
        self.fc1_neurons_num = 32
        strides_prod_rgbd = self.strides_grasp['conv1'] * self.strides_grasp['conv2'] * \
                                self.strides_grasp['pool1'] * self.strides_grasp['conv3'] * \
                                self.strides_grasp['pool2']
        # fc1, 2(conv1)*2(conv2)*2(pool1)*2(conv3)*2(pool2) = 32
        # 400/32 * 400/32 * 8 inputs, 32 outputs
        fc1_input_dim = np.ceil(float(self.grasp_patch_rows) / float(strides_prod_rgbd)) * \
                np.ceil(float(self.grasp_patch_cols) / float(strides_prod_rgbd)) * self.conv3_w_par_grasp[-1]
        fc1_w_par_grasp = [fc1_input_dim, self.fc1_neurons_num]
        self.fc2_neurons_num = 32
        fc2_w_par_grasp = [self.fc1_neurons_num, self.fc2_neurons_num]
        out_w_par_grasp = [self.fc2_neurons_num, self.classes_num]

        self.weights_grasp = {
                'conv1': tf.get_variable(name='w_conv1_grasp', shape=self.conv1_w_par_grasp,
                        initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                'conv2': tf.get_variable(name='w_conv2_grasp', shape=self.conv2_w_par_grasp, 
                    initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                'fc_config': tf.get_variable(name='w_fc_config_grasp', shape=fc_config_w_par_grasp, 
                    initializer=tf.contrib.layers.xavier_initializer()),
                'conv3': tf.get_variable(name='w_conv3_grasp', shape=self.conv3_w_par_grasp, 
                    initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                'fc1': tf.get_variable(name='w_fc1_grasp', shape=fc1_w_par_grasp, 
                    initializer=tf.contrib.layers.xavier_initializer()),
                'fc2': tf.get_variable(name='w_fc2_grasp', shape=fc2_w_par_grasp, 
                    initializer=tf.contrib.layers.xavier_initializer()),
                'out': tf.get_variable(name='w_out_grasp', shape=out_w_par_grasp, 
                    initializer=tf.contrib.layers.xavier_initializer()),
                }

        self.biases_grasp = {
                'conv1': tf.get_variable(name='b_conv1_grasp', shape=self.conv1_b_par_grasp, 
                    #initializer=tf.constant_initializer(0.0)),
                    initializer=tf.random_normal_initializer()),
                'conv2': tf.get_variable(name='b_conv2_grasp', shape=self.conv2_b_par_grasp, 
                    #initializer=tf.constant_initializer(0.0)),
                    initializer=tf.random_normal_initializer()),
                'fc_config': tf.get_variable(name='b_fc_config_grasp', shape=[self.conv2_w_par_grasp[-1]], 
                    initializer=tf.random_normal_initializer()),
                'conv3': tf.get_variable(name='b_conv3_grasp', shape=self.conv3_b_par_grasp, 
                    #initializer=tf.constant_initializer(0.0)),
                    initializer=tf.random_normal_initializer()),
                'fc1': tf.get_variable(name='b_fc1_grasp', shape=[self.fc1_neurons_num], 
                    initializer=tf.random_normal_initializer()),
                'fc2': tf.get_variable(name='b_fc2_grasp', shape=[self.fc2_neurons_num], 
                    #initializer=tf.constant_initializer(0.0)),
                    initializer=tf.random_normal_initializer()),
                'out': tf.get_variable(name='b_out_grasp', shape=[self.classes_num], 
                    #initializer=tf.constant_initializer(0.0)),
                    initializer=tf.random_normal_initializer()),
                }


    def cost_function(self):
        with tf.name_scope('Pred'):
            (fc1_grasp, fc2_grasp, logits) = \
                    self.conv_net_grasp(self.holder_grasp_patch, self.holder_config, self.weights_grasp, 
                                        self.biases_grasp, self.strides_grasp)
            self.pred = tf.nn.sigmoid(logits)
            #self.pred = logits

        with tf.name_scope('F2Vis'):
            self.feature_to_vis = tf.concat(axis=1, values=[fc1_grasp, fc2_grasp]) 
        
        with tf.name_scope('Pred_error'):
            self.pred_error = tf.reduce_mean(tf.abs(self.holder_labels - self.pred))
        
        with tf.name_scope('Pred_error_test'):
            self.pred_error_test = tf.reduce_mean(tf.abs(self.holder_labels - self.pred))
        
        with tf.name_scope('Cost'):
            self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.holder_labels))

        with tf.name_scope('SGD'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        
        # Create a summary to monitor cost
        loss_sum = tf.summary.scalar('loss', self.cost)
        # Create a summary to monitor training prediction errors
        pred_err_sum = tf.summary.scalar('pred_err_train', self.pred_error)
        # Learnining rate
        learn_rate_sum = tf.summary.scalar('learning_rate', self.learning_rate)
        self.train_summary = tf.summary.merge([loss_sum, pred_err_sum, learn_rate_sum])

        # Testing prediction errors
        self.pred_err_test_sum = tf.summary.scalar('pred_err_test', self.pred_error_test)

    def read_rgbd_data(self):
        grasp_patches_file = h5py.File(self.grasp_patches_file_path, 'r')
        grasps_num = grasp_patches_file['grasps_number'][()] 

        grasp_configs = np.zeros((grasps_num, self.config_dim))       
        grasp_patches = np.zeros((grasps_num, self.grasp_patch_rows, self.grasp_patch_cols, self.rgbd_channels))
        grasp_labels = np.zeros((grasps_num, self.classes_num))       

        for i in xrange(grasps_num):
            print 'reading ', i
            grasp_sample_id = 'grasp_' + str(i)
            grasp_label_key = grasp_sample_id + '_grasp_label'
            grasp_labels[i] = grasp_patches_file[grasp_label_key]
            grasp_patch_key = grasp_sample_id + '_grasp_patch'
            grasp_patches[i] = grasp_patches_file[grasp_patch_key]
            grasp_config_key = grasp_sample_id + '_preshape_true_config'
            grasp_configs[i] = grasp_patches_file[grasp_config_key]

        print 'Reading is done.'
        grasp_patches_file.close() 

        #print grasp_configs
        print 'grasp_configs.shape: ', grasp_configs.shape

        self.train_samples_num = int(grasps_num * self.train_samples_pct)
        self.testing_samples_num = grasps_num - self.train_samples_num
       
        if self.use_all_data_train:
            self.training_labels = grasp_labels[0:grasps_num]
            self.training_configs = grasp_configs[0:grasps_num]
            self.training_grasp_samples = grasp_patches[0:grasps_num]                                   
        else:
            self.training_labels = grasp_labels[0:self.train_samples_num]
            self.training_configs = grasp_configs[0:self.train_samples_num]
            self.training_grasp_samples = grasp_patches[0:self.train_samples_num]                                   


        self.testing_labels = grasp_labels[self.train_samples_num:grasps_num]                                                                
        self.testing_configs = grasp_configs[self.train_samples_num:grasps_num]                                                                
        self.testing_grasp_samples = grasp_patches[self.train_samples_num:grasps_num]
        
        print self.training_labels.shape
        print 'training positive #:', np.sum(self.training_labels[:, 0]==1)
        print self.training_grasp_samples.shape
        print self.training_configs.shape

        print self.testing_labels.shape
        print 'testing positive #:', np.sum(self.testing_labels[:, 0]==1)
        print self.testing_grasp_samples.shape
        print self.testing_configs.shape



    def read_grasp_data_from_indices(self, train_indices, test_indices):
        '''
        Read training and testing data from indices. This is used for cross validation.
        '''
        grasp_patches_file = h5py.File(self.grasp_patches_file_path, 'r')

        if train_indices is not None:
            self.train_samples_num = np.shape(train_indices)[0]
            self.training_configs = np.zeros((self.train_samples_num, self.config_dim))       
            self.training_grasp_samples = \
                    np.zeros((self.train_samples_num, self.grasp_patch_rows, self.grasp_patch_cols, self.rgbd_channels))
            self.training_labels = np.zeros((self.train_samples_num, self.classes_num))       

            for i, j in enumerate(train_indices):
                grasp_sample_id = 'grasp_' + str(j)
                grasp_label_key = grasp_sample_id + '_grasp_label'
                self.training_labels[i] = grasp_patches_file[grasp_label_key]
                grasp_patch_key = grasp_sample_id + '_grasp_patch'
                self.training_grasp_samples[i] = grasp_patches_file[grasp_patch_key]
                grasp_config_key = grasp_sample_id + '_preshape_true_config'
                self.training_configs[i] = grasp_patches_file[grasp_config_key]

            print self.training_labels.shape
            print 'training positive #:', np.sum(self.training_labels[:, 0]==1)
            print self.training_grasp_samples.shape
            print self.training_configs.shape

        if test_indices is not None:
            self.testing_samples_num = np.shape(test_indices)[0]
            self.testing_configs = np.zeros((self.testing_samples_num, self.config_dim))       
            self.testing_grasp_samples = \
                    np.zeros((self.testing_samples_num, self.grasp_patch_rows, self.grasp_patch_cols, self.rgbd_channels))
            self.testing_labels = np.zeros((self.testing_samples_num, self.classes_num))       

            for i, j in enumerate(test_indices):
                grasp_sample_id = 'grasp_' + str(j)
                grasp_label_key = grasp_sample_id + '_grasp_label'
                self.testing_labels[i] = grasp_patches_file[grasp_label_key]
                grasp_patch_key = grasp_sample_id + '_grasp_patch'
                self.testing_grasp_samples[i] = grasp_patches_file[grasp_patch_key]
                grasp_config_key = grasp_sample_id + '_preshape_true_config'
                self.testing_configs[i] = grasp_patches_file[grasp_config_key]

            print self.testing_labels.shape
            print 'testing positive #:', np.sum(self.testing_labels[:, 0]==1)
            print self.testing_grasp_samples.shape
            print self.testing_configs.shape

        print 'Reading is done.'
        grasp_patches_file.close() 

    def read_rgbd_data_batch(self, batch_ratio=1.):
        data_path = '/media/kai/cornell_grasp_data/'
        patch_file = h5py.File(data_path + 'h5_data/patch.h5', 'r')
      
        print patch_file['labels']
        print patch_file['labels'].shape
        grasps_num = patch_file['labels'].shape[0]        
        #grasps_num = 1000
        
        train_pct = 0.9
        batch_grasps_num = int(grasps_num * batch_ratio)
        self.train_samples_num = int(batch_grasps_num * train_pct)
        self.testing_samples_num = batch_grasps_num - self.train_samples_num
        
        self.training_labels = np.zeros((self.train_samples_num, self.config_dim))
        self.training_grasp_samples = np.zeros((self.train_samples_num, self.grasp_patch_rows, self.grasp_patch_cols, self.rgbd_channels))
        self.training_samples_f_1 = np.zeros((self.train_samples_num, self.finger_patch_rows, self.finger_patch_cols, self.rgbd_channels))
        self.training_samples_f_2 = np.zeros((self.train_samples_num, self.finger_patch_rows, self.finger_patch_cols, self.rgbd_channels))
        
        self.testing_labels = np.zeros((self.testing_samples_num, self.config_dim))
        self.testing_grasp_samples = np.zeros((self.testing_samples_num, self.grasp_patch_rows, self.grasp_patch_cols, self.rgbd_channels))
        self.testing_samples_f_1 = np.zeros((self.testing_samples_num, self.finger_patch_rows, self.finger_patch_cols, self.rgbd_channels))
        self.testing_samples_f_2 = np.zeros((self.testing_samples_num, self.finger_patch_rows, self.finger_patch_cols, self.rgbd_channels))
       
        train_samples_total_num = int(grasps_num * train_pct)
        train_batch_indices = random.sample(range(0, train_samples_total_num), self.train_samples_num)
        test_batch_indices = random.sample(range(train_samples_total_num, grasps_num), self.testing_samples_num)
        print 'train_batch_indices: ', train_batch_indices
        print np.min(train_batch_indices), np.max(train_batch_indices)
        print 'test_batch_indices: ', test_batch_indices
        print np.min(test_batch_indices), np.max(test_batch_indices)

        for i, grasp_idx in enumerate(train_batch_indices):
            #print 'read train sample: ', i, grasp_idx
            self.training_grasp_samples[i] = patch_file['palm_' + str(grasp_idx)]
            self.training_samples_f_1[i] = patch_file['f_1_' + str(grasp_idx)]
            self.training_samples_f_2[i] = patch_file['f_2_' + str(grasp_idx)]
            self.training_labels[i, 0] = patch_file['labels'][grasp_idx] 

        for i, grasp_idx in enumerate(test_batch_indices):
            #print 'read test sample: ', i, grasp_idx
            self.testing_grasp_samples[i] = patch_file['palm_' + str(grasp_idx)]
            self.testing_samples_f_1[i] = patch_file['f_1_' + str(grasp_idx)]
            self.testing_samples_f_2[i] = patch_file['f_2_' + str(grasp_idx)]
            self.testing_labels[i, 0] = patch_file['labels'][grasp_idx] 

        #for i, grasp_idx in enumerate(train_batch_indices):
        #    print 'read train sample: ', i, grasp_idx
        #    if i % 2 == 0:
        #        self.training_grasp_samples[i] = patch_file['palm_' + str(grasp_idx)]
        #        self.training_samples_f_1[i] = patch_file['f_1_' + str(grasp_idx)]
        #        self.training_samples_f_2[i] = patch_file['f_2_' + str(grasp_idx)]
        #        self.training_labels[i, 0] = patch_file['labels'][grasp_idx] 
        #    else:
        #        self.training_grasp_samples[i] = patch_file['palm_' + str(grasp_idx)]
        #        self.training_samples_f_1[i] = patch_file['f_2_' + str(grasp_idx)]
        #        self.training_samples_f_2[i] = patch_file['f_1_' + str(grasp_idx)]
        #        self.training_labels[i, 0] = patch_file['labels'][grasp_idx] 


        #for i, grasp_idx in enumerate(test_batch_indices):
        #    print 'read test sample: ', i, grasp_idx
        #    if i % 2 == 0:
        #        self.testing_grasp_samples[i] = patch_file['palm_' + str(grasp_idx)]
        #        self.testing_samples_f_1[i] = patch_file['f_1_' + str(grasp_idx)]
        #        self.testing_samples_f_2[i] = patch_file['f_2_' + str(grasp_idx)]
        #        self.testing_labels[i, 0] = patch_file['labels'][grasp_idx] 
        #    else:
        #        self.testing_grasp_samples[i] = patch_file['palm_' + str(grasp_idx)]
        #        self.testing_samples_f_1[i] = patch_file['f_2_' + str(grasp_idx)]
        #        self.testing_samples_f_2[i] = patch_file['f_1_' + str(grasp_idx)]
        #        self.testing_labels[i, 0] = patch_file['labels'][grasp_idx] 

        print 'reading is done.'
        patch_file.close() 

        self.training_labels[self.training_labels == -1] = 0
        self.testing_labels[self.testing_labels == -1] = 0

        print self.training_grasp_samples.shape
        print self.training_samples_f_1.shape
        print self.training_samples_f_2.shape
        print self.training_labels.shape
        print 'training positive #:', np.sum(self.training_labels[:, 0]==1)
        
        print self.testing_grasp_samples.shape
        print self.testing_samples_f_1.shape
        print self.testing_samples_f_2.shape
        print self.testing_labels.shape
        print 'testing positive #:', np.sum(self.testing_labels[:, 0]==1)

    def feed_dict_func(self, train, l_rate=.0, batch_idx=None):
        xs_config = None
        xs_grasp = None
        ys = None
        if train:
            #rand_indices = random.sample(range(0, self.train_samples_num), self.batch_size)
            #xs_config = self.training_configs[rand_indices]
            #xs_grasp = self.training_grasp_samples[rand_indices]
            #ys = self.training_labels[rand_indices]

            contain_positive_label = False
            while not contain_positive_label:
                rand_indices = random.sample(range(0, self.train_samples_num), self.batch_size)
                ys = self.training_labels[rand_indices]
                contain_positive_label = np.sum(ys) > 1

            xs_config = self.training_configs[rand_indices]
            xs_grasp = self.training_grasp_samples[rand_indices]
            ys = self.training_labels[rand_indices]

            k = self.dropout_prob
        else:
            if batch_idx is None:
                rand_indices = random.sample(range(0, self.testing_samples_num), self.batch_size)
                xs_config = self.testing_configs[rand_indices]
                xs_grasp = self.testing_grasp_samples[rand_indices]
                ys = self.testing_labels[rand_indices]
            else:
                xs_config = self.testing_configs[batch_idx * self.batch_size: 
                                     min((batch_idx + 1) * self.batch_size, self.testing_samples_num)]
                xs_grasp = self.testing_grasp_samples[batch_idx * self.batch_size: 
                                     min((batch_idx + 1) * self.batch_size, self.testing_samples_num)]
                ys = self.testing_labels[batch_idx * self.batch_size: 
                                     min((batch_idx + 1) * self.batch_size, self.testing_samples_num)]
            k = 1.0

        return {self.holder_labels: ys, self.holder_config: xs_config, self.holder_grasp_patch: xs_grasp,
                self.keep_prob: k, self.learning_rate: l_rate}
    
    def train(self, train_indices=None, test_indices=None):
        #Read data
        if self.read_data_batch:
            self.read_rgbd_data_batch(self.read_data_ratio)
        elif train_indices is not None:
            self.read_grasp_data_from_indices(train_indices, test_indices)
        else:
            self.read_rgbd_data()
        #Create the variables.
        self.create_net_var()
        #Call the network building function.
        self.cost_function()
        #Initialize the variables
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        tf.get_default_graph().finalize()   

        logs_path = self.logs_path + '_train'
    
        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        
        start_time = time.time()
        train_costs = []
        train_pred_errors = []
        test_pred_errors = []
        learn_rate = 0.001
        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            # Training cycle
            for epoch in range(self.training_epochs):
                if epoch % self.rate_dec_epochs == 0 and epoch != 0:
                    learn_rate *= 0.1
                if self.read_data_batch and epoch % self.read_batch_epochs == 0 and epoch != 0:
                    self.read_rgbd_data_batch(self.read_data_ratio)

                #[pool_1_out, fc_config_out] = sess.run([self.max_pool_1, self.fc_config], feed_dict=self.feed_dict_func(True, learn_rate))
                #print np.array(pool_1_out).shape, np.array(fc_config_out).shape

                [_, cost_output, pred_error_output, train_summary_output, f2vis_train_output, labels_train_output, pred_train_output] = \
                        sess.run([self.optimizer, self.cost, self.pred_error, self.train_summary, self.feature_to_vis, 
                            self.holder_labels, self.pred], feed_dict=self.feed_dict_func(True, learn_rate))

                # Write logs at every iteration
                summary_writer.add_summary(train_summary_output, epoch)
    
                # Display logs per epoch step
                train_costs.append(cost_output)
                train_pred_errors.append(pred_error_output)

                [pred_error_test_output, pred_err_test_sum_output, f2vis_output_test, labels_test_output, pred_test_output] = sess.run(
                        [self.pred_error_test, self.pred_err_test_sum, self.feature_to_vis, self.holder_labels, self.pred], 
                        feed_dict=self.feed_dict_func(False))
                test_pred_errors.append(pred_error_test_output)
                summary_writer.add_summary(pred_err_test_sum_output, epoch)

                if epoch % self.display_step == 0:
                    print 'epoch: ', epoch
                    print 'labels_train_output:', labels_train_output
                    print 'pred_train_output', pred_train_output
                    print 'labels_test_output:', labels_test_output
                    print 'pred_test_output', pred_test_output
                    print 'train_cost: ', cost_output
                    print 'pred_error_train: ', pred_error_output
                    print 'train pred std: ', np.std(pred_train_output[:, 0])
                    print 'pred_error_test: ', pred_error_test_output
                    print 'test pred std: ', np.std(pred_test_output[:, 0])
                    pred_train_output[pred_train_output > 0.5] = 1.
                    pred_train_output[pred_train_output <= 0.5] = 0.
                    train_prfc = precision_recall_fscore_support(labels_train_output, pred_train_output)
                    print 'training precision, recall, fscore, support:'
                    print train_prfc
                    pred_test_output[pred_test_output > 0.5] = 1.
                    pred_test_output[pred_test_output <= 0.5] = 0.
                    test_prfc = precision_recall_fscore_support(labels_test_output, pred_test_output)
                    print 'testing precision, recall, fscore, support:'
                    print test_prfc

                if epoch % (10 * self.display_step) == 0:
                    tsne_vis.tsne_vis(f2vis_train_output, labels_train_output[:, 0], 
                            '../models/tsne/train_tsne_' + str(epoch) + '_gt.png')
                    tsne_vis.tsne_vis(f2vis_output_test, labels_test_output[:, 0], 
                            '../models/tsne/test_tsne_' + str(epoch) + '_gt.png')
                    tsne_vis.tsne_vis(f2vis_train_output, pred_train_output[:, 0], 
                            '../models/tsne/train_tsne_' + str(epoch) + '_pred.png')
                    tsne_vis.tsne_vis(f2vis_output_test, pred_test_output[:, 0], 
                            '../models/tsne/test_tsne_' + str(epoch) + '_pred.png')
                    

            print("Optimization Finished!")
            saver.save(sess, self.cnn_model_path)
            
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.plot(np.linspace(1,self.training_epochs,self.training_epochs), train_costs, 'r')
            fig.savefig('../models/train_costs.png' )
            plt.clf()
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.plot(np.linspace(1,self.training_epochs,self.training_epochs), train_pred_errors, 'b')
            fig.savefig('../models/train_pred_errors.png' )
            plt.clf()
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.plot(np.linspace(1,self.training_epochs,self.training_epochs), test_pred_errors, 'k')
            fig.savefig('../models/test_pred_errors.png' )
            plt.clf()
            plt.close(fig)

        elapsed_time = time.time() - start_time
        print 'Total training elapsed_time: ', elapsed_time

    def test(self, train_indices=None, test_indices=None, seen_or_unseen=None, k_fold=None):
        #Read data
        if test_indices is not None:
            self.read_grasp_data_from_indices(train_indices, test_indices)
        else:
            self.read_rgbd_data()

        #Create the variables.
        self.create_net_var()
        #Call the network building function.
        self.cost_function()
        #Initialize the variables
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        tf.get_default_graph().finalize()   

        logs_path = self.logs_path + '_test'
        
        start_time = time.time()
        learn_rate = 0.001
        #learn_rate = 0.0001

        with tf.Session() as sess:
            # Test model
            saver.restore(sess, self.cnn_model_path)
            pred_score_output = np.array([]).reshape(0, self.classes_num)
            f2vis_output = np.zeros([self.testing_samples_num, self.fc1_neurons_num + self.fc2_neurons_num])
            for i in xrange(np.ceil(float(self.testing_samples_num) / self.batch_size).astype(int)):
                [b_grasp, w_grasp, f2vis_batch_output, pred_batch_output] = sess.run(
                        [self.biases_grasp['out'], self.weights_grasp['out'], self.feature_to_vis, self.pred],
                        feed_dict=self.feed_dict_func(False, .0, i))
                pred_score_output = np.concatenate((pred_score_output, pred_batch_output))
                start_idx = i * self.batch_size
                f2vis_output[start_idx : start_idx + len(f2vis_batch_output), :] = f2vis_batch_output
            print 'pred_score_output: ', pred_score_output
            print 'self.testing_labels: ', self.testing_labels
            pred_output = np.copy(pred_score_output)
            pred_output[pred_output > 0.4] = 1.
            pred_output[pred_output <= 0.4] = 0.
            print 'binary pred_output: ', pred_output
            print 'pred_output.shape: ', pred_output.shape
            print 'self.testing_labels.shape: ', self.testing_labels.shape
            pred_errors_all = np.abs(pred_output - self.testing_labels) 
            avg_pred_error_test = np.mean(pred_errors_all)
            print('avg_pred_error_test:', avg_pred_error_test)

            #print f2vis_output.shape, self.testing_labels.shape
            #tsne_vis.tsne_vis(f2vis_output, self.testing_labels[:, 0], '../models/tsne/test_tsne_gt.png', True)
            #tsne_vis.tsne_vis(f2vis_output, pred_output[:, 0], '../models/tsne/test_tsne_pred.png', True)

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.plot(np.linspace(1,self.testing_samples_num,self.testing_samples_num), self.testing_labels, 'ro', label='gt labels')
            ax.plot(np.linspace(1,self.testing_samples_num,self.testing_samples_num), pred_output, 'bo', label='pred labels')
            ax.plot(np.linspace(1,self.testing_samples_num,self.testing_samples_num), pred_errors_all, 'k*', label='pred errors')
            #plt.legend(handles=[gt_plot, pred_plot, error_plot])
            legend = ax.legend(loc='upper right', shadow=True)
            frame = legend.get_frame()
            frame.set_facecolor('0.90')
            fig.savefig('../models/labels_test.png')
            plt.clf()
            plt.close(fig)
        
        prfc = precision_recall_fscore_support(self.testing_labels, pred_output)
        print 'precision, recall, fscore, support:'
        print prfc

        gt_labels_file_name = '../cross_val/' + seen_or_unseen + '/gt_labels_fold_' + k_fold + '.txt'
        np.savetxt(gt_labels_file_name, self.testing_labels)
        pred_score_file_name = '../cross_val/' + seen_or_unseen + '/pred_score_fold_' + k_fold + '.txt'
        np.savetxt(pred_score_file_name, pred_score_output)

        roc_fig_name = '../cross_val/' + seen_or_unseen + '/config_net_roc' + k_fold + '.png'
        plot_roc_pr_curve.plot_roc_curve(self.testing_labels, pred_score_output, roc_fig_name)
        pr_fig_name = '../cross_val/' + seen_or_unseen + '/config_net_pr' + k_fold + '.png'
        plot_roc_pr_curve.plot_pr_curve(self.testing_labels, pred_score_output, pr_fig_name)
        elapsed_time = time.time() - start_time
        print 'Total testing elapsed_time: ', elapsed_time

    def create_net_var_inf(self):
        self.var_config = tf.get_variable(name='var_config', 
                shape=[1, self.config_dim],
                initializer=tf.constant_initializer(.0))

    def construct_grad_graph_inf(self):
        self.assign_config_var_op = self.var_config.assign(self.holder_config)

        with tf.name_scope('Pred'):
            (fc1_grasp, fc2_grasp, logits) = \
                    self.conv_net_grasp(self.holder_grasp_patch, self.var_config, self.weights_grasp, 
                                        self.biases_grasp, self.strides_grasp)
            self.pred = tf.nn.sigmoid(logits)

        with tf.name_scope('GD'):
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.)
        
        with tf.name_scope('RGBD_GRAD'):
            self.config_gradients = self.optimizer.compute_gradients(logits[0, 0],
                    var_list=[self.var_config])

    def init_net_inf(self):
        #Create the network variables.
        self.create_net_var()
        
        saver = tf.train.Saver()
        self.sess_inf = tf.Session()
        #Initialize the variables
        self.init = tf.global_variables_initializer()
        self.sess_inf.run(self.init)
        # Restore trained model
        saver.restore(self.sess_inf, self.cnn_model_path)
        
        #Create variables for rgbd patches
        self.create_net_var_inf()
        #Call the network building function.
        self.construct_grad_graph_inf()
        
        tf.get_default_graph().finalize()   
        logs_path = self.logs_path + '_inf'
        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    def get_config_gradients(self, grasp_patch, config):
        self.sess_inf.run([self.assign_config_var_op], feed_dict={self.holder_config:np.array([config])})
        [gradients, pred_out] = self.sess_inf.run([self.config_gradients, self.pred], 
                feed_dict={self.holder_grasp_patch:np.array([grasp_patch]), self.keep_prob:1.})
        #print 'gradients:', gradients
        return gradients[0][0][0], pred_out[0, 0]

    def get_suc_prob(self, grasp_patch, config):
        self.sess_inf.run([self.assign_config_var_op], feed_dict={self.holder_config:np.array([config])})
        [pred_out] = self.sess_inf.run([self.pred], 
                feed_dict={self.holder_grasp_patch:np.array([grasp_patch]), self.keep_prob:1.})
        return pred_out[0, 0]


    def read_grasp_config(self):
        grasp_patches_file = h5py.File(self.grasp_patches_file_path, 'r')
        grasps_num = grasp_patches_file['grasps_number'][()] 

        grasp_configs = np.zeros((grasps_num, self.config_dim))       

        for i in xrange(grasps_num):
            print 'reading ', i
            grasp_sample_id = 'grasp_' + str(i)
            grasp_config_key = grasp_sample_id + '_preshape_true_config'
            grasp_configs[i] = grasp_patches_file[grasp_config_key]

        print 'Reading is done.'
        grasp_patches_file.close() 

        return grasp_configs 


    def update_prior_poses_client(self, prior_means):
        '''
        Client to update the GMM prior mean poses.
        '''
        #Add average grasp config
        prior_poses = []
        for config in prior_means:
            hand_config = self.proc_grasp.convert_preshape_to_full_config(config) 
            prior_poses.append(hand_config.palm_pose)

        rospy.loginfo('Waiting for service update_grasp_prior_poses.')
        rospy.wait_for_service('update_grasp_prior_poses')
        rospy.loginfo('Calling service update_grasp_prior_poses.')
        try:
            update_prior_poses_proxy = rospy.ServiceProxy('update_grasp_prior_poses', UpdatePriorPoses)
            update_prior_poses_request = UpdatePriorPosesRequest()
            update_prior_poses_request.prior_poses = prior_poses
            update_prior_poses_response = update_prior_poses_proxy(update_prior_poses_request) 
        except rospy.ServiceException, e:
            rospy.loginfo('Service update_grasp_prior_poses call failed: %s'%e)
        rospy.loginfo('Service update_grasp_prior_poses is executed.')


    def fit_prior(self):
        grasp_configs = self.read_grasp_config()
        num_components = 4
        #g = mixture.GaussianMixture(n_components=num_components, covariance_type='full', 
        #        random_state=0, init_params='random', n_init=5)
        g = mixture.GaussianMixture(n_components=num_components, covariance_type='full', 
                random_state=0, init_params='kmeans', n_init=5)
        g.fit(grasp_configs)
        #pred_prob = g.predict_proba(grasp_configs)
        #print pred_prob
        #print g.score_samples(grasp_configs)
        np.save(self.prior_path + 'covariances.npy', g.covariances_)
        np.save(self.prior_path + 'weights.npy', g.weights_)
        np.save(self.prior_path + 'means.npy', g.means_)
        print 'weights:', g.weights_
        print 'means:', g.means_
        pickle.dump(g, open(self.prior_path + 'gmm.model', 'wb'))
        self.update_prior_poses_client(g.means_)


def cross_val():
    print sys.argv
    train_or_test = sys.argv[1]
    seen_or_unseen = sys.argv[2]
    k_fold = sys.argv[3]
    train_file_name = '../cross_val/cross_val_train_' + seen_or_unseen
    test_file_name = '../cross_val/cross_val_test_' + seen_or_unseen
    train_indices, test_indices = \
            cross_validation.get_k_fold_train_test_indices(int(k_fold), train_file_name, test_file_name)
    grasp_image_config_net = GraspRgbdConfigNet()
    if train_or_test == 'train':
        grasp_image_config_net.train(train_indices, test_indices)
    elif train_or_test == 'test':
        grasp_image_config_net.test(train_indices, test_indices, seen_or_unseen, k_fold)

def train_and_test():
    grasp_image_config_net = GraspRgbdConfigNet()
    #train_or_test = 'train'
    train_or_test = 'test'
    if train_or_test == 'train':
        grasp_image_config_net.train()
    elif train_or_test == 'test':
        grasp_image_config_net.test()

if __name__ == '__main__':
    #cross_val()
    #train_and_test()
    grasp_image_config_net = GraspRgbdConfigNet()
    #grasp_image_config_net.read_rgbd_data()
    grasp_image_config_net.fit_prior()
    
