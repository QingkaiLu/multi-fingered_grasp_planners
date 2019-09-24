import tensorflow as tf
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tsne_vis
import h5py
import numpy as np
import time
#import gen_rgbd_patches as gp
import cv2
from grasp_net_interface import GraspNet

class GraspRgbdPatchesDeeperNet(GraspNet):
    def __init__(self):
        # Parameters
        self.palm_patch_rows = 200
        self.palm_patch_cols = 200
        self.finger_patch_rows = 100
        self.finger_patch_cols = 100
        self.rgbd_channels = 8
        self.rate_dec_epochs = 2000#20000
        self.training_epochs = 10000#60000
        self.batch_size = 16#32
        self.display_step = 10#0
        self.dropout_prob = 0.75
        #self.dropout_prob = 0.5
        #self.dropout_prob = 1.0
        #self.classes_num = 2
        self.classes_num = 1
        #self.read_data_ratio = 0.25
        #self.read_batch_epochs = 5000
        self.read_data_ratio = 0.5
        self.read_batch_epochs = 5000
        self.fingers_num = 4

        #rgbd_patches_save_path = '/media/kai/multi_finger_grasp_data/'
        #self.grasp_patches_file_path = rgbd_patches_save_path + 'grasps_rgbd_patches.h5'
        #self.logs_path = '/media/kai/tf_logs/multi_finger_grasp_data'
        #self.cnn_model_path = '/home/kai/Workspace/grasps_detection_CNN_GD_ws/src/' + \
        #        'prob_grasp_planner/models/cnn_hand.ckpt'       

        rgbd_patches_save_path = '/data_space/data_kai/multi_finger_sim_data/'
        self.grasp_patches_file_path = rgbd_patches_save_path + 'grasps_rgbd_patches.h5'
        self.logs_path = '/home/kai/tf_logs/multi_finger_sim_data_classification'
        self.cnn_model_path = '/home/kai/Workspace/grasps_detection_CNN_GD_ws/src/' + \
                'prob_grasp_planner/models/deeper_cnn_hand.ckpt'       

        # tf Graph input
        self.holder_palm = tf.placeholder(tf.float32, 
                [None, self.palm_patch_rows, self.palm_patch_cols, self.rgbd_channels], name = 'holder_palm')
        self.holder_fingers = tf.placeholder(tf.float32, 
                [self.fingers_num, None, self.finger_patch_rows, self.finger_patch_cols, self.rgbd_channels], name = 'holder_fingers')
        self.holder_labels = tf.placeholder(tf.float32, [None, self.classes_num], name = 'holder_labels')
        #self.holder_labels = tf.placeholder(tf.int64, [None], name = 'holder_labels')
        self.keep_prob = tf.placeholder(tf.float32, name = 'holder_keep_prob')
        self.learning_rate = tf.placeholder(tf.float32, name = 'holder_learn_rate')
               
        self.train_samples_pct = 0.9
        self.train_samples_num = -1 
        self.testing_samples_num = -1 
        self.training_labels = None
        self.training_samples_palm = None 
        self.training_samples_fingers = None 
        self.testing_labels = None
        self.testing_samples_palm = None 
        self.testing_samples_fingers = None 
        self.read_data_batch = False

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
    
    # CNN model for each finger/palm voxel
    def conv_net_finger(self, x, weights, biases):
        #Convolution 1
        conv1 = self.conv2d(x, weights['conv1'], biases['conv1'], 2)
        #Convolution 2
        conv2 = self.conv2d(conv1, weights['conv2'], biases['conv2'], 2)
        #Max pool layer
        max_pool = self.max_pool2d(conv2, 2, 2)
        #Convolution 3
        conv3 = self.conv2d(max_pool, weights['conv3'], biases['conv3'], 1)
        #FC layer 1
        fc1 = tf.add(tf.matmul(tf.reshape(conv3, [-1, int(np.prod(conv3.get_shape()[1:]))]), 
                               weights['fc1']), biases['fc1'])
        fc1 = tf.nn.relu(fc1)
        fc1 = tf.nn.dropout(fc1, self.keep_prob)
        return fc1
    
    def conv_net_hand(self, x_palm, x_fingers, w_palm, b_palm,
                        w_fingers, b_fingers, w_hand, b_hand):
        with tf.name_scope('Palm'):
            fc_palm = self.conv_net_finger(x_palm, w_palm, b_palm)
        #with tf.name_scope('F_1'):
        #    fc_f_1 = self.conv_net_finger(x_f_1, w_f_1, b_f_1)
        #with tf.name_scope('F_2'):
        #    fc_f_2 = self.conv_net_finger(x_f_2, w_f_2, b_f_2)
        fc_fingers = []
        for i in xrange(self.fingers_num):
            with tf.name_scope('F_' + str(i)):
                fc_finger = self.conv_net_finger(x_fingers[i], w_fingers[i], b_fingers[i]) 
                fc_fingers.append(fc_finger)
        #with tf.name_scope('Concat'):
        #    fc_concat = tf.concat(1, [fc_palm, fc_f_1, fc_f_2])
        with tf.name_scope('Concat'):
            fc_concat = fc_palm
            for i in xrange(self.fingers_num):
                fc_concat = tf.concat(axis=1, values=[fc_concat, fc_fingers[i]])

        with tf.name_scope('Hand'):
            #fc1 layer for concatenated fingers and palm features
            fc1 = tf.add(tf.matmul(fc_concat, w_hand['fc1']), b_hand['fc1'])
            fc1 = tf.nn.relu(fc1)
            fc1 = tf.nn.dropout(fc1, self.keep_prob)
            #fc2 layer for concatenated fingers and palm features
            fc2 = tf.add(tf.matmul(fc1, w_hand['fc2']), b_hand['fc2'])
            fc2 = tf.nn.relu(fc2)
            fc2 = tf.nn.dropout(fc2, self.keep_prob)
            # Output layer with linear activation
            out_layer = tf.matmul(fc2, w_hand['out']) + b_hand['out']

        #return (fc_concat, fc1, out_layer)
        return (fc1, fc2, out_layer)


    def create_net_var(self):
        # Store layers weight & bias
        # conv shapes for the palm.
        # 12x12x12 conv, 8 input channels, 32 outputs
        conv1_w_par_palm = [12, 12, self.rgbd_channels, 32] 
        conv1_b_par_palm = [32]
        # 6x6x6 conv, 32 input channels, 16 outputs
        conv2_w_par_palm = [6, 6, 32, 16]
        conv2_b_par_palm = [16]
        # 3x3x3 conv, 16 input channels, 8 outputs
        conv3_w_par_palm = [3, 3, 16, 8]
        conv3_b_par_palm = [8]
        # fc1, 2(conv1)*2(conv2)*2(max pool) = 8
        # 200/8 * 200/8 * 8 inputs, 32 outputs
        fc1_w_par_palm = [self.palm_patch_rows*self.palm_patch_cols/8, 64]
        fc1_b_par_palm = [64]

        #conv filter shapes for fingers.
        # 6x6x6 conv, 1 input channels, 32 outputs
        conv1_w_par_finger = [6, 6, self.rgbd_channels, 32]
        conv1_b_par_finger = [32]
        # 3x3x3 conv, 32 input channels, 16 outputs
        conv2_w_par_finger = [3, 3, 32, 16]
        conv2_b_par_finger = [16]
        # 3x3x3 conv, 16 input channels, 8 outputs
        conv3_w_par_finger = [3, 3, 16, 8]
        conv3_b_par_finger = [8]
        # fc1, 2(conv1)*2(conv2)*2(max pool) = 8
        # ceil(100/8) * ceil(100/8) * 8 inputs, 32 outputs
        fc1_w_par_finger = [int(np.ceil(self.finger_patch_rows/8.)*np.ceil(self.finger_patch_cols/8.)*8), 64]
        # fc2, 32 inputs, 16 outputs
        #fc2_w_par_finger = [32, 16]

        fc1_b_par_finger = [64]
        #fc2_b_par_finger = [16]

        self.weights_palm = {
                'conv1': tf.get_variable(name='w_conv1_palm', shape=conv1_w_par_palm,
                        initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                'conv2': tf.get_variable(name='w_conv2_palm', shape=conv2_w_par_palm, 
                    initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                'conv3': tf.get_variable(name='w_conv3_palm', shape=conv3_w_par_palm, 
                    initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                'fc1': tf.get_variable(name='w_fc1_palm', shape=fc1_w_par_palm, 
                    initializer=tf.contrib.layers.xavier_initializer()),
                }

        self.biases_palm = {
                'conv1': tf.get_variable(name='b_conv1_palm', shape=conv1_b_par_palm, 
                    #initializer=tf.constant_initializer(0.0)),
                    initializer=tf.random_normal_initializer()),
                'conv2': tf.get_variable(name='b_conv2_palm', shape=conv2_b_par_palm, 
                    #initializer=tf.constant_initializer(0.0)),
                    initializer=tf.random_normal_initializer()),
                'conv3': tf.get_variable(name='b_conv3_palm', shape=conv3_b_par_palm, 
                    #initializer=tf.constant_initializer(0.0)),
                    initializer=tf.random_normal_initializer()),
                'fc1': tf.get_variable(name='b_fc1_palm', shape=fc1_b_par_palm, 
                    initializer=tf.random_normal_initializer()),
                }

        self.weights_fingers = []
        self.biases_fingers = []
        for i in xrange(self.fingers_num):
            weights_finger = {
                    'conv1': tf.get_variable(name='w_conv1_f_' + str(i), shape=conv1_w_par_finger, 
                        initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                    'conv2': tf.get_variable(name='w_conv2_f_' + str(i), shape=conv2_w_par_finger, 
                        initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                    'conv3': tf.get_variable(name='w_conv3_f_' + str(i), shape=conv3_w_par_finger, 
                        initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                    'fc1': tf.get_variable(name='w_fc1_f_' + str(i), shape=fc1_w_par_finger, 
                        initializer=tf.contrib.layers.xavier_initializer()),
                    }

            biases_finger = {
                    'conv1': tf.get_variable(name='b_conv1_f_' + str(i), shape=conv1_b_par_finger, 
                        #initializer=tf.constant_initializer(0.0)),
                        initializer=tf.random_normal_initializer()),
                    'conv2': tf.get_variable(name='b_conv2_f_' + str(i), shape=conv2_b_par_finger, 
                        #initializer=tf.constant_initializer(0.0)),
                        initializer=tf.random_normal_initializer()),
                    'conv3': tf.get_variable(name='b_conv3_f_' + str(i), shape=conv3_b_par_finger, 
                        #initializer=tf.constant_initializer(0.0)),
                        initializer=tf.random_normal_initializer()),
                    'fc1': tf.get_variable(name='b_fc1_f_' + str(i), shape=fc1_b_par_finger, 
                        initializer=tf.random_normal_initializer()),
                    }

            self.weights_fingers.append(weights_finger)
            self.biases_fingers.append(biases_finger)

        self.weights_hand = {
                # fc1 32*3=96, 96 inputs, 32 outputs
                'fc1': tf.get_variable(name='w_fc1_hand', 
                    shape=[fc1_w_par_finger[-1] * self.fingers_num + fc1_w_par_palm[-1], 64], 
                    initializer=tf.contrib.layers.xavier_initializer()),
                'fc2': tf.get_variable(name='w_fc2_hand', 
                    shape=[64, 32], 
                    initializer=tf.contrib.layers.xavier_initializer()),
                # output layer, 32 inputs, 2 output
                'out': tf.get_variable(name='w_out_hand', shape=[32, self.classes_num], 
                    initializer=tf.contrib.layers.xavier_initializer()),
                }

        self.biases_hand = {
                'fc1': tf.get_variable(name='b_fc1_hand', shape=[64], 
                    #initializer=tf.constant_initializer(0.0)),
                    initializer=tf.random_normal_initializer()),
                'fc2': tf.get_variable(name='b_fc2_hand', shape=[32], 
                    #initializer=tf.constant_initializer(0.0)),
                    initializer=tf.random_normal_initializer()),
                'out': tf.get_variable(name='b_out_hand', shape=[self.classes_num], 
                    #initializer=tf.constant_initializer(0.0)),
                    initializer=tf.random_normal_initializer()),
                }

    def cost_function(self):
        with tf.name_scope('Pred'):
            (fc_concat_hand, fc1_hand, logits) = self.conv_net_hand(self.holder_palm, self.holder_fingers,
                                  self.weights_palm, self.biases_palm, self.weights_fingers, self.biases_fingers, 
                                  self.weights_hand, self.biases_hand)

            self.pred = tf.nn.sigmoid(logits)
            #self.pred = logits

        with tf.name_scope('F2Vis'):
            self.feature_to_vis = tf.concat(axis=1, values=[fc_concat_hand, fc1_hand]) 
        
        with tf.name_scope('Pred_error'):
            self.pred_error = tf.reduce_mean(tf.abs(self.holder_labels - self.pred))
        
        with tf.name_scope('Pred_error_test'):
            self.pred_error_test = tf.reduce_mean(tf.abs(self.holder_labels - self.pred))
        
        with tf.name_scope('Cost'):
            #logistic regression (or cross entropy) loss
            #http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
            #http://stackoverflow.com/questions/12578336/why-is-the-bias-term-not-regularized-in-ridge-regression
            #http://www.ritchieng.com/machine-learning/deep-learning/tensorflow/regularization/
            #self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, self.holder_labels))
            #self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, self.holder_labels))
            #https://www.tensorflow.org/api_docs/python/nn/classification#sigmoid_cross_entropy_with_logits
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
        # Mean and std of the fc1 layer of two fingers and the palm
        #palm_w_mean, palm_w_var = tf.nn.moments(self.weights_palm['fc1'], axes=[0,1])
        #palm_w_mean_sum = tf.scalar_summary('palm_w_mean', palm_w_mean)
        #palm_w_var_sum = tf.scalar_summary('palm_w_var', palm_w_var)
        #f_1_w_mean, f_1_w_var = tf.nn.moments(self.weights_f_1['fc1'], axes=[0,1])
        #f_1_w_mean_sum = tf.scalar_summary('f_1_w_mean', f_1_w_mean)
        #f_1_w_var_sum = tf.scalar_summary('f_1_w_var', f_1_w_var)
        #f_2_w_mean, f_2_w_var = tf.nn.moments(self.weights_f_2['fc1'], axes=[0,1])
        #f_2_w_mean_sum = tf.scalar_summary('f_2_w_mean', f_2_w_mean)
        #f_2_w_var_sum = tf.scalar_summary('f_2_w_var', f_2_w_var)
        #self.train_summary = tf.merge_summary([loss_sum, pred_err_sum, learn_rate_sum, palm_w_mean_sum,
        #    palm_w_var_sum, f_1_w_mean_sum, f_1_w_var_sum, f_2_w_mean_sum, f_2_w_var_sum])

        # Testing prediction errors
        self.pred_err_test_sum = tf.summary.scalar('pred_err_test', self.pred_error_test)

    def read_rgbd_data(self):
        grasp_patches_file = h5py.File(self.grasp_patches_file_path, 'r')
        grasps_num = grasp_patches_file['grasps_number'][()] 

        grasp_labels = np.zeros((grasps_num, self.classes_num))       
        palm_rgbd_patches = np.zeros((grasps_num, self.palm_patch_rows, self.palm_patch_cols, self.rgbd_channels))
        fingers_rgbd_patches = np.zeros((self.fingers_num, grasps_num, self.finger_patch_rows, 
                                                    self.finger_patch_cols, self.rgbd_channels))
        #for i in xrange(grasps_num):
        #    #print 'reading ', i
        #    grasp_sample_id = 'grasp_' + str(i)
        #    grasp_label_key = grasp_sample_id + '_grasp_label'
        #    palm_patch_key = grasp_sample_id + '_palm_patch'
        #    finger_tip_patches_key = grasp_sample_id + '_finger_tip_patches'
        #    grasp_labels[i] = grasp_patches_file[grasp_label_key]
        #    palm_rgbd_patches[i] = grasp_patches_file[palm_patch_key]
        #    fingers_rgbd_patches[:, i] = grasp_patches_file[finger_tip_patches_key] 

        grasp_rand_indices = random.sample(range(0, grasps_num), grasps_num)
        #grasp_rand_indices = range(0, grasps_num)
        for i, j in enumerate(grasp_rand_indices):
            #print 'reading ', i
            grasp_sample_id = 'grasp_' + str(j)
            grasp_label_key = grasp_sample_id + '_grasp_label'
            palm_patch_key = grasp_sample_id + '_palm_patch'
            finger_tip_patches_key = grasp_sample_id + '_finger_tip_patches'
            grasp_labels[i] = grasp_patches_file[grasp_label_key]
            palm_rgbd_patches[i] = grasp_patches_file[palm_patch_key]
            fingers_rgbd_patches[:, i] = grasp_patches_file[finger_tip_patches_key] 

        print 'Reading is done.'
        grasp_patches_file.close() 

        print 'grasp_labels.shape: ', grasp_labels.shape
        print 'positive #:', np.sum(grasp_labels[:,0]==1)
        print 'negative #:', np.sum(grasp_labels[:,0]==0)

        self.train_samples_num = int(grasps_num * self.train_samples_pct)
        self.testing_samples_num = grasps_num - self.train_samples_num

        ##Shuffle the data
        #np.random.shuffle(grasp_labels)
        #np.random.shuffle(palm_rgbd_patches)
        #map(np.random.shuffle, fingers_rgbd_patches)
        
        self.training_labels = grasp_labels[0:self.train_samples_num]
        self.training_samples_palm = palm_rgbd_patches[0:self.train_samples_num]                                   
        self.training_samples_fingers = fingers_rgbd_patches[:, 0:self.train_samples_num]                                   

        #self.training_labels = grasp_labels
        #self.training_samples_palm = palm_rgbd_patches                                   
        #self.training_samples_fingers = fingers_rgbd_patches                                   

        self.testing_labels = grasp_labels[self.train_samples_num:grasps_num]                                                                
        self.testing_samples_palm = palm_rgbd_patches[self.train_samples_num:grasps_num]
        self.testing_samples_fingers = fingers_rgbd_patches[:, self.train_samples_num:grasps_num]
        
        print self.training_samples_palm.shape
        print self.training_samples_fingers.shape
        print self.training_labels.shape
        print 'training positive #:', np.sum(self.training_labels[:, 0]==1)
        print self.testing_samples_palm.shape
        print self.testing_samples_fingers.shape
        print self.testing_labels.shape
        print 'testing positive #:', np.sum(self.testing_labels[:, 0]==1)

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
        
        self.training_labels = np.zeros((self.train_samples_num, self.classes_num))
        self.training_samples_palm = np.zeros((self.train_samples_num, self.palm_patch_rows, self.palm_patch_cols, self.rgbd_channels))
        self.training_samples_f_1 = np.zeros((self.train_samples_num, self.finger_patch_rows, self.finger_patch_cols, self.rgbd_channels))
        self.training_samples_f_2 = np.zeros((self.train_samples_num, self.finger_patch_rows, self.finger_patch_cols, self.rgbd_channels))
        
        self.testing_labels = np.zeros((self.testing_samples_num, self.classes_num))
        self.testing_samples_palm = np.zeros((self.testing_samples_num, self.palm_patch_rows, self.palm_patch_cols, self.rgbd_channels))
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
            self.training_samples_palm[i] = patch_file['palm_' + str(grasp_idx)]
            self.training_samples_f_1[i] = patch_file['f_1_' + str(grasp_idx)]
            self.training_samples_f_2[i] = patch_file['f_2_' + str(grasp_idx)]
            self.training_labels[i, 0] = patch_file['labels'][grasp_idx] 

        for i, grasp_idx in enumerate(test_batch_indices):
            #print 'read test sample: ', i, grasp_idx
            self.testing_samples_palm[i] = patch_file['palm_' + str(grasp_idx)]
            self.testing_samples_f_1[i] = patch_file['f_1_' + str(grasp_idx)]
            self.testing_samples_f_2[i] = patch_file['f_2_' + str(grasp_idx)]
            self.testing_labels[i, 0] = patch_file['labels'][grasp_idx] 

        #for i, grasp_idx in enumerate(train_batch_indices):
        #    print 'read train sample: ', i, grasp_idx
        #    if i % 2 == 0:
        #        self.training_samples_palm[i] = patch_file['palm_' + str(grasp_idx)]
        #        self.training_samples_f_1[i] = patch_file['f_1_' + str(grasp_idx)]
        #        self.training_samples_f_2[i] = patch_file['f_2_' + str(grasp_idx)]
        #        self.training_labels[i, 0] = patch_file['labels'][grasp_idx] 
        #    else:
        #        self.training_samples_palm[i] = patch_file['palm_' + str(grasp_idx)]
        #        self.training_samples_f_1[i] = patch_file['f_2_' + str(grasp_idx)]
        #        self.training_samples_f_2[i] = patch_file['f_1_' + str(grasp_idx)]
        #        self.training_labels[i, 0] = patch_file['labels'][grasp_idx] 


        #for i, grasp_idx in enumerate(test_batch_indices):
        #    print 'read test sample: ', i, grasp_idx
        #    if i % 2 == 0:
        #        self.testing_samples_palm[i] = patch_file['palm_' + str(grasp_idx)]
        #        self.testing_samples_f_1[i] = patch_file['f_1_' + str(grasp_idx)]
        #        self.testing_samples_f_2[i] = patch_file['f_2_' + str(grasp_idx)]
        #        self.testing_labels[i, 0] = patch_file['labels'][grasp_idx] 
        #    else:
        #        self.testing_samples_palm[i] = patch_file['palm_' + str(grasp_idx)]
        #        self.testing_samples_f_1[i] = patch_file['f_2_' + str(grasp_idx)]
        #        self.testing_samples_f_2[i] = patch_file['f_1_' + str(grasp_idx)]
        #        self.testing_labels[i, 0] = patch_file['labels'][grasp_idx] 

        print 'reading is done.'
        patch_file.close() 

        self.training_labels[self.training_labels == -1] = 0
        self.testing_labels[self.testing_labels == -1] = 0

        print self.training_samples_palm.shape
        print self.training_samples_f_1.shape
        print self.training_samples_f_2.shape
        print self.training_labels.shape
        print 'training positive #:', np.sum(self.training_labels[:, 0]==1)
        
        print self.testing_samples_palm.shape
        print self.testing_samples_f_1.shape
        print self.testing_samples_f_2.shape
        print self.testing_labels.shape
        print 'testing positive #:', np.sum(self.testing_labels[:, 0]==1)


    def feed_dict_func(self, train, l_rate = .0, batch_idx = None):
        ys = None
        xs_palm = None
        xs_fingers = None
        if train:
            #rand_indices = random.sample(range(0, self.train_samples_num), self.batch_size)
            #ys = self.training_labels[rand_indices]
            #xs_palm = self.training_samples_palm[rand_indices]
            #xs_fingers = self.training_samples_fingers[:, rand_indices]

            contain_positive_label = False
            while not contain_positive_label:
                rand_indices = random.sample(range(0, self.train_samples_num), self.batch_size)
                ys = self.training_labels[rand_indices]
                contain_positive_label = np.sum(ys) > 3

            xs_palm = self.training_samples_palm[rand_indices]
            xs_fingers = self.training_samples_fingers[:, rand_indices]

            k = self.dropout_prob
        else:
            if batch_idx is None:
                rand_indices = random.sample(range(0, self.testing_samples_num), self.batch_size)
                ys = self.testing_labels[rand_indices]
                xs_palm = self.testing_samples_palm[rand_indices]
                xs_fingers = self.testing_samples_fingers[:, rand_indices]
            else:
                ys = self.testing_labels[batch_idx * self.batch_size: 
                                     min((batch_idx + 1) * self.batch_size, self.testing_samples_num)]
                xs_palm = self.testing_samples_palm[batch_idx * self.batch_size: 
                                     min((batch_idx + 1) * self.batch_size, self.testing_samples_num)]
                xs_fingers = self.testing_samples_fingers[:, batch_idx * self.batch_size: 
                                     min((batch_idx + 1) * self.batch_size, self.testing_samples_num)]
    
            k = 1.0

        return {self.holder_labels: ys, self.holder_palm: xs_palm, self.holder_fingers: xs_fingers,
                self.keep_prob: k, self.learning_rate: l_rate}
    
    def train(self):
        #Read data
        if self.read_data_batch:
            self.read_rgbd_data_batch(self.read_data_ratio)
        #elif not train:
        #    self.read_rgbd_test_data()
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
        #return

        logs_path = self.logs_path + '_train'
    
        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        
        start_time = time.time()
        train_costs = []
        train_pred_errors = []
        test_pred_errors = []
        learn_rate = 0.001
        #learn_rate = 0.001
        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            # Training cycle
            for epoch in range(self.training_epochs):
                if epoch % self.rate_dec_epochs == 0 and epoch != 0:
                    learn_rate *= 0.1
                if self.read_data_batch and epoch % self.read_batch_epochs == 0 and epoch != 0:
                    self.read_rgbd_data_batch(self.read_data_ratio)

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
                    #print 'train pred std: ', np.std(np.argmax(pred_train_output, 1))
                    print 'train pred std: ', np.std(pred_train_output[:, 0])
                    print 'pred_error_test: ', pred_error_test_output
                    #print 'test pred std: ', np.std(np.argmax(pred_test_output, 1))
                    print 'test pred std: ', np.std(pred_test_output[:, 0])

                #if epoch % (1000 * self.display_step) == 0:
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

    def test(self):
        #Read data
        self.read_rgbd_data()
        #Create the variables.
        self.create_net_var()
        #Call the network building function.
        self.cost_function()
        #Initialize the variables
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        tf.get_default_graph().finalize()   
        #return

        logs_path = self.logs_path + '_test'
        
        start_time = time.time()
        learn_rate = 0.001
        #learn_rate = 0.0001

        with tf.Session() as sess:
            # Test model
            saver.restore(sess, self.cnn_model_path)
            pred_output = np.array([]).reshape(0, self.classes_num)
            #f2vis_output = np.zeros([self.testing_samples_num, 32*(self.fingers_num + 1) + 32])
            f2vis_output = np.zeros([self.testing_samples_num, 32 + 64])
            for i in xrange(np.ceil(float(self.testing_samples_num) / self.batch_size).astype(int)):
                [b_hand, w_hand, f2vis_batch_output, pred_batch_output] = sess.run(
                        [self.biases_hand['out'], self.weights_hand['out'], self.feature_to_vis, self.pred],
                        feed_dict=self.feed_dict_func(False, .0, i))
                #print pred_batch_output.shape
                #print 'bias_output: ', b_hand
                #print 'weights_output: ', w_hand
                #print 'pred_batch_output: ', pred_batch_output
                pred_output = np.concatenate((pred_output, pred_batch_output))
                #print f2vis_batch_output.shape
                start_idx = i * self.batch_size
                f2vis_output[start_idx : start_idx + len(f2vis_batch_output), :] = f2vis_batch_output
            #pred_output = pred_output[0:200]
            #self.testing_labels = self.testing_labels[0:200]
            print 'pred_output: ', pred_output
            print 'self.testing_labels: ', self.testing_labels
            #print 'np.argmax(self.testing_labels, 1):' , np.argmax(self.testing_labels, 1) 
            #print np.argmax(self.testing_labels, 1).shape
            #print 'np.argmax(pred_output, 1):' , np.argmax(pred_output, 1)  
            #print np.argmax(pred_output, 1).shape
            #pred_errors_all = np.abs(np.argmax(self.testing_labels, 1) - np.argmax(pred_output, 1))
            #avg_pred_error_test = np.mean(pred_errors_all)
            #print('avg_pred_error_test:', avg_pred_error_test)
            pred_output[pred_output > 0.5] = 1.
            pred_output[pred_output <= 0.5] = 0.
            print 'binary pred_output: ', pred_output
            print 'pred_output.shape: ', pred_output.shape
            print 'self.testing_labels.shape: ', self.testing_labels.shape
            pred_errors_all = np.abs(pred_output - self.testing_labels) 
            avg_pred_error_test = np.mean(pred_errors_all)
            print 'avg_pred_error_test:', avg_pred_error_test
            print np.sum(self.testing_labels), np.sum(pred_output)

            print f2vis_output.shape, self.testing_labels.shape
            #tsne_vis.tsne_vis(f2vis_output, np.argmax(self.testing_labels, 1), '../models/tsne/test_tsne.png', True)
            #tsne_vis.tsne_vis(f2vis_output, np.argmax(pred_output, 1), '../models/tsne/test_tsne.png', True)
            tsne_vis.tsne_vis(f2vis_output, self.testing_labels[:, 0], '../models/tsne/test_tsne_gt.png', True)
            tsne_vis.tsne_vis(f2vis_output, pred_output[:, 0], '../models/tsne/test_tsne_pred.png', True)

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.plot(np.linspace(1,self.testing_samples_num,self.testing_samples_num), self.testing_labels, 'ro', label='gt labels')
            ax.plot(np.linspace(1,self.testing_samples_num,self.testing_samples_num), pred_output, 'bo', label='pred labels')
            ax.plot(np.linspace(1,self.testing_samples_num,self.testing_samples_num), pred_errors_all, 'k*', label='pred errors')
    #         plt.legend(handles=[gt_plot, pred_plot, error_plot])
            legend = ax.legend(loc='upper right', shadow=True)
            frame = legend.get_frame()
            frame.set_facecolor('0.90')
            fig.savefig('../models/labels_test.png')
            plt.clf()
            plt.close(fig)
        
        elapsed_time = time.time() - start_time
        print 'Total testing elapsed_time: ', elapsed_time
    
    def create_net_var_inf(self):
        self.var_palm = tf.get_variable(name='var_palm', 
                shape=[1, self.palm_patch_rows, self.palm_patch_cols, self.rgbd_channels],
                initializer=tf.constant_initializer(.0))
        #self.var_f_1 = tf.get_variable(name='var_f_1', 
        #        shape=[1, self.finger_patch_rows, self.finger_patch_cols, self.rgbd_channels],
        #        initializer=tf.constant_initializer(.0))
        #self.var_f_2 = tf.get_variable(name='var_f_2', 
        #        shape=[1, self.finger_patch_rows, self.finger_patch_cols, self.rgbd_channels],
        #        initializer=tf.constant_initializer(.0))
        #self.var_fingers = tf.get_variable(name='var_fingers', 
        #        shape=[self.fingers_num, 1, self.finger_patch_rows, self.finger_patch_cols, self.rgbd_channels],
        #        initializer=tf.constant_initializer(.0))
        self.var_fingers_list = []
        for i in xrange(self.fingers_num):
            var_finger = tf.get_variable(name='var_f_' + str(i), 
                    shape=[1, self.finger_patch_rows, self.finger_patch_cols, self.rgbd_channels],
                    initializer=tf.constant_initializer(.0))
            self.var_fingers_list.append(var_finger) 


    def construct_grad_graph_inf(self):
        self.assign_palm_var_op = self.var_palm.assign(self.holder_palm)
        #self.assign_f_1_var_op = self.var_f_1.assign(self.holder_f_1)
        #self.assign_f_2_var_op = self.var_f_2.assign(self.holder_f_2)
        #self.assign_fingers_var_op = self.var_fingers.assign(self.holder_fingers)
        self.assign_fingers_var_ops_list = []
        for i in xrange(self.fingers_num):
            assign_finger_var_op = self.var_fingers_list[i].assign(self.holder_fingers[i])
            self.assign_fingers_var_ops_list.append(assign_finger_var_op)

        with tf.name_scope('Pred'):
            #(fc_concat_hand, fc1_hand, logits) = self.conv_net_hand(self.var_palm, self.var_f_1, self.var_f_2,
            #                      self.weights_palm, self.biases_palm, self.weights_f_1, self.biases_f_1, 
            #                      self.weights_f_2, self.biases_f_2, self.weights_hand, self.biases_hand)
            (fc_concat_hand, fc1_hand, logits) = self.conv_net_hand(self.var_palm, self.var_fingers_list,
                                  self.weights_palm, self.biases_palm, self.weights_fingers, self.biases_fingers, 
                                  self.weights_hand, self.biases_hand)

            #self.pred = tf.nn.softmax(logits)
            self.pred = tf.nn.sigmoid(logits)

        with tf.name_scope('GD'):
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.)
        
        with tf.name_scope('RGBD_GRAD'):
            variables_list = [self.var_palm] + self.var_fingers_list
            #for i in xrange(self.fingers_num):
            #    variables_list.append(self.var_fingers[i])
            self.rgbd_gradients = self.optimizer.compute_gradients(logits[0, 0],
                    var_list=variables_list)

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
        #logs_path = '/media/kai/tf_logs/grasp_net_hand_cornell_inf'
        logs_path = self.logs_path + '_inf'
        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())


    def get_rgbd_gradients(self, palm_patch, finger_patches):
        assign_palm_fingers_list = [self.assign_palm_var_op] + self.assign_fingers_var_ops_list
        self.sess_inf.run(assign_palm_fingers_list, feed_dict={self.holder_palm:np.array([palm_patch]), 
                    self.holder_fingers:np.array(list(map(lambda x: [x], finger_patches)))})
        
        [gradients, pred_out] = self.sess_inf.run([self.rgbd_gradients, self.pred], feed_dict={self.keep_prob:1.})
        gradients_fingers = []
        for i in xrange(self.fingers_num):
            gradients_fingers.append(gradients[i+1][0][0])
        return gradients[0][0][0], gradients_fingers, pred_out[0, 0]

def main():
    grasp_rgbd_deeper_net = GraspRgbdPatchesDeeperNet()
    grasp_rgbd_deeper_net.train()
    #grasp_rgbd_deeper_net.test()

if __name__ == '__main__':
    main()

