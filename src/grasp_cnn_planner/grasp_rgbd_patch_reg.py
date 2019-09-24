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

# One network that directly takes the large subimage and 
# predicate the grasp preshape configuration as a regression problem.
class GraspRgbdPatchesShallowRegNet(GraspNet):
    def __init__(self):
        # Parameters
        self.grasp_patch_rows = 400
        self.grasp_patch_cols = 400
        self.rgbd_channels = 8
        self.rate_dec_epochs = 200#20000
        self.training_epochs = 1000#60000
        self.batch_size = 4#16
        self.display_step = 10#0
        self.dropout_prob = 0.75
        self.config_dim = 11
        #self.read_data_ratio = 0.25
        #self.read_batch_epochs = 5000
        self.read_data_ratio = 0.5
        self.read_batch_epochs = 5000

        rgbd_patches_save_path = '/data_space/data_kai/multi_finger_sim_data/'
        self.grasp_patches_file_path = rgbd_patches_save_path + 'grasps_patches.h5'
        self.grasp_data_file_path = rgbd_patches_save_path + 'grasp_data.h5'
        self.logs_path = '/home/kai/tf_logs/multi_finger_sim_data_reg'
        self.cnn_model_path = '/home/kai/Workspace/grasps_detection_CNN_GD_ws/src/' + \
                'prob_grasp_planner/models/cnn_grasp_regression.ckpt'       

        # tf Graph input
        self.holder_grasp_patch = tf.placeholder(tf.float32, 
                [None, self.grasp_patch_rows, self.grasp_patch_cols, self.rgbd_channels], name='holder_grasp_patch')
        self.holder_config = tf.placeholder(tf.float32, [None, self.config_dim], name='holder_config')
        self.keep_prob = tf.placeholder(tf.float32, name='holder_keep_prob')
        self.learning_rate = tf.placeholder(tf.float32, name='holder_learn_rate')
               
        self.train_samples_pct = 0.9
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
    
    def conv_net_grasp(self, x_grasp, w_grasp, b_grasp):
        with tf.name_scope('Grasp'):
            #Convolution 1
            conv1 = self.conv2d(x_grasp, w_grasp['conv1'], b_grasp['conv1'], 2)
            #Convolution 2
            conv2 = self.conv2d(conv1, w_grasp['conv2'], b_grasp['conv2'], 2)
            #Max pool layer
            max_pool = self.max_pool2d(conv2, 2, 2)
            #FC layer 1
            fc1 = tf.add(tf.matmul(tf.reshape(max_pool, [-1, int(np.prod(max_pool.get_shape()[1:]))]), 
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
        # conv shapes for the palm.
        # 12x12x12 conv, 8 input channels, 32 outputs
        conv1_w_par_grasp = [12, 12, self.rgbd_channels, 32] 
        conv1_b_par_grasp = [32]
        # 6x6x6 conv, 32 input channels, 8 outputs
        conv2_w_par_grasp = [6, 6, 32, 8]
        conv2_b_par_grasp = [8]
        # fc1, 2(conv1)*2(conv2)*2(max pool) = 8
        # 200/8 * 200/8 * 8 inputs, 32 outputs
        self.fc1_neurons_num = 32
        fc1_w_par_grasp = [self.grasp_patch_rows*self.grasp_patch_cols/8, self.fc1_neurons_num]
        self.fc2_neurons_num = 32
        fc2_w_par_grasp = [self.fc1_neurons_num, self.fc2_neurons_num]
        out_w_par_grasp = [self.fc2_neurons_num, self.config_dim]

        self.weights_grasp = {
                'conv1': tf.get_variable(name='w_conv1_grasp', shape=conv1_w_par_grasp,
                        initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                'conv2': tf.get_variable(name='w_conv2_grasp', shape=conv2_w_par_grasp, 
                    initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                'fc1': tf.get_variable(name='w_fc1_grasp', shape=fc1_w_par_grasp, 
                    initializer=tf.contrib.layers.xavier_initializer()),
                # fc1 32*3=96, 96 inputs, 32 outputs
                'fc2': tf.get_variable(name='w_fc2_grasp', 
                    shape=fc2_w_par_grasp, 
                    initializer=tf.contrib.layers.xavier_initializer()),
                # output layer, 32 inputs, 2 output
                'out': tf.get_variable(name='w_out_grasp', shape=out_w_par_grasp, 
                    initializer=tf.contrib.layers.xavier_initializer()),
                }

        self.biases_grasp = {
                'conv1': tf.get_variable(name='b_conv1_grasp', shape=conv1_b_par_grasp, 
                    #initializer=tf.constant_initializer(0.0)),
                    initializer=tf.random_normal_initializer()),
                'conv2': tf.get_variable(name='b_conv2_grasp', shape=conv2_b_par_grasp, 
                    #initializer=tf.constant_initializer(0.0)),
                    initializer=tf.random_normal_initializer()),
                'fc1': tf.get_variable(name='b_fc1_grasp', shape=[self.fc1_neurons_num], 
                    initializer=tf.random_normal_initializer()),
                'fc2': tf.get_variable(name='b_fc2_grasp', shape=[self.fc2_neurons_num], 
                    #initializer=tf.constant_initializer(0.0)),
                    initializer=tf.random_normal_initializer()),
                'out': tf.get_variable(name='b_out_grasp', shape=[self.config_dim], 
                    #initializer=tf.constant_initializer(0.0)),
                    initializer=tf.random_normal_initializer()),
                }

    def cost_function(self):
        with tf.name_scope('Pred'):
            (fc1_grasp, fc2_grasp, logits) = self.conv_net_grasp(self.holder_grasp_patch, self.weights_grasp, self.biases_grasp)

            #self.pred = tf.nn.sigmoid(logits)
            self.pred = logits

        with tf.name_scope('F2Vis'):
            self.feature_to_vis = tf.concat(axis=1, values=[fc1_grasp, fc2_grasp]) 
        
        with tf.name_scope('Pred_error'):
            self.pred_error = tf.reduce_mean(tf.abs(self.holder_config - self.pred))
        
        with tf.name_scope('Pred_error_test'):
            self.pred_error_test = tf.reduce_mean(tf.abs(self.holder_config - self.pred))
        
        with tf.name_scope('Cost'):
            #self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.holder_config))
            self.cost = tf.reduce_sum(tf.pow(self.pred - self.holder_config, 2)) + \
                    self.alpha_ridge * tf.reduce_sum(tf.pow(self.weights_grasp['out'], 2))

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
        grasp_data_file = h5py.File(self.grasp_data_file_path, 'r')
        suc_grasps_num = grasp_data_file['suc_grasps_num'][()] 
        grasp_data_file.close()

        grasp_patches_file = h5py.File(self.grasp_patches_file_path, 'r')
        grasps_num = grasp_patches_file['grasps_number'][()] 

        grasp_configs = np.zeros((suc_grasps_num, self.config_dim))       
        grasp_patches = np.zeros((suc_grasps_num, self.grasp_patch_rows, self.grasp_patch_cols, self.rgbd_channels))

        suc_grasp_id = 0
        for i in xrange(grasps_num):
            #print 'reading ', i
            grasp_sample_id = 'grasp_' + str(i)
            grasp_label_key = grasp_sample_id + '_grasp_label'
            grasp_label = grasp_patches_file[grasp_label_key][()]
            if grasp_label == 1:
                grasp_patch_key = grasp_sample_id + '_grasp_patch'
                grasp_patches[suc_grasp_id] = grasp_patches_file[grasp_patch_key]
                grasp_config_key = grasp_sample_id + '_preshape_true_config'
                grasp_configs[suc_grasp_id] = grasp_patches_file[grasp_config_key]
                suc_grasp_id += 1

        #grasp_rand_indices = random.sample(range(0, grasps_num), grasps_num)
        ##grasp_rand_indices = range(0, grasps_num)
        #suc_grasp_id = 0
        #for i in grasp_rand_indices:
        #    print 'reading ', i
        #    grasp_sample_id = 'grasp_' + str(i)
        #    grasp_label_key = grasp_sample_id + '_grasp_label'
        #    grasp_label = grasp_patches_file[grasp_label_key][()]
        #    if grasp_label == 1:
        #        grasp_patch_key = grasp_sample_id + '_grasp_patch'
        #        grasp_patches[suc_grasp_id] = grasp_patches_file[grasp_patch_key]
        #        grasp_config_key = grasp_sample_id + '_preshape_true_config'
        #        grasp_configs[suc_grasp_id] = grasp_patches_file[grasp_config_key]
        #        suc_grasp_id += 1

        print 'Reading is done.'
        grasp_patches_file.close() 

        #print grasp_configs
        print 'grasp_configs.shape: ', grasp_configs.shape

        self.train_samples_num = int(suc_grasps_num * self.train_samples_pct)
        self.testing_samples_num = suc_grasps_num - self.train_samples_num
        
        self.training_configs = grasp_configs[0:self.train_samples_num]
        self.training_grasp_samples = grasp_patches[0:self.train_samples_num]                                   

        #self.training_configs = grasp_configs
        #self.training_grasp_samples = grasp_patches                                   
        #self.training_samples_fingers = fingers_rgbd_patches                                   
        
        self.testing_configs = grasp_configs[self.train_samples_num:suc_grasps_num]                                                                
        self.testing_grasp_samples = grasp_patches[self.train_samples_num:suc_grasps_num]
        
        print self.training_grasp_samples.shape
        print self.training_configs.shape
        print self.testing_grasp_samples.shape
        print self.testing_configs.shape

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

    def feed_dict_func(self, train, l_rate = .0, batch_idx = None):
        ys = None
        xs_grasp = None
        if train:
            rand_indices = random.sample(range(0, self.train_samples_num), self.batch_size)
            ys = self.training_configs[rand_indices]
            xs_grasp = self.training_grasp_samples[rand_indices]

            k = self.dropout_prob
        else:
            if batch_idx is None:
                rand_indices = random.sample(range(0, self.testing_samples_num), self.batch_size)
                ys = self.testing_configs[rand_indices]
                xs_grasp = self.testing_grasp_samples[rand_indices]
            else:
                ys = self.testing_configs[batch_idx * self.batch_size: 
                                     min((batch_idx + 1) * self.batch_size, self.testing_samples_num)]
                xs_grasp = self.testing_grasp_samples[batch_idx * self.batch_size: 
                                     min((batch_idx + 1) * self.batch_size, self.testing_samples_num)]
    
            k = 1.0

        return {self.holder_config: ys, self.holder_grasp_patch: xs_grasp,
                self.keep_prob: k, self.learning_rate: l_rate}
    
    def train(self):
        #Read data
        if self.read_data_batch:
            self.read_rgbd_data_batch(self.read_data_ratio)
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

                [_, cost_output, pred_error_output, train_summary_output, f2vis_train_output, configs_train_output, pred_train_output] = \
                        sess.run([self.optimizer, self.cost, self.pred_error, self.train_summary, self.feature_to_vis, 
                            self.holder_config, self.pred], feed_dict=self.feed_dict_func(True, learn_rate))

                # Write logs at every iteration
                summary_writer.add_summary(train_summary_output, epoch)
    
                # Display logs per epoch step
                train_costs.append(cost_output)
                train_pred_errors.append(pred_error_output)

                [pred_error_test_output, pred_err_test_sum_output, f2vis_output_test, configs_test_output, pred_test_output] = sess.run(
                        [self.pred_error_test, self.pred_err_test_sum, self.feature_to_vis, self.holder_config, self.pred], 
                        feed_dict=self.feed_dict_func(False))
                test_pred_errors.append(pred_error_test_output)
                summary_writer.add_summary(pred_err_test_sum_output, epoch)

                if epoch % self.display_step == 0:
                    print 'epoch: ', epoch
                    print 'configs_train_output:', configs_train_output
                    print 'pred_train_output', pred_train_output
                    print 'configs_test_output:', configs_test_output
                    print 'pred_test_output', pred_test_output
                    print 'train_cost: ', cost_output
                    print 'pred_error_train: ', pred_error_output
                    print 'train pred std: ', np.std(pred_train_output[:, 0])
                    print 'pred_error_test: ', pred_error_test_output
                    print 'test pred std: ', np.std(pred_test_output[:, 0])

                #if epoch % (10 * self.display_step) == 0:
                #    tsne_vis.tsne_vis(f2vis_train_output, labels_train_output[:, 0], 
                #            '../models/tsne/train_tsne_' + str(epoch) + '_gt.png')
                #    tsne_vis.tsne_vis(f2vis_output_test, labels_test_output[:, 0], 
                #            '../models/tsne/test_tsne_' + str(epoch) + '_gt.png')
                #    tsne_vis.tsne_vis(f2vis_train_output, pred_train_output[:, 0], 
                #            '../models/tsne/train_tsne_' + str(epoch) + '_pred.png')
                #    tsne_vis.tsne_vis(f2vis_output_test, pred_test_output[:, 0], 
                #            '../models/tsne/test_tsne_' + str(epoch) + '_pred.png')
                    

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

        logs_path = self.logs_path + '_test'
        
        start_time = time.time()
        learn_rate = 0.001
        #learn_rate = 0.0001

        with tf.Session() as sess:
            # Test model
            saver.restore(sess, self.cnn_model_path)
            pred_output = np.array([]).reshape(0, self.config_dim)
            f2vis_output = np.zeros([self.testing_samples_num, self.fc1_neurons_num + self.fc2_neurons_num])
            for i in xrange(np.ceil(float(self.testing_samples_num) / self.batch_size).astype(int)):
                [b_grasp, w_grasp, f2vis_batch_output, pred_batch_output] = sess.run(
                        [self.biases_grasp['out'], self.weights_grasp['out'], self.feature_to_vis, self.pred],
                        feed_dict=self.feed_dict_func(False, .0, i))
                pred_output = np.concatenate((pred_output, pred_batch_output))
                start_idx = i * self.batch_size
                f2vis_output[start_idx : start_idx + len(f2vis_batch_output), :] = f2vis_batch_output
            print 'pred_output: ', pred_output
            print 'self.testing_configs: ', self.testing_configs
            #pred_output[pred_output > 0.5] = 1.
            #pred_output[pred_output <= 0.5] = 0.
            #print 'binary pred_output: ', pred_output
            #print 'pred_output.shape: ', pred_output.shape
            print 'self.testing_configs.shape: ', self.testing_configs.shape
            pred_errors_all = np.abs(pred_output - self.testing_configs) 
            avg_pred_error_test = np.mean(pred_errors_all)
            print('avg_pred_error_test:', avg_pred_error_test)

            #print f2vis_output.shape, self.testing_labels.shape
            #tsne_vis.tsne_vis(f2vis_output, self.testing_labels[:, 0], '../models/tsne/test_tsne_gt.png', True)
            #tsne_vis.tsne_vis(f2vis_output, pred_output[:, 0], '../models/tsne/test_tsne_pred.png', True)

            #fig, ax = plt.subplots(figsize=(8, 8))
            #ax.plot(np.linspace(1,self.testing_samples_num,self.testing_samples_num), self.testing_labels, 'ro', label='gt labels')
            #ax.plot(np.linspace(1,self.testing_samples_num,self.testing_samples_num), pred_output, 'bo', label='pred labels')
            #ax.plot(np.linspace(1,self.testing_samples_num,self.testing_samples_num), pred_errors_all, 'k*', label='pred errors')
            ##plt.legend(handles=[gt_plot, pred_plot, error_plot])
            #legend = ax.legend(loc='upper right', shadow=True)
            #frame = legend.get_frame()
            #frame.set_facecolor('0.90')
            #fig.savefig('../models/labels_test.png')
            #plt.clf()
            #plt.close(fig)
        
        elapsed_time = time.time() - start_time
        print 'Total testing elapsed_time: ', elapsed_time

def main():
    grasp_rgbd_shallow_reg_net = GraspRgbdPatchesShallowRegNet()
    grasp_rgbd_shallow_reg_net.train()
    #grasp_rgbd_shallow_reg_net.test()

if __name__ == '__main__':
    main()

