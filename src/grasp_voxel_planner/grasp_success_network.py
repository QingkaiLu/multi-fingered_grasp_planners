import numpy as np
import tensorflow as tf
import roslib.packages as rp
pkg_path = rp.get_pkg_dir('prob_grasp_planner')
import sys
sys.path.append(pkg_path + '/src/voxel_ae')
from voxel_ae import VoxelAE 


class GraspSuccessNetwork():


    def __init__(self, update_voxel_enc=True, dropout=False):

        self.momentum = 0.9
        self.config_dim = 14
        self.classes_num = 1
        self.obj_size_dim = 3

        self.is_train = tf.placeholder(tf.bool, name='holder_is_train')
        self.learning_rate = tf.placeholder(tf.float32, name='holder_learn_rate')
        self.holder_config = tf.placeholder(tf.float32, [None, self.config_dim],
                                            name='holder_config')
        self.holder_obj_size = tf.placeholder(tf.float32, [None, self.obj_size_dim],
                                            name='holder_obj_size')
        self.holder_labels = tf.placeholder(tf.float32, [None, self.classes_num], 
                                            name = 'holder_labels')
        self.dropout = dropout
        if self.dropout:
            self.keep_prob = tf.placeholder(tf.float32, name='holder_keep_prob')

        self.voxel_ae = VoxelAE()
        self.grasp_net_res = {}
        self.update_voxel_enc = update_voxel_enc


    def fully_connected(self, x, W):
        # Fully connected wrapper, with relu activation
        x = tf.matmul(x, W)
        #x = tf.add(tf.matmul(x, W), b)
        # x = tf.layers.batch_normalization(x, training=self.is_train)
        x = tf.contrib.layers.layer_norm(x)
        #return tf.nn.elu(x)
        x = tf.nn.relu(x)
        if self.dropout:
            x = tf.nn.dropout(x, self.keep_prob)
        return x


    def create_grasp_net_var(self):
        with tf.variable_scope('grasp_net_var'):
            self.weights_grasp_net = {
                    # Xavier initializion is also called Glorot 
                    # initialization(tf.glorot_uniform_initializer)
                    'voxel_fc_1': tf.get_variable(name='w_voxel_fc_1', 
                        shape=[self.voxel_ae.latents_num + self.obj_size_dim, 128], 
                        initializer=tf.contrib.layers.xavier_initializer()),
                    'voxel_fc_2': tf.get_variable(name='w_voxel_fc_2', 
                        shape=[128, 64], 
                        initializer=tf.contrib.layers.xavier_initializer()),
                    'config_fc_1': tf.get_variable(name='w_config_fc_1', 
                        shape=[self.config_dim, 14], 
                        initializer=tf.contrib.layers.xavier_initializer()),
                    'config_fc_2': tf.get_variable(name='w_config_fc_2', 
                        shape=[14, 8], 
                        initializer=tf.contrib.layers.xavier_initializer()),
                    'grasp_fc_1': tf.get_variable(name='w_grasp_fc_1', 
                        shape=[64 + 8, 64], 
                        initializer=tf.contrib.layers.xavier_initializer()),
                    'grasp_fc_2': tf.get_variable(name='w_grasp_fc_2', 
                        shape=[64, 32], 
                        initializer=tf.contrib.layers.xavier_initializer()),
                    'grasp_output': tf.get_variable(name='w_grasp_output', 
                        shape=[32, self.classes_num], 
                        initializer=tf.contrib.layers.xavier_initializer()),
                    }
 

    def build_grasp_network(self):
        #self.voxel_ae.build_voxel_ae()
        self.voxel_ae.build_voxel_ae_enc()

        # Get the voxel ae variables to restore the voxel ae before
        # creating new variables for grasp network.
        #self.voxel_ae_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        
        self.create_grasp_net_var()
        with tf.variable_scope('grasp_net_struct'):
            voxel_obj_size_concat = tf.concat(axis=1, values=
                                        [self.voxel_ae.ae_struct_res['embedding'], 
                                        self.holder_obj_size])
            voxel_fc_1 = self.fully_connected(voxel_obj_size_concat, 
                                                self.weights_grasp_net['voxel_fc_1'])
            voxel_fc_2 = self.fully_connected(voxel_fc_1, 
                                                self.weights_grasp_net['voxel_fc_2'])
            config_fc_1 = self.fully_connected(self.holder_config, 
                                                self.weights_grasp_net['config_fc_1'])
            config_fc_2 = self.fully_connected(config_fc_1, 
                                                self.weights_grasp_net['config_fc_2'])
            voxel_config_concat = tf.concat(axis=1, values=[voxel_fc_2, config_fc_2])
            # self.grasp_net_res['voxel_config_concat'] = voxel_config_concat
            grasp_fc_1 = self.fully_connected(voxel_config_concat, 
                                                self.weights_grasp_net['grasp_fc_1'])
            grasp_fc_2 = self.fully_connected(grasp_fc_1, 
                                                self.weights_grasp_net['grasp_fc_2'])

            logits = tf.matmul(grasp_fc_2, self.weights_grasp_net['grasp_output'])
            suc_prob = tf.nn.sigmoid(logits)
            self.grasp_net_res['logits'] = logits
            self.grasp_net_res['suc_prob'] = suc_prob
            #return logits, suc_prob    


    def weighted_binary_crossentropy(self, pred, target, alpha=0.75):
        return -alpha * target * tf.log(pred) - \
                (1 - alpha) * (1.0 - target) * tf.log(1.0 - pred)


    def focal_loss(self, pred, target, alpha=0.25, gamma=2.):
    #def focal_loss(self, pred, target, alpha=0.75, gamma=2.):
        return -alpha * target * ((1 - pred) ** gamma) *  tf.log(pred) - \
                (1 - alpha) * (1.0 - target) * (pred ** gamma) * tf.log(1.0 - pred)


    def grasp_network_loss(self, logits=None, pred_probs=None, train_mode=True): 
        with tf.variable_scope('grasp_net_loss'):
            if pred_probs == None:
                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, 
                                        labels=self.holder_labels))
            else:
                loss = tf.reduce_mean(self.focal_loss(tf.clip_by_value(pred_probs, 
                                                        1e-8, 1.0), self.holder_labels))
                                                        # 1e-7, 1.0 - 1e-7), self.holder_labels))
                # loss = tf.reduce_mean(self.weighted_binary_crossentropy(tf.clip_by_value(pred_probs, 
                #                         1e-7, 1.0 - 1e-7), self.holder_labels))

            if train_mode:
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                # UPDATE_OPS is for batch norm
                # https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm
                with tf.control_dependencies(update_ops):
                    optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                    # optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, 
                    #                                        momentum=self.momentum,
                    #                                        use_nesterov=True)
                    if self.update_voxel_enc:
                        opt_loss = optimizer.minimize(loss)
                    else:
                        opt_loss = optimizer.minimize(loss, var_list=
                                     tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'grasp_net'))

        loss_sum = tf.summary.scalar('loss', loss)
        learn_rate_sum = tf.summary.scalar('learning_rate', self.learning_rate)
        train_summary = tf.summary.merge([loss_sum, learn_rate_sum])
        self.grasp_net_res['loss'] = loss
        if train_mode:
            self.grasp_net_res['train_summary'] = train_summary
            self.grasp_net_res['opt_loss'] = opt_loss


    def grasp_net_train_test(self, train_mode):
        self.build_grasp_network()
        self.grasp_network_loss(logits=self.grasp_net_res['logits'], train_mode=train_mode)
        # self.grasp_network_loss(pred_probs=self.grasp_net_res['suc_prob'], train_mode=train_mode)



