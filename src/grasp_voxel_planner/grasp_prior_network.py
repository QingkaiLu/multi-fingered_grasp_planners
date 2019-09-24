import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import roslib.packages as rp
pkg_path = rp.get_pkg_dir('prob_grasp_planner')
import sys
sys.path.append(pkg_path + '/src/voxel_ae')
from voxel_ae import VoxelAE 


class GraspPriorNetwork():
    '''
        Build the grasp conditional prior mixture density network.
    '''

    def __init__(self, update_voxel_enc=True, 
                    dropout=False, voxel_ae=None):

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
        # Prior only for successful grasps?
        # self.holder_labels = tf.placeholder(tf.float32, [None, self.classes_num], 
        #                                     name = 'holder_labels')
        self.dropout = dropout
        if self.dropout:
            self.keep_prob = tf.placeholder(tf.float32, name='holder_keep_prob')

        self.prior_net_res = {}
        self.update_voxel_enc = update_voxel_enc

        self.num_components = 2

    
    def build_prior_network(self, voxel_ae=None):
        if voxel_ae is None:
            self.voxel_ae = VoxelAE()
            self.voxel_ae.build_voxel_ae_enc()
            # Get the voxel ae variables to restore the voxel ae before
            # creating new variables for grasp network.
            self.voxel_ae_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        else:
            self.voxel_ae = voxel_ae

        with tf.variable_scope('prior_net_struct'):
            voxel_obj_size_concat = tf.concat(axis=1, values=
                                        [self.voxel_ae.ae_struct_res['embedding'], 
                                        self.holder_obj_size])
            prior_fc_1 = tf.layers.dense(voxel_obj_size_concat, 128) 
            prior_fc_1 = tf.layers.batch_normalization(prior_fc_1, training=self.is_train)
            prior_fc_1 = tf.nn.relu(prior_fc_1)
            prior_fc_2 = tf.layers.dense(prior_fc_1, 32) 
            prior_fc_2 = tf.layers.batch_normalization(prior_fc_2, training=self.is_train)
            prior_fc_2 = tf.nn.relu(prior_fc_2)
            locs = tf.layers.dense(prior_fc_2, self.num_components * self.config_dim, 
                                    activation=None)
            scales = tf.layers.dense(prior_fc_2, self.num_components * self.config_dim, 
                                    activation=tf.exp)
            logits = tf.layers.dense(prior_fc_2, self.num_components, activation=None)
            # code from Mat's MDN
            locs = tf.reshape(locs, [-1, self.num_components, self.config_dim])
            scales = tf.reshape(scales, [-1, self.num_components, self.config_dim])
            logits = tf.reshape(logits, [-1, self.num_components])
            self.prior_net_res['locs'] = locs
            self.prior_net_res['scales'] = scales
            self.prior_net_res['logits'] = logits
            # reshape so that the first dim is the mixture, because we are doing to unstack them
            # also swap the batch size and the ones that come from the steps of this run
            # (K x N x D)
            mix_first_locs = tf.transpose(locs, [1, 0, 2])
            mix_first_scales = tf.transpose(scales, [1, 0, 2])
            cat = tfd.Categorical(logits=logits)
            components = []
            for loc, scale in zip(tf.unstack(mix_first_locs), tf.unstack(mix_first_scales)):
                normal = tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)
                components.append(normal)
            mixture = tfd.Mixture(cat=cat, components=components)
            self.prior_net_res['mixture'] = mixture


    def prior_network_loss(self, mixture, train_mode=True): 
        with tf.variable_scope('prior_net_loss'):
                loss = tf.reduce_mean(-mixture.log_prob(self.holder_config))

        if train_mode:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                # optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, 
                #                                        momentum=self.momentum,
                #                                        use_nesterov=True)
                if self.update_voxel_enc:
                    opt_loss = optimizer.minimize(loss)
                else:
                    opt_loss = optimizer.minimize(loss, var_list=
                                 tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'prior_net'))

        loss_sum = tf.summary.scalar('loss', loss)
        learn_rate_sum = tf.summary.scalar('learning_rate', self.learning_rate)
        train_summary = tf.summary.merge([loss_sum, learn_rate_sum])
        self.prior_net_res['loss'] = loss
        if train_mode:
            self.prior_net_res['train_summary'] = train_summary
            self.prior_net_res['opt_loss'] = opt_loss


    def prior_net_train_test(self, train_mode):
        self.build_prior_network()
        self.prior_network_loss(self.prior_net_res['mixture'], train_mode=train_mode)

