import rospy
import tf as ros_tf
import roslib.packages as rp
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import numpy as np
pkg_path = rp.get_pkg_dir('prob_grasp_planner')
import sys
sys.path.append(pkg_path + '/src/grasp_common_library')
sys.path.append(pkg_path + '/src/grasp_voxel_planner')
from grasp_success_network import GraspSuccessNetwork
from grasp_prior_network import GraspPriorNetwork
import grasp_common_functions as gcf
from set_config_limits import SetConfigLimits
#from scipy.optimize import fmin_bfgs
#from scipy.optimize import fmin_l_bfgs_b
from scipy.optimize import minimize
import pickle
import h5py
import time
from active_data_loader import ActiveDataLoader
from sklearn.metrics import precision_recall_fscore_support
from sklearn import mixture
import os
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from grasp_kdl import GraspKDL 
import random


class GraspActiveLearnerFK:
    '''
    Active grasp learner.
    '''

    def __init__(self, prior_name, grasp_net_model_path=None, 
                    prior_model_path=None, suc_prior_path=None,
                    al_file_path=None, active_models_path=None, 
                    spv_data_path=None, suc_spv_data_path=None, 
                    active_data_path=None, suc_active_data_path=None,
                    cvtf=None, vis_preshape=False, virtual_hand_parent_tf=''):
        # rospy.init_node('grasp_active_learner')
        self.prior_name = prior_name
        self.grasp_net_model_path = grasp_net_model_path
        self.prior_model_path = prior_model_path
        self.suc_prior_path = suc_prior_path
        self.active_models_path = active_models_path
        self.cvtf = cvtf

        self.config_limits = SetConfigLimits()

        #0.5 is used for active learning.
        self.reg_log_prior = 0.5 
        #Regularization for log likelihood, this is mainly used to 
        #test the inference with only prior.
        self.reg_log_lkh = 1.
        #Use success prior or general prior for the max success arm
        self.use_suc_prior = False

        self.init_active_learning(al_file_path, spv_data_path, 
                                suc_spv_data_path, active_data_path, 
                                suc_active_data_path)

        self.vis_preshape = vis_preshape
        if self.vis_preshape:
            self.tf_br = ros_tf.TransformBroadcaster()
            self.js_pub = rospy.Publisher('/virtual_hand/allegro_hand_right/joint_states', 
                                          JointState, queue_size=1)
            self.preshape_config = None
        self.virtual_hand_parent_tf = virtual_hand_parent_tf


    def init_active_learning(self, al_file_path, spv_data_path,
                            suc_spv_data_path, active_data_path,
                            suc_active_data_path):
        self.adl = None
        tf_config = tf.ConfigProto(log_device_placement=False)
        tf_config.gpu_options.allow_growth = True
        self.tf_sess = tf.Session(config=tf_config)
        
        if self.prior_name == 'MDN':
            self.mdn_adl = None
            self.suc_mdn_adl = None

        if self.active_models_path is not None:
            self.al_file_path = al_file_path
            if self.al_file_path is not None:
                self.init_al_file()
    
            if spv_data_path is not None and \
                active_data_path is not None:
                self.adl = ActiveDataLoader(spv_data_path, active_data_path)
                if self.prior_name == 'MDN':
                    self.mdn_adl = ActiveDataLoader(spv_data_path, 
                                                    active_data_path)
                    self.suc_mdn_adl = ActiveDataLoader(suc_spv_data_path, 
                                                        suc_active_data_path)
                self.tf_logs_path = '/home/qingkai/tf_logs'
                self.gmm_num_components = 2

            self.batch_size = 16 #8
            self.active_update_batches = 4 #8
            self.load_al_models()
        else:
            # Load model for active planning
            self.load_spv_models()

        
    def create_grasp_net(self, dropout=False):
        update_voxel_enc = False
        self.grasp_net = GraspSuccessNetwork(update_voxel_enc, dropout)
        # Set train_mode on to define the training loss
        is_train = True #False
        self.grasp_net.grasp_net_train_test(train_mode=is_train)
        self.config_grad = tf.gradients(self.grasp_net.grasp_net_res['suc_prob'], 
                                        self.grasp_net.holder_config)
        self.saver = tf.train.Saver(max_to_keep=None)


    def create_mdn_prior(self):
        update_voxel_enc = False
        self.prior_net = GraspPriorNetwork(update_voxel_enc, dropout=False)
        is_train = True
        self.prior_net.prior_net_train_test(train_mode=is_train, 
                                            voxel_ae=self.grasp_net.voxel_ae)
        # self.prior_net.build_prior_network(voxel_ae=self.grasp_net.voxel_ae)
        self.prior_saver = tf.train.Saver(var_list=tf.get_collection(
                                    tf.GraphKeys.GLOBAL_VARIABLES, 'prior_net'), 
                                    max_to_keep=None)


    def create_suc_mdn_prior(self):
        update_voxel_enc = False
        self.suc_prior_scope = 'suc_prior_net'
        self.suc_prior_net = GraspPriorNetwork(update_voxel_enc, 
                                            dropout=True,
                                            scope_name=self.suc_prior_scope)
        is_train = True
        self.suc_prior_net.prior_net_train_test(train_mode=is_train, 
                                            voxel_ae=self.grasp_net.voxel_ae)
        self.suc_prior_saver = tf.train.Saver(var_list=tf.get_collection(
                                    tf.GraphKeys.GLOBAL_VARIABLES, 
                                    self.suc_prior_scope), 
                                    max_to_keep=None)


    def restore_deep_models(self, grasp_net_model_path, 
                            prior_net_model_path=None,
                            suc_prior_net_path=None):
        init = tf.global_variables_initializer()
        self.tf_sess.run(init)

        print 'Load grasp network from: ', grasp_net_model_path
        self.saver.restore(self.tf_sess, grasp_net_model_path)
        
        if self.prior_name == 'MDN':
            print 'Loading MDN prior from: ', prior_net_model_path
            self.prior_saver.restore(self.tf_sess, prior_net_model_path)
            if suc_prior_net_path is not None:
                print 'Loading suc MDN prior from: ', suc_prior_net_path
                self.suc_prior_saver.restore(self.tf_sess, 
                                        suc_prior_net_path)

        tf.get_default_graph().finalize()


    def build_mdn_inf_model(self):
        k = self.prior_net.num_components 
        n = self.prior_net.config_dim
        self.config_holder = tf.placeholder(tf.float32, [None, n],
                                            name='config_holder')
        self.locs_holder = tf.placeholder(tf.float32, [None, k, n], 
                                            name='locs_holder')
        self.scales_holder = tf.placeholder(tf.float32, [None, k, n], 
                                            name='scales_holder')
        self.logits_holder = tf.placeholder(tf.float32, [None, k], 
                                            name='logits_holder')

        mix_first_locs = tf.transpose(self.locs_holder, [1, 0, 2])
        mix_first_scales = tf.transpose(self.scales_holder, [1, 0, 2])
        cat = tfd.Categorical(logits=self.logits_holder)
        mix_components = []
        for loc, scale in zip(tf.unstack(mix_first_locs), tf.unstack(mix_first_scales)):
        # for loc, scale in zip(mix_first_locs, mix_first_scales):
            normal = tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)
            mix_components.append(normal)
        mixture = tfd.Mixture(cat=cat, components=mix_components)

        self.prior_prob = mixture.prob(self.config_holder)
        self.prior_prob_grad = tf.gradients(self.prior_prob, self.config_holder)
        self.prior_log_prob = mixture.log_prob(self.config_holder)
        self.prior_log_prob_grad = tf.gradients(self.prior_log_prob, self.config_holder)
        self.prior_sample = mixture.sample()
        self.seperate_comp_sample = [mix_components[0].sample(), 
                                     mix_components[1].sample()]

        
    def build_suc_mdn_inf_model(self):
        k = self.suc_prior_net.num_components 
        n = self.suc_prior_net.config_dim
        self.suc_config_holder = tf.placeholder(tf.float32, [None, n],
                                            name='suc_config_holder')
        self.suc_locs_holder = tf.placeholder(tf.float32, [None, k, n], 
                                            name='suc_locs_holder')
        self.suc_scales_holder = tf.placeholder(tf.float32, [None, k, n], 
                                            name='suc_scales_holder')
        self.suc_logits_holder = tf.placeholder(tf.float32, [None, k], 
                                            name='suc_logits_holder')

        mix_first_locs = tf.transpose(self.suc_locs_holder, [1, 0, 2])
        mix_first_scales = tf.transpose(self.suc_scales_holder, [1, 0, 2])
        cat = tfd.Categorical(logits=self.suc_logits_holder)
        mix_components = []
        for loc, scale in zip(tf.unstack(mix_first_locs), tf.unstack(mix_first_scales)):
        # for loc, scale in zip(mix_first_locs, mix_first_scales):
            normal = tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)
            mix_components.append(normal)
        mixture = tfd.Mixture(cat=cat, components=mix_components)

        self.suc_prior_prob = mixture.prob(self.suc_config_holder)
        self.suc_prior_prob_grad = tf.gradients(self.suc_prior_prob, 
                                            self.suc_config_holder)
        self.suc_prior_log_prob = mixture.log_prob(self.suc_config_holder)
        self.suc_prior_log_prob_grad = tf.gradients(self.suc_prior_log_prob, 
                                                        self.suc_config_holder)
        self.suc_prior_sample = mixture.sample()
        self.suc_sep_comp_sample = [mix_components[0].sample(), 
                                     mix_components[1].sample()]


    def get_prior_mixture(self, grasp_voxel_grid, grasp_obj_size):
        feed_dict = {
            self.prior_net.voxel_ae.is_train: False,
            self.prior_net.voxel_ae.partial_voxel_grids: [grasp_voxel_grid],
            self.prior_net.is_train: False,
            self.prior_net.holder_obj_size: [grasp_obj_size],
        }
        [self.locs, self.scales, self.logits] = self.tf_sess.run(
                        [self.prior_net.prior_net_res['locs'],
                        self.prior_net.prior_net_res['scales'],
                        self.prior_net.prior_net_res['logits']], 
                        feed_dict=feed_dict)


    def get_suc_prior_mixture(self, grasp_voxel_grid, grasp_obj_size):
        feed_dict = {
            self.suc_prior_net.voxel_ae.is_train: False,
            self.suc_prior_net.voxel_ae.partial_voxel_grids: [grasp_voxel_grid],
            self.suc_prior_net.is_train: False,
            self.suc_prior_net.holder_obj_size: [grasp_obj_size],
        }
        [self.suc_locs, self.suc_scales, self.suc_logits] = self.tf_sess.run(
                        [self.suc_prior_net.prior_net_res['locs'],
                        self.suc_prior_net.prior_net_res['scales'],
                        self.suc_prior_net.prior_net_res['logits']], 
                        feed_dict=feed_dict)


    def init_batch_id_file(self):
        self.batch_id_file_path = self.active_models_path + 'last_batch_id.h5'
        batch_id_file = h5py.File(self.batch_id_file_path, 'a')
        if 'last_batch_id' not in batch_id_file:
            batch_id_file.create_dataset('last_batch_id', data=-1)
        batch_id_file.close()


    def get_last_batch_id(self):
        batch_id_file = h5py.File(self.batch_id_file_path, 'r')
        last_batch_id = batch_id_file['last_batch_id'][()]
        batch_id_file.close()
        return last_batch_id


    def update_last_batch_id(self, last_batch_id):
        batch_id_file = h5py.File(self.batch_id_file_path, 'r+')
        batch_id_file['last_batch_id'][()] = last_batch_id
        batch_id_file.close()


    def load_spv_models(self):
        self.create_grasp_net()
        if self.prior_name == 'MDN':
            self.create_mdn_prior()
            self.build_mdn_inf_model()
            self.create_suc_mdn_prior()
            self.build_suc_mdn_inf_model()
        elif self.prior_name == 'GMM':
            self.load_gmm_prior(self.prior_model_path)
            self.load_suc_gmm_prior(self.suc_prior_path)

        self.restore_deep_models(self.grasp_net_model_path,
                                self.prior_model_path,
                                self.suc_prior_path)


    def load_gmm_prior(self, prior_path):
        print 'Loading GMM prior from: ', prior_path
        self.gmm_prior_model = pickle.load(open(prior_path, 'rb'))


    def load_suc_gmm_prior(self, suc_prior_path):
        print 'Loading suc GMM prior from: ', suc_prior_path
        self.suc_gmm_prior_model = pickle.load(open(
                                            suc_prior_path, 'rb'))


    def load_al_models(self):
        '''
        Load models for active learning.
        '''
        self.init_batch_id_file()
        last_batch_id = self.get_last_batch_id()
        print 'last_batch_id:', last_batch_id
        if last_batch_id >= 0:
            batch_models_path = self.active_models_path + 'batch_' \
                                + str(last_batch_id) + '_models'
            batches_num = last_batch_id + 1
            grasp_model_path = None
            mdn_model_path = None
            if batches_num % self.active_update_batches == 0:
                grasp_model_path = batch_models_path + '/batch_' \
                           + str(last_batch_id) + '_retrain_net' 
            else:
                grasp_model_path = batch_models_path + '/batch_' \
                           + str(last_batch_id) + '_incrmt_net' 
            self.create_grasp_net()

            prior_batch_id = last_batch_id - batches_num % \
                                    self.active_update_batches
            prior_batch_models_path = self.active_models_path + 'batch_' \
                                + str(prior_batch_id) + '_models'
 
            if self.prior_name == 'MDN':
                if batches_num % self.active_update_batches == 0:
                    mdn_model_path = batch_models_path + '/batch_' \
                               + str(last_batch_id) + '_retrain_mdn' 
                else:
                    mdn_model_path = batch_models_path + '/batch_' \
                               + str(last_batch_id) + '_incrmt_mdn' 
                suc_mdn_path = prior_batch_models_path + '/batch_' \
                               + str(prior_batch_id) + '_retrain_suc_mdn' 
                self.create_mdn_prior()
                self.build_mdn_inf_model()
                self.create_suc_mdn_prior()
                self.build_suc_mdn_inf_model()
            elif self.prior_name == 'GMM':
                # last_prior_path = batch_models_path + '/batch_' \
                #                     + str(last_batch_id) + '_gmm_prior' 
                # self.load_gmm_prior(last_prior_path)
                last_prior_path = batch_models_path + '/batch_' \
                                    + str(last_batch_id) + '_gmm_prior' 
                self.load_gmm_prior(last_prior_path)
                last_suc_prior_path = prior_batch_models_path + '/batch_' \
                                    + str(prior_batch_id) + '_suc_gmm_prior' 
                self.load_suc_gmm_prior(last_suc_prior_path)

            else:
                rospy.logerr('Wrong prior model name!')
            
            self.restore_deep_models(grasp_model_path,
                        mdn_model_path, suc_mdn_path)

            if self.adl is not None:
                self.adl.count_samples()
                expect_last_batch_id = self.adl.num_act_samples / \
                                        self.batch_size - 1
                print 'num_act_samples:', self.adl.num_act_samples
                print 'expect_last_batch_id:', expect_last_batch_id
                print 'last_batch_id:', last_batch_id
                if expect_last_batch_id > last_batch_id:
                    print 'Update the model for expected last batch id!'
                    self.active_model_update(expect_last_batch_id)
                    last_batch_id = expect_last_batch_id
        else:
            self.load_spv_models()
    

    def init_al_file(self):
        al_file = h5py.File(self.al_file_path, 'a')
        if 'total_action_num' not in al_file:
            self.total_action_num = 0
            al_file.create_dataset('total_action_num', data=self.total_action_num)
            self.al_strategies = ['grasp_suc', 'grasp_uct', 'config_explore']
            al_file.create_dataset('al_strategies', data=self.al_strategies)
            self.actions = []
            al_file.create_dataset('actions', data=self.actions)
            self.rewards = [[], [] , []]
            for i in xrange(len(self.al_strategies)):
                al_file.create_dataset('rewards_strategy_' + str(i), 
                                        data=self.rewards[i])
            self.inf_time = [[], [] , []]
            for i in xrange(len(self.al_strategies)):
                al_file.create_dataset('inf_time_strategy_' + str(i), 
                                        data=self.inf_time[i])
            self.avg_rewards = [0., 0., 0.]
            al_file.create_dataset('avg_rewards', data=self.avg_rewards)
            actions = al_file['actions'][()]
        else:
            self.total_action_num = al_file['total_action_num'][()]
            self.al_strategies = list(al_file['al_strategies'][()])
            self.actions = list(al_file['actions'][()])
            self.rewards = []
            for i in xrange(len(self.al_strategies)):
                self.rewards.append(list(al_file['rewards_strategy_' 
                                    + str(i)][()]))
            self.inf_time = []
            for i in xrange(len(self.al_strategies)):
                self.inf_time.append(list(al_file['inf_time_strategy_' 
                                    + str(i)][()]))
            self.avg_rewards = list(al_file['avg_rewards'][()])

        self.strategies_num = len(self.al_strategies)

        al_file.close()


    def pub_preshape_config(self):
        if self.preshape_config is not None:
            for i in xrange(2): 
                preshape_pose = self.preshape_config.palm_pose
                self.tf_br.sendTransform((preshape_pose.pose.position.x, 
                                          preshape_pose.pose.position.y,
                                          preshape_pose.pose.position.z),
                                          (preshape_pose.pose.orientation.x, 
                                          preshape_pose.pose.orientation.y, 
                                          preshape_pose.pose.orientation.z, 
                                          preshape_pose.pose.orientation.w), 
                                        rospy.Time.now(), '/virtual_hand/allegro_mount', 
                                        '/virtual_hand/' + self.virtual_hand_parent_tf)
                self.js_pub.publish(self.preshape_config.hand_joint_state)
                rospy.sleep(0.5)


    def pred_clf_suc_prob(self, grasp_config, grasp_voxel_grid, 
                        grasp_obj_size):
        feed_dict = {
            self.grasp_net.voxel_ae.is_train: False,
            self.grasp_net.voxel_ae.partial_voxel_grids: [grasp_voxel_grid],
            self.grasp_net.is_train: False,
            self.grasp_net.holder_config: [grasp_config],
            self.grasp_net.holder_obj_size: [grasp_obj_size],
     }
        [suc_prob] = self.tf_sess.run(
                [self.grasp_net.grasp_net_res['suc_prob']],
                feed_dict=feed_dict)
        return suc_prob[0][0]


    def compute_clf_config_grad(self, grasp_config, grasp_voxel_grid, 
                            grasp_obj_size):
        feed_dict = {
            self.grasp_net.voxel_ae.is_train: False,
            self.grasp_net.voxel_ae.partial_voxel_grids: [grasp_voxel_grid],
            self.grasp_net.is_train: False,
            self.grasp_net.holder_config: [grasp_config],
            self.grasp_net.holder_obj_size: [grasp_obj_size],
     }
        [suc_prob, config_gradient] = self.tf_sess.run(
                [self.grasp_net.grasp_net_res['suc_prob'], self.config_grad], 
                feed_dict=feed_dict)
        return config_gradient[0][0], suc_prob[0][0]


    def grasp_clf_log_suc_prob(self, grasp_config, grasp_voxel_grid,
                            grasp_obj_size):
        suc_prob = self.pred_clf_suc_prob(grasp_config, 
                                    grasp_voxel_grid, grasp_obj_size)
        log_suc_prob = np.log(suc_prob)
        # log_suc_prob *= self.reg_log_lkh
        # neg_log_suc_prob = -np.float64(log_suc_prob)
        return log_suc_prob


    def grasp_clf_log_suc_grad(self, grasp_config, grasp_voxel_grid,
                            grasp_obj_size):
        config_grad, suc_prob = self.compute_clf_config_grad(grasp_config, 
                                        grasp_voxel_grid, grasp_obj_size)
        # print 'config_grad:', config_grad
        if np.isclose(suc_prob, 0.):
            suc_prob = 10**-10
        log_config_grad = config_grad / suc_prob
        return log_config_grad


    def compute_grasp_log_prior(self, grasp_config):
        log_prior = None
        if self.prior_name == 'MDN':
            log_prior = self.compute_grasp_log_prior_mdn(grasp_config)
        elif self.prior_name == 'GMM':
            log_prior = self.compute_grasp_log_prior_gmm(grasp_config)
        return log_prior


    def compute_suc_log_prior(self, grasp_config):
        log_prior = None
        if self.prior_name == 'MDN':
            log_prior = self.compute_suc_log_prior_mdn(grasp_config)
        elif self.prior_name == 'GMM':
            log_prior = self.compute_suc_log_prior_gmm(grasp_config)
        return log_prior


    def neg_grasp_log_prior(self, grasp_config):
        return -self.compute_grasp_log_prior(grasp_config)


    def compute_grasp_log_prior_mdn(self, grasp_config):
        '''
        Compute the grasp configuration MDN prior.
        '''
        feed_dict = {self.locs_holder: self.locs, 
                    self.scales_holder: self.scales,
                    self.logits_holder: self.logits,
                    self.config_holder: [grasp_config]}
        [log_prior] = self.tf_sess.run([self.prior_log_prob], 
                                        feed_dict=feed_dict)  
        return log_prior[0]


    def compute_grasp_prior_mdn(self, grasp_config):
        '''
        Compute the grasp configuration MDN prior.
        '''
        feed_dict = {self.locs_holder: self.locs, 
                    self.scales_holder: self.scales,
                    self.logits_holder: self.logits,
                    self.config_holder: [grasp_config]}
        [prior] = self.tf_sess.run([self.prior_prob], 
                                        feed_dict=feed_dict)  
        return prior[0]


    def compute_suc_log_prior_mdn(self, grasp_config):
        '''
        Compute the grasp configuration MDN prior of successful grasps.
        '''
        feed_dict = {self.suc_locs_holder: self.suc_locs, 
                    self.suc_scales_holder: self.suc_scales,
                    self.suc_logits_holder: self.suc_logits,
                    self.suc_config_holder: [grasp_config]}
        [suc_log_prior] = self.tf_sess.run([self.suc_prior_log_prob], 
                                        feed_dict=feed_dict)  
        return suc_log_prior[0]


    def compute_suc_prior_mdn(self, grasp_config):
        '''
        Compute the grasp configuration MDN prior of successful grasps.
        '''
        feed_dict = {self.suc_locs_holder: self.suc_locs, 
                    self.suc_scales_holder: self.suc_scales,
                    self.suc_logits_holder: self.suc_logits,
                    self.suc_config_holder: [grasp_config]}
        [suc_prior] = self.tf_sess.run([self.suc_prior_prob], 
                                        feed_dict=feed_dict)  
        return suc_prior[0]


    def compute_grasp_log_prior_gmm(self, grasp_config):
        '''
        Compute the grasp configuration GMM prior.
        '''
        log_prior = self.gmm_prior_model.score_samples([grasp_config])[0]
        return log_prior


    def compute_grasp_prior_gmm(self, grasp_config):
        '''
        Compute the grasp configuration GMM prior.
        '''
        log_prior = self.gmm_prior_model.score_samples([grasp_config])[0]
        prior = np.exp(log_prior)
        return prior


    def compute_suc_log_prior_gmm(self, grasp_config):
        '''
        Compute the grasp configuration GMM prior of successful grasps.
        '''
        suc_log_prior = self.suc_gmm_prior_model.score_samples([grasp_config])[0]
        return suc_log_prior


    def compute_suc_prior_gmm(self, grasp_config):
        '''
        Compute the grasp configuration GMM prior of successful grasps.
        '''
        suc_log_prior = self.suc_gmm_prior_model.score_samples([grasp_config])[0]
        suc_prior = np.exp(suc_log_prior)
        return suc_prior


    def grasp_log_prior_grad(self, grasp_config):
        grad = None
        if self.prior_name == 'MDN':
            grad = self.grasp_log_prior_grad_mdn(grasp_config)
        elif self.prior_name == 'GMM':
            grad = self.grasp_log_prior_grad_gmm(grasp_config)
        return grad


    def suc_log_prior_grad(self, grasp_config):
        grad = None
        if self.prior_name == 'MDN':
            grad = self.suc_log_prior_grad_mdn(grasp_config)
        elif self.prior_name == 'GMM':
            grad = self.suc_log_prior_grad_gmm(grasp_config)
        return grad


    def grasp_log_prior_grad_mdn(self, grasp_config):
        feed_dict = {self.locs_holder: self.locs, 
                    self.scales_holder: self.scales,
                    self.logits_holder: self.logits,
                    self.config_holder: [grasp_config]}
        [log_prior_grad] = self.tf_sess.run([self.prior_log_prob_grad], 
                                        feed_dict=feed_dict)  
        return log_prior_grad[0][0]
       

    def grasp_prior_grad_mdn(self, grasp_config):
        feed_dict = {self.locs_holder: self.locs, 
                    self.scales_holder: self.scales,
                    self.logits_holder: self.logits,
                    self.config_holder: [grasp_config]}
        [prior_grad] = self.tf_sess.run([self.prior_prob_grad], 
                                        feed_dict=feed_dict)  
        return prior_grad[0][0]


    def suc_log_prior_grad_mdn(self, grasp_config):
        feed_dict = {self.suc_locs_holder: self.suc_locs, 
                    self.suc_scales_holder: self.suc_scales,
                    self.suc_logits_holder: self.suc_logits,
                    self.suc_config_holder: [grasp_config]}
        [suc_log_prior_grad] = self.tf_sess.run([self.suc_prior_log_prob_grad], 
                                        feed_dict=feed_dict)  
        return suc_log_prior_grad[0][0]
       

    def suc_prior_grad_mdn(self, grasp_config):
        feed_dict = {self.suc_locs_holder: self.suc_locs, 
                    self.suc_scales_holder: self.suc_scales,
                    self.suc_logits_holder: self.suc_logits,
                    self.suc_config_holder: [grasp_config]}
        [suc_prior_grad] = self.tf_sess.run([self.suc_prior_prob_grad], 
                                        feed_dict=feed_dict)  
        return suc_prior_grad[0][0]


    def grasp_log_prior_grad_gmm(self, grasp_config):
        '''
        Compute the grasp configuration prior gradient with respect
        to grasp configuration.

        To avoid numerical issues for inference with gmm prior. The inference objective function is in this format:
        (w1 * exp(s1) + w2 * exp(s2) ... + wk * exp(sk) ) / (exp(s1) + exp(s2) + ... + exp(sk) )

        To avoid numerical issues (sum of exponential is too small or too large), we find s' = max(s_1, s_2. s_k). 
        Then we divide both the numerator and denominator by exp(s') to rewrite the objective function as: 
            (w1 * exp(s1 - s') + w2 * exp(s2 - s') ... + wk * exp(sk - s') ) / (exp(s1 - s') + exp(s2 - s') + ... + exp(sk - s') )

        By doing this, 1. the denominator will always be no less than 1, which avoids the numerical issue of denominator being zero;
        2. it can also avoid the numerical issue that both the denominator and numerator are very large. 

        '''
        #_estimate_weighted_log_prob dimension: (n_samples, n_components)
        weighted_log_prob = self.gmm_prior_model._estimate_weighted_log_prob(np.array([grasp_config]))[0]
        max_wlp = np.max(weighted_log_prob)
        #Dim: n_components
        wlp_minus_max = weighted_log_prob - max_wlp

        #Dim: n_config
        p_x_prime = np.zeros(len(grasp_config))
        for i in xrange(self.gmm_prior_model.weights_.shape[0]):
            #log of inverse of covariance matrix multiply distance (x - mean)
            #Dim: n_config 
            inv_sigma_dist = np.matmul(np.linalg.inv(self.gmm_prior_model.covariances_[i]), \
                                (grasp_config - self.gmm_prior_model.means_[i]))
            p_x_prime += -np.exp(wlp_minus_max[i]) * inv_sigma_dist

        prior = np.sum(np.exp(wlp_minus_max))
        grad = p_x_prime / prior 

        return grad


    def grasp_prior_grad_gmm(self, grasp_config):
        '''
        Compute the grasp configuration prior gradient with respect
        to grasp configuration.
        '''
        weighted_log_prob = self.gmm_prior_model._estimate_weighted_log_prob(
                                                np.array([grasp_config]))[0]
        weighted_prob = np.exp(weighted_log_prob)
        # weighted_prob can also be computed by: 
        # multivariate_normal(mean=g.means_[i], cov=g.covariances_[i], allow_singular=True)
        grad = np.zeros(len(grasp_config))
        for i, w in enumerate(self.gmm_prior_model.weights_):
            grad += -weighted_prob[i] * np.matmul(np.linalg.inv(
                        self.gmm_prior_model.covariances_[i]), \
                        (grasp_config - self.gmm_prior_model.means_[i]))
        return grad


    def suc_log_prior_grad_gmm(self, grasp_config):
        '''
        Compute the grasp configuration success prior gradient with respect
        to grasp configuration.
        '''
        #_estimate_weighted_log_prob dimension: (n_samples, n_components)
        weighted_log_prob = self.suc_gmm_prior_model._estimate_weighted_log_prob(
                                                        np.array([grasp_config]))[0]
        max_wlp = np.max(weighted_log_prob)
        #Dim: n_components
        wlp_minus_max = weighted_log_prob - max_wlp

        #Dim: n_config
        p_x_prime = np.zeros(len(grasp_config))
        for i in xrange(self.suc_gmm_prior_model.weights_.shape[0]):
            #log of inverse of covariance matrix multiply distance (x - mean)
            #Dim: n_config 
            inv_sigma_dist = np.matmul(np.linalg.inv(self.suc_gmm_prior_model.covariances_[i]), \
                                (grasp_config - self.suc_gmm_prior_model.means_[i]))
            p_x_prime += -np.exp(wlp_minus_max[i]) * inv_sigma_dist

        prior = np.sum(np.exp(wlp_minus_max))
        grad = p_x_prime / prior 

        return grad


    def suc_prior_grad_gmm(self, grasp_config):
        '''
        Compute the grasp configuration success prior gradient with respect
        to grasp configuration.
        '''
        weighted_log_prob = self.suc_gmm_prior_model._estimate_weighted_log_prob(
                                                np.array([grasp_config]))[0]
        weighted_prob = np.exp(weighted_log_prob)
        # weighted_prob can also be computed by: 
        # multivariate_normal(mean=g.means_[i], cov=g.covariances_[i], allow_singular=True)
        grad = np.zeros(len(grasp_config))
        for i, w in enumerate(self.suc_gmm_prior_model.weights_):
            grad += -weighted_prob[i] * np.matmul(np.linalg.inv(
                        self.suc_gmm_prior_model.covariances_[i]), \
                        (grasp_config - self.suc_gmm_prior_model.means_[i]))
        return grad


    def compute_num_grad(self, func, grasp_config, 
                        grasp_voxel_grid, grasp_obj_size, 
                        mat_world_to_obj=None):
        eps = 10**-6
        # eps = 1.49e-08
        grad = np.zeros(len(grasp_config))
        for i in xrange(len(grasp_config)):
            grasp_config_plus = np.copy(grasp_config)
            grasp_config_plus[i] += eps
            if grasp_voxel_grid is not None:
                if mat_world_to_obj is not None:
                    obj_prob_plus = func(grasp_config_plus, 
                                grasp_voxel_grid, grasp_obj_size,
                                mat_world_to_obj)
                else:
                    obj_prob_plus = func(grasp_config_plus, 
                                grasp_voxel_grid, grasp_obj_size)
            else:
                obj_prob_plus = func(grasp_config_plus)
            grasp_config_minus = np.copy(grasp_config)
            grasp_config_minus[i] -= eps
            if grasp_voxel_grid is not None:
                if mat_world_to_obj is not None:
                    obj_prob_minus = func(grasp_config_minus, 
                                grasp_voxel_grid, grasp_obj_size,
                                mat_world_to_obj)
                else:
                    obj_prob_minus = func(grasp_config_minus, 
                                grasp_voxel_grid, grasp_obj_size)
            else:
                obj_prob_minus = func(grasp_config_minus)
            #print 'grasp_config_plus:', grasp_config_plus
            #print 'grasp_config_minus:', grasp_config_minus
            #print 'obj_prob_plus:', obj_prob_plus
            #print 'obj_prob_minus:', obj_prob_minus
            ith_grad = (obj_prob_plus - obj_prob_minus) / (2. * eps)
            grad[i] = ith_grad
        return grad


    def compute_num_ik_grad(self, func, ik_config, 
                            grasp_config, grasp_voxel_grid, 
                            grasp_obj_size, mat_world_to_obj):
        num_suc_grad = self.compute_num_grad(func, grasp_config, grasp_voxel_grid, 
                                                    grasp_obj_size) 
        num_jac = self.cvtf.compute_num_jac(self.cvtf.convert_to_fk_config, 
                                            ik_config, mat_world_to_obj)
        num_grad = np.zeros(num_suc_grad.shape[0] + 1)
        num_grad[:7] = np.matmul(num_suc_grad[:6], num_jac)
        num_grad[7:] = num_suc_grad[6:]
        return num_grad
 

    def grasp_log_posterior_fk(self, grasp_config, grasp_voxel_grid,
                            grasp_obj_size):
        log_suc_prob = self.grasp_clf_log_suc_prob(grasp_config, grasp_voxel_grid,
                            grasp_obj_size)
        if self.use_suc_prior:
            log_prior = self.compute_suc_log_prior(grasp_config)
        else:
            log_prior = self.compute_grasp_log_prior(grasp_config)
        reg_log_suc_prob = log_suc_prob * self.reg_log_lkh
        reg_log_prior = log_prior * self.reg_log_prior
        log_posterior = reg_log_suc_prob + reg_log_prior
        # print 'reg_log_suc_prob:', reg_log_suc_prob 
        # print 'reg_log_prior:', reg_log_prior
        # print 'log_posterior:', log_posterior
        return np.float64(log_posterior)


    def grasp_log_posterior(self, ik_config, grasp_voxel_grid,
                            grasp_obj_size, mat_world_to_obj):
        # grasp_config_tf = self.cvtf.convert_to_fk_config_tf(ik_config)
        grasp_config = self.cvtf.convert_to_fk_config(ik_config, mat_world_to_obj)
        return self.grasp_log_posterior_fk(grasp_config, grasp_voxel_grid,
                                            grasp_obj_size)


    def grasp_norm_posterior(self, grasp_config, grasp_voxel_grid,
                            grasp_obj_size):
        suc_prob = self.pred_clf_suc_prob(grasp_config, 
                                    grasp_voxel_grid, grasp_obj_size)
        log_prior = self.compute_grasp_log_prior(grasp_config)
        reg_suc_prob = 0.5 * suc_prob
        sigmoid_log_prior = self.sigmoid(log_prior)
        reg_log_prior = 0.5 * sigmoid_log_prior
        log_posterior = reg_suc_prob + reg_log_prior
        # print 'reg_suc_prob:', reg_suc_prob 
        # print 'reg_log_prior:', reg_log_prior
        # print 'log_posterior:', log_posterior
        return np.float64(log_posterior)


    def grasp_neg_log_posterior(self, ik_config, grasp_voxel_grid,
                            grasp_obj_size, mat_world_to_obj,
                            spatial_jac=None):
        return -self.grasp_log_posterior(ik_config, 
                        grasp_voxel_grid, grasp_obj_size, 
                        mat_world_to_obj)
        # return -self.grasp_norm_posterior(grasp_config, 
        #                 grasp_voxel_grid, grasp_obj_size)


    def grasp_log_posterior_grad(self, ik_config, grasp_voxel_grid,
                                grasp_obj_size, mat_world_to_obj, 
                                spatial_jac):
        J = self.cvtf.grasp_kin.jacobian(ik_config[:self.cvtf.arm_joints_dof])
        palm_pose_jac = np.matmul(spatial_jac, J)
        # Convert numpy matrix to numpy array
        palm_pose_jac = np.array(palm_pose_jac)

        num_jac = self.cvtf.compute_num_jac(self.cvtf.convert_to_fk_config, 
                                            ik_config, mat_world_to_obj)

        # print palm_pose_jac
        # print '##########################'
        # print num_jac
        # print '*************************'
        # print palm_pose_jac - num_jac

        grasp_config = self.cvtf.convert_to_fk_config(ik_config, mat_world_to_obj)
        clf_log_suc_grad = self.grasp_clf_log_suc_grad(grasp_config, grasp_voxel_grid,
                            grasp_obj_size)
        if self.use_suc_prior:
            log_prior_grad = self.suc_log_prior_grad(grasp_config)
        else:
            log_prior_grad = self.grasp_log_prior_grad(grasp_config)
        clf_log_suc_grad_arm = np.matmul(clf_log_suc_grad[:self.cvtf.palm_dof_dim], 
                                        palm_pose_jac) 
        clf_log_suc_grad_ik = np.concatenate((clf_log_suc_grad_arm, 
                                        clf_log_suc_grad[self.cvtf.palm_dof_dim:]))

        log_prior_grad_arm = np.matmul(log_prior_grad[:self.cvtf.palm_dof_dim], 
                                        palm_pose_jac) 
        log_prior_grad_ik = np.concatenate((log_prior_grad_arm, 
                                        log_prior_grad[self.cvtf.palm_dof_dim:]))

        reg_log_suc_grad = clf_log_suc_grad_ik * self.reg_log_lkh
        reg_log_prior_grad = log_prior_grad_ik * self.reg_log_prior
        log_post_grad = reg_log_suc_grad + reg_log_prior_grad
        # print 'reg_log_suc_grad', reg_log_suc_grad
        # print 'reg_log_prior_grad:', reg_log_prior_grad
        # print 'log_post_grad:', log_post_grad 

        #Gradient checking
        grad_check = False 
        if grad_check:
            num_grad = self.compute_num_ik_grad(self.grasp_log_posterior_fk, 
                            ik_config, grasp_config, grasp_voxel_grid, 
                            grasp_obj_size, mat_world_to_obj)
            grad_diff = log_post_grad - num_grad
            print 'log_post_grad:', log_post_grad
            print 'num_grad:', num_grad
            print 'Gradient difference:', grad_diff
            print 'Gradient difference abs mean:', np.mean(abs(grad_diff))
            print 'Gradient difference abs mean percentage:',  \
                    np.mean(abs(grad_diff)) / np.mean(abs(log_post_grad))
            print '~~~~~~~~~~~~~~~~~~~~~~~~~~~'

            num_grad = self.compute_num_grad(self.compute_suc_log_prior, 
                                                grasp_config, None, None)
            grad_diff = log_prior_grad - num_grad
            print 'log_prior_grad:', log_prior_grad
            print 'num_grad:', num_grad
            print 'Gradient difference:', grad_diff
            print 'Gradient difference abs mean:', np.mean(abs(grad_diff))
            print 'Gradient difference abs mean percentage:',  \
                    np.mean(abs(grad_diff)) / np.mean(abs(log_prior_grad))
            print '-----------------------------'

            num_grad = self.compute_num_grad(self.grasp_clf_log_suc_prob, 
                            grasp_config, grasp_voxel_grid, grasp_obj_size)
            grad_diff = clf_log_suc_grad - num_grad
            print 'clf_log_suc_grad:', clf_log_suc_grad
            print 'num_grad:', num_grad
            print 'Gradient difference:', grad_diff
            print 'Gradient difference abs mean:', np.mean(abs(grad_diff))
            print 'Gradient difference abs mean percentage:',  \
                    np.mean(abs(grad_diff)) / np.mean(abs(clf_log_suc_grad))
            print '+++++++++++++++++++++++++++++'

            print '################################################################'

        return log_post_grad.astype('float64')


    def grasp_norm_posterior_grad(self, grasp_config, grasp_voxel_grid,
                            grasp_obj_size):
        suc_grad, suc_prob = self.compute_clf_config_grad(grasp_config, 
                                        grasp_voxel_grid, grasp_obj_size)
        log_prior = self.compute_grasp_log_prior(grasp_config) 
        log_prior_grad = self.grasp_log_prior_grad(grasp_config)
        reg_suc_grad = 0.5 * suc_grad
        reg_log_prior_grad = 0.5 * self.sigmoid_grad(log_prior) * log_prior_grad
        log_post_grad = reg_suc_grad + reg_log_prior_grad
        # print 'reg_suc_grad', reg_suc_grad
        # print 'reg_log_prior_grad:', reg_log_prior_grad
        # print 'log_post_grad:', log_post_grad 

        #Gradient checking
        grad_check = False #True 
        if grad_check:
            num_grad = self.compute_num_grad(self.compute_grasp_log_prior, 
                                                grasp_config, None, None)
            grad_diff = log_prior_grad - num_grad
            print 'log_prior_grad:', log_prior_grad
            print 'num_grad:', num_grad
            print 'Gradient difference:', grad_diff
            print 'Gradient difference abs mean:', np.mean(abs(grad_diff))
            print 'Gradient difference abs mean percentage:',  \
                    np.mean(abs(grad_diff)) / np.mean(abs(log_prior_grad))
            print '-----------------------------'

            num_grad = self.compute_num_grad(self.pred_clf_suc_prob, 
                            grasp_config, grasp_voxel_grid, grasp_obj_size)
            grad_diff = suc_grad - num_grad
            print 'suc_grad:', suc_grad
            print 'num_grad:', num_grad
            print 'Gradient difference:', grad_diff
            print 'Gradient difference abs mean:', np.mean(abs(grad_diff))
            print 'Gradient difference abs mean percentage:',  \
                    np.mean(abs(grad_diff)) / np.mean(abs(suc_grad))
            print '+++++++++++++++++++++++++++++'

            num_grad = self.compute_num_grad(self.grasp_norm_posterior,
                            grasp_config, grasp_voxel_grid, grasp_obj_size)
            grad_diff = log_post_grad - num_grad
            print 'log_post_grad:', log_post_grad
            print 'num_grad:', num_grad
            print 'Gradient difference:', grad_diff
            print 'Gradient difference abs mean:', np.mean(abs(grad_diff))
            print 'Gradient difference abs mean percentage:',  \
                    np.mean(abs(grad_diff)) / np.mean(abs(log_post_grad))
            print '~~~~~~~~~~~~~~~~~~~~~~~~~~~'

            print '################################################################'

        return log_post_grad.astype('float64')


    def grasp_neg_log_post_grad(self, ik_config, grasp_voxel_grid,
                                grasp_obj_size, mat_world_to_obj, 
                                spatial_jac):
        return  -self.grasp_log_posterior_grad(ik_config, grasp_voxel_grid, 
                                            grasp_obj_size, mat_world_to_obj,
                                            spatial_jac)
        # return  -self.grasp_norm_posterior_grad(grasp_config, 
        #                 grasp_voxel_grid, grasp_obj_size)


    def grasp_clf_uncertainty(self, grasp_config, grasp_voxel_grid,
                            grasp_obj_size):
        suc_prob = self.pred_clf_suc_prob(grasp_config, 
                                    grasp_voxel_grid, grasp_obj_size)
        # print 'suc_prob:', suc_prob
        if suc_prob <= 0.5:
            uncertainty = suc_prob
        else:
            uncertainty = 1. - suc_prob
        return np.float64(uncertainty)
 

    def grasp_clf_uncertainty_grad(self, grasp_config, grasp_voxel_grid,
                            grasp_obj_size):
        config_grad, suc_prob = self.compute_clf_config_grad(grasp_config, 
                                        grasp_voxel_grid, grasp_obj_size)
        # print 'config_grad:', config_grad
        if suc_prob < 0.5:
            uncertainty_grad = config_grad
        elif suc_prob > 0.5:
            uncertainty_grad = -config_grad
        else:
            uncertainty_grad = np.zeros(self.config_limits.config_dim)
        return uncertainty_grad.astype('float64')


    def grasp_clf_square_uct(self, grasp_config, grasp_voxel_grid,
                            grasp_obj_size):
        suc_prob = self.pred_clf_suc_prob(grasp_config, 
                                    grasp_voxel_grid, grasp_obj_size)
        # print 'suc_prob:', suc_prob
        square_uct = 2. * (0.5 - suc_prob) ** 2.
        return np.float64(square_uct)
 

    def grasp_clf_sq_uct_grad(self, grasp_config, grasp_voxel_grid,
                            grasp_obj_size):
        config_grad, suc_prob = self.compute_clf_config_grad(grasp_config, 
                                        grasp_voxel_grid, grasp_obj_size)
        # print 'config_grad:', config_grad
        sq_uct_grad = -4. * (0.5 - suc_prob) * config_grad
        return sq_uct_grad.astype('float64')


    def grasp_neg_clf_uct(self, grasp_config, grasp_voxel_grid,
                        grasp_obj_size):
        # return -self.grasp_clf_uncertainty(grasp_config, 
        #         grasp_voxel_grid, grasp_obj_size)

        return -self.grasp_clf_square_uct(grasp_config, 
                grasp_voxel_grid, grasp_obj_size)


    def grasp_neg_clf_uct_grad(self, grasp_config, grasp_voxel_grid,
                            grasp_obj_size):
        # return -self.grasp_clf_uncertainty_grad(grasp_config, 
        #         grasp_voxel_grid, grasp_obj_size)

        return -self.grasp_clf_sq_uct_grad(grasp_config, 
                grasp_voxel_grid, grasp_obj_size)


    def sigmoid(self, x, base=np.exp(1)):
        "Numerically stable sigmoid function."
        if x >= 0:
            z = base ** -x
            return 1 / (1 + z)
        else:
            # if x is less than zero then z will be small, denom can't be
            # zero because it's 1+z.
            z = base ** x
            return z / (1 + z)


    def sigmoid_grad(self, x, base=np.exp(1)):
        sig = self.sigmoid(x, base)
        return sig * (1. - sig)


    def grasp_reg_clf_uct_fk(self, grasp_config, grasp_voxel_grid,
                        grasp_obj_size):
        # clf_uct = self.grasp_clf_uncertainty(grasp_config, 
        #         grasp_voxel_grid, grasp_obj_size)
        clf_uct = self.grasp_clf_square_uct(grasp_config, 
                grasp_voxel_grid, grasp_obj_size)
        log_prior = self.compute_grasp_log_prior(grasp_config) 
        # sigmoid_log_prior = 0.5 / (1. + np.exp(-log_prior))
        sigmoid_log_prior = self.sigmoid(log_prior) 
        reg_uct = clf_uct + 0.5 * sigmoid_log_prior 
        # print 'clf_uct:', clf_uct
        # print 'log_prior:', log_prior
        # print 'sigmoid_log_prior:', sigmoid_log_prior
        # print 'reg_uct:', reg_uct
        return np.float64(reg_uct)


    def grasp_reg_clf_uct(self, ik_config, grasp_voxel_grid,
                        grasp_obj_size, mat_world_to_obj):
        grasp_config = self.cvtf.convert_to_fk_config(ik_config, 
                                                    mat_world_to_obj)
        reg_uct = self.grasp_reg_clf_uct_fk(grasp_config, grasp_voxel_grid,
                                        grasp_obj_size)
        return reg_uct


    def sigmoid_log_prior(self, grasp_config):
        log_prior = self.compute_grasp_log_prior(grasp_config) 
        return self.sigmoid(log_prior)
           

    def grasp_reg_clf_uct_grad(self, ik_config, grasp_voxel_grid,
                        grasp_obj_size, mat_world_to_obj, spatial_jac):
        J = self.cvtf.grasp_kin.jacobian(ik_config[:self.cvtf.arm_joints_dof])
        palm_pose_jac = np.matmul(spatial_jac, J)
        # Convert numpy matrix to numpy array
        palm_pose_jac = np.array(palm_pose_jac)

        grasp_config = self.cvtf.convert_to_fk_config(ik_config, mat_world_to_obj)

        # clf_uct_grad = self.grasp_clf_uncertainty_grad(grasp_config, 
        #                             grasp_voxel_grid, grasp_obj_size)
        clf_uct_grad = self.grasp_clf_sq_uct_grad(grasp_config, 
                                    grasp_voxel_grid, grasp_obj_size)
        log_prior = self.compute_grasp_log_prior(grasp_config) 
        log_prior_grad = self.grasp_log_prior_grad(grasp_config) 

        clf_uct_grad_arm = np.matmul(clf_uct_grad[:self.cvtf.palm_dof_dim], 
                                        palm_pose_jac) 
        clf_uct_grad_ik = np.concatenate((clf_uct_grad_arm, 
                                        clf_uct_grad[self.cvtf.palm_dof_dim:]))

        log_prior_grad_arm = np.matmul(log_prior_grad[:self.cvtf.palm_dof_dim], 
                                        palm_pose_jac)
        log_prior_grad_ik = np.concatenate((log_prior_grad_arm, 
                                        log_prior_grad[self.cvtf.palm_dof_dim:]))

        # reg_prior_grad_ik = 0.5 * np.exp(-log_prior) / \
        #                 (1. + np.exp(-log_prior)) ** 2 * log_prior_grad_ik
        reg_prior_grad_ik = 0.5 * self.sigmoid_grad(log_prior) * log_prior_grad_ik

        reg_uct_grad_ik = clf_uct_grad_ik + reg_prior_grad_ik
        # print 'log_prior:', log_prior
        # print 'self.sigmoid_grad(log_prior):', self.sigmoid_grad(log_prior)
        # print 'log_prior_grad:', log_prior_grad
        # print 'clf_uct_grad_ik:', clf_uct_grad_ik
        # print 'reg_prior_grad_ik:', reg_prior_grad_ik
        # print 'reg_uct_grad_ik:', reg_uct_grad_ik 

        #Gradient checking
        grad_check = False 
        if grad_check:
            print 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
            # num_grad = self.compute_num_grad(self.grasp_reg_clf_uct, 
            #                         ik_config, grasp_voxel_grid, 
            #                         grasp_obj_size, mat_world_to_obj)
            num_grad = self.compute_num_ik_grad(self.grasp_reg_clf_uct_fk, 
                            ik_config, grasp_config, grasp_voxel_grid, 
                            grasp_obj_size, mat_world_to_obj)
            grad_diff = reg_uct_grad_ik - num_grad
            print 'reg_uct_grad:', reg_uct_grad_ik
            print 'num_grad:', num_grad
            print 'Gradient difference:', grad_diff
            print 'Gradient difference abs mean:', np.mean(abs(grad_diff))
            print 'Gradient difference abs mean percentage:',  \
                    np.mean(abs(grad_diff)) / np.mean(abs(reg_uct_grad_ik))
            print '+++++++++++++++++++++++++++++'

            num_grad = self.compute_num_grad(self.compute_grasp_log_prior, 
                                                grasp_config, None, None)
            grad_diff = log_prior_grad - num_grad
            print 'log_prior_grad:', log_prior_grad
            print 'num_grad:', num_grad
            print 'Gradient difference:', grad_diff
            print 'Gradient difference abs mean:', np.mean(abs(grad_diff))
            print 'Gradient difference abs mean percentage:',  \
                    np.mean(abs(grad_diff)) / np.mean(abs(log_prior_grad))
            print '-----------------------------'

            num_grad = self.compute_num_grad(self.sigmoid_log_prior, 
                                                grasp_config, None, None)
            num_grad *= 0.5
            reg_prior_grad = 0.5 * self.sigmoid_grad(log_prior) * log_prior_grad
            grad_diff = reg_prior_grad - num_grad
            print 'reg_prior_grad:', reg_prior_grad
            print 'num_grad:', num_grad
            print 'Gradient difference:', grad_diff
            print 'Gradient difference abs mean:', np.mean(abs(grad_diff))
            print 'Gradient difference abs mean percentage:',  \
                    np.mean(abs(grad_diff)) / np.mean(abs(reg_prior_grad))
            print '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$'

            # num_grad = self.compute_num_grad(self.grasp_clf_uncertainty, 
            #                 grasp_config, grasp_voxel_grid, grasp_obj_size)
            num_grad = self.compute_num_grad(self.grasp_clf_square_uct, 
                            grasp_config, grasp_voxel_grid, grasp_obj_size)
            grad_diff = clf_uct_grad - num_grad
            print 'clf_uct_grad:', clf_uct_grad
            print 'num_grad:', num_grad
            print 'Gradient difference:', grad_diff
            print 'Gradient difference abs mean:', np.mean(abs(grad_diff))
            print 'Gradient difference abs mean percentage:',  \
                    np.mean(abs(grad_diff)) / np.mean(abs(clf_uct_grad))
            print '******************************'

        return reg_uct_grad_ik


    def neg_reg_clf_uct(self, ik_config, grasp_voxel_grid,
                        grasp_obj_size, mat_world_to_obj,
                        spatial_jac=None):
        return -self.grasp_reg_clf_uct(ik_config, grasp_voxel_grid,
                        grasp_obj_size, mat_world_to_obj)


    def neg_reg_clf_uct_grad(self, ik_config, grasp_voxel_grid,
                            grasp_obj_size, mat_world_to_obj, 
                            spatial_jac):
        return -self.grasp_reg_clf_uct_grad(ik_config, grasp_voxel_grid,
                        grasp_obj_size, mat_world_to_obj, spatial_jac)


    def sample_grasp_config(self):
        if self.prior_name == 'MDN':
            config_sample = self.sample_grasp_config_mdn()
        elif self.prior_name == 'GMM':
            config_sample, _ = self.sample_grasp_config_gmm()

        return config_sample


    def sample_suc_grasp_config(self):
        if self.prior_name == 'MDN':
            config_sample = self.sample_suc_config_mdn()
        elif self.prior_name == 'GMM':
            config_sample, _ = self.sample_suc_config_gmm()

        return config_sample


    def sample_grasp_config_mdn(self):
        feed_dict = {self.locs_holder: self.locs, 
                    self.scales_holder: self.scales,
                    self.logits_holder: self.logits}
        [sample] = self.tf_sess.run([self.prior_sample], 
                                    feed_dict=feed_dict)  
        return sample


    def sample_suc_config_mdn(self):
        feed_dict = {self.suc_locs_holder: self.suc_locs, 
                    self.suc_scales_holder: self.suc_scales,
                    self.suc_logits_holder: self.suc_logits}
        [sample] = self.tf_sess.run([self.suc_prior_sample], 
                                    feed_dict=feed_dict)  
        return sample


    def sample_grasp_config_gmm(self):
        # Set the random_state of the GMM to be None so that it 
        # generates different random samples
        if self.gmm_prior_model.random_state != None:
            self.gmm_prior_model.random_state = None
        return self.gmm_prior_model.sample()


    def sample_suc_config_gmm(self):
        if self.suc_gmm_prior_model.random_state != None:
            self.suc_gmm_prior_model.random_state = None
        return self.suc_gmm_prior_model.sample()


    def ik_config_strategy_sample(self, score_func, mat_obj_to_world, 
                                   grasp_voxel_grid=None, grasp_obj_size=None,
                                   suc_prior=False, samples_num=50):
        max_score = -float('inf')
        explore_config = None
        explore_ik_config = None
        i = 0
        fail_num = 0 
        max_fail_num = 10 * samples_num
        while i < samples_num and fail_num < max_fail_num:
            if suc_prior:
                config_sample = self.sample_suc_grasp_config()
            else:
                config_sample = self.sample_grasp_config()
            config_sample = config_sample[0].astype('float64')
            ik_config_sample = self.cvtf.convert_to_ik_config(config_sample, 
                                                            mat_obj_to_world)
            if ik_config_sample is None:
                fail_num += 1
                continue
            if not self.config_limits.inside_ik_config_limits(ik_config_sample):
                fail_num += 1
                continue
            if grasp_voxel_grid is None:
                score = score_func(config_sample)
            else:
                score = score_func(config_sample, grasp_voxel_grid, 
                                    grasp_obj_size)
            if score > max_score:
                explore_config = config_sample
                explore_ik_config = ik_config_sample
                max_score = score
            i += 1
        return explore_ik_config, explore_config, max_score


    def grasp_config_init(self, mat_obj_to_world, suc_prior=False):
        ik_config_init = None
        config_init = None
        i = -1
        max_init_times = 100
        while ik_config_init is None and i < max_init_times:
            i += 1
            if suc_prior:
                config_sample = self.sample_suc_grasp_config()
            else:
                config_sample = self.sample_grasp_config()
            config_init = config_sample[0].astype('float64')
            # print 'config_init:', config_init
            # ik_config_init_tf = self.cvtf.convert_to_ik_config_tf(config_init)
            ik_config_init = self.cvtf.convert_to_ik_config(config_init, mat_obj_to_world)
            if ik_config_init is None:
                continue
            if not self.config_limits.inside_ik_config_limits(ik_config_init):
                ik_config_init = None

        return ik_config_init, config_init

    
    def grasp_strategies(self, strategy, grasp_voxel_grid,
                        grasp_obj_size, pooling=False, bfgs=False): 
        t = time.time()
        
        opt_method = 'L-BFGS-B'
        if bfgs:
            opt_method = 'BFGS'

        bnds = []
        for i in xrange(self.config_limits.ik_config_dim):
            bnds.append((self.config_limits.ik_config_lower_limit[i], 
                         self.config_limits.ik_config_upper_limit[i]))

        bnds = np.array(bnds).astype('float64')
        #print 'bnds:', bnds
        
        res_info = {'inf_log_prior':-1, 'inf_suc_prob':-1, 'inf_uct':-1,
                    'init_log_prior':-1, 'init_suc_prob':-1, 'init_uct':-1}
        res_info['action'] = strategy

        ret_param_num = 9
        
        # Look up transformation from world to object
        mat_world_to_obj = self.cvtf.dpl.lookup_transform(self.cvtf.dpl.object_frame_id, 
                                            self.cvtf.dpl.world_frame_id)
        # Look up transformation from object to world
        mat_obj_to_world = self.cvtf.dpl.lookup_transform(self.cvtf.dpl.world_frame_id, 
                                            self.cvtf.dpl.object_frame_id)

        if self.prior_name == 'MDN':
            self.locs, self.scales, self.logits = [None] * 3
            self.get_prior_mixture(grasp_voxel_grid, grasp_obj_size)
            self.suc_locs, self.suc_scales, self.suc_logits = [None] * 3
            self.get_suc_prior_mixture(grasp_voxel_grid, grasp_obj_size)


        config_init = None
        ik_config_init = None
        if not pooling and strategy != 'config_explore':
            if strategy == 'grasp_suc':
                ik_config_init, config_init = self.grasp_config_init(mat_obj_to_world, 
                                                                    suc_prior=self.use_suc_prior)
            elif strategy == 'grasp_uct':
                ik_config_init, config_init = self.grasp_config_init(mat_obj_to_world)

            if ik_config_init is None:
                rospy.logerr('Could not find a valid initialization!')
                return [None] * ret_param_num 

            print 'ik_config_init:', ik_config_init

            spatial_jac = np.zeros((6, 6)) 
            spatial_jac[:3, :3] = mat_world_to_obj[:3, :3]
            spatial_jac[3:, 3:] = mat_world_to_obj[:3, :3]

        if strategy == 'grasp_suc':
            if pooling:
                ik_config_inf, config_inf, obj_val_inf = \
                        self.ik_config_strategy_sample(self.grasp_log_posterior_fk,
                                                        mat_obj_to_world, 
                                                        grasp_voxel_grid, 
                                                        grasp_obj_size,
                                                        suc_prior=self.use_suc_prior)
                if ik_config_inf is None:
                    rospy.logerr('Could not sample a valid config for max success!')
                    return [None] * ret_param_num 
            else:
                opt_res = minimize(self.grasp_neg_log_posterior, ik_config_init, 
                                    jac=self.grasp_neg_log_post_grad, 
                                    args=(grasp_voxel_grid, grasp_obj_size, 
                                            mat_world_to_obj, spatial_jac), 
                                    method=opt_method, bounds=bnds)
                print opt_res

                obj_val_inf = -opt_res.fun
                ik_config_inf = opt_res.x
                config_inf = self.cvtf.convert_to_fk_config(ik_config_inf, 
                                                        mat_world_to_obj)
                obj_val_init = self.grasp_log_posterior(ik_config_init, 
                                                grasp_voxel_grid, grasp_obj_size, 
                                                mat_world_to_obj)
                # obj_val_init = self.grasp_norm_posterior(config_init, 
                #                                 grasp_voxel_grid, grasp_obj_size)


                print 'obj_val_init:', obj_val_init
                init_suc_prob = self.pred_clf_suc_prob(config_init, 
                                        grasp_voxel_grid, grasp_obj_size)
                init_log_prior = self.compute_grasp_log_prior(config_init)
                res_info['init_suc_prob'] = init_suc_prob
                res_info['init_log_prior'] = init_log_prior
                print 'init_suc_prob:', init_suc_prob
                print 'init_suc_log_prob:', np.log(init_suc_prob)
                print 'init_log_prior:', init_log_prior
 
            print 'obj_val_inf:', obj_val_inf
            inf_suc_prob = self.pred_clf_suc_prob(config_inf, 
                                    grasp_voxel_grid, grasp_obj_size)
            inf_log_prior = self.compute_grasp_log_prior(config_inf)
            res_info['inf_suc_prob'] = inf_suc_prob
            res_info['inf_log_prior'] = inf_log_prior
            print 'inf_suc_prob:', inf_suc_prob
            print 'inf_suc_log_prob:', np.log(inf_suc_prob)
            print 'inf_log_prior:', inf_log_prior

            # reward = self.sigmoid(obj_val_inf, base=1.1)
            reward = self.sigmoid(obj_val_inf, base=1.1)
            if np.isclose(reward, 0.):
                return [None] * ret_param_num  
            reward += 0.35
            # reward = obj_val_inf
        elif strategy == 'grasp_uct':
            if pooling:
                ik_config_inf, config_inf, obj_val_inf = \
                        self.ik_config_strategy_sample(self.grasp_reg_clf_uct_fk,
                                                        mat_obj_to_world, 
                                                        grasp_voxel_grid, 
                                                        grasp_obj_size)
                if ik_config_inf is None:
                    rospy.logerr('Could not sample a valid config for max uncertainty!')
                    return [None] * ret_param_num 
            else:
                # opt_res = minimize(self.grasp_neg_clf_uct, config_init, 
                #                     jac=self.grasp_neg_clf_uct_grad, 
                #                     args=(grasp_voxel_grid, grasp_obj_size,), 
                #                     method=opt_method, bounds=bnds)
                # obj_val_init = self.grasp_clf_uncertainty(config_init, 
                #                                 grasp_voxel_grid, grasp_obj_size)
                # obj_val_inf = -opt_res.fun
                # config_inf = opt_res.x
                # reward = 2 * obj_val_inf

                opt_res = minimize(self.neg_reg_clf_uct, ik_config_init, 
                                    jac=self.neg_reg_clf_uct_grad, 
                                    args=(grasp_voxel_grid, grasp_obj_size, 
                                            mat_world_to_obj, spatial_jac), 
                                    method=opt_method, bounds=bnds)
                print opt_res

                obj_val_inf = -opt_res.fun
                ik_config_inf = opt_res.x
                config_inf = self.cvtf.convert_to_fk_config(ik_config_inf, mat_world_to_obj)

                obj_val_init = self.grasp_reg_clf_uct(ik_config_init, grasp_voxel_grid, 
                                                    grasp_obj_size, mat_world_to_obj)
                print 'obj_val_init:', obj_val_init
                clf_uct_init = self.grasp_clf_square_uct(config_init, 
                        grasp_voxel_grid, grasp_obj_size)
                log_prior_init = self.compute_grasp_log_prior(config_init) 
                res_info['init_uct'] = clf_uct_init
                res_info['init_log_prior'] = log_prior_init
                # sigmoid_log_prior_init = self.sigmoid(log_prior_init) 
                # reg_uct_init = clf_uct_init + 0.5 * sigmoid_log_prior_init 
                print 'clf_uct_init:', clf_uct_init
                print 'log_prior_init:', log_prior_init
                # print 'sigmoid_log_prior_init:', sigmoid_log_prior_init
                # print 'reg_uct_init:', reg_uct_init
            
            print 'obj_val_inf:', obj_val_inf
            clf_uct_inf = self.grasp_clf_square_uct(config_inf, 
                    grasp_voxel_grid, grasp_obj_size)
            log_prior_inf = self.compute_grasp_log_prior(config_inf) 
            res_info['inf_uct'] = clf_uct_inf
            res_info['inf_log_prior'] = log_prior_inf
            # sigmoid_log_prior_inf = self.sigmoid(log_prior_inf) 
            # reg_uct_inf = clf_uct_inf + 0.5 * sigmoid_log_prior_inf 
            print 'clf_uct_inf:', clf_uct_inf
            print 'log_prior_inf:', log_prior_inf
            # print 'sigmoid_log_prior_inf:', sigmoid_log_prior_inf
            # print 'reg_uct_inf:', reg_uct_inf

            reward = obj_val_inf
            reward -= 0.05
        elif strategy == 'config_explore': 
            # opt_res = minimize(self.compute_grasp_prior, config_init, 
            #                     jac=self.grasp_prior_grad, 
            #                     method=opt_method, bounds=bnds)
            # obj_val_init = self.compute_grasp_prior(config_init) 

            # # opt_res = minimize(self.compute_grasp_log_prior, config_init, 
            # #                     jac=self.grasp_log_prior_grad, 
            # #                     method=opt_method, bounds=bnds)
            # # obj_val_init = self.compute_grasp_log_prior(config_init) 
            # obj_val_inf = opt_res.fun
            # config_inf = opt_res.x
            
            # ik_config_inf, config_inf, obj_val_inf = \
            #         self.ik_config_explore_sample(mat_obj_to_world)

            ik_config_inf, config_inf, obj_val_inf = \
                    self.ik_config_strategy_sample(self.neg_grasp_log_prior,
                                                    mat_obj_to_world)

            if ik_config_inf is None:
                rospy.logerr('Could not sample a valid config for exploration!')
                return [None] * ret_param_num 

            res_info['inf_log_prior'] = -obj_val_inf
            print '-obj_val_inf:', -obj_val_inf

            # Notice: it should be the sigmoid for the negative log prior
            reward = self.sigmoid(obj_val_inf, base=1.1) 
            reward += 0.6
        else:
            print 'Strategy not found!'

        if pooling or strategy == 'config_explore':
            obj_val_init = -1.
            config_init = np.zeros(config_inf.shape[0])
            ik_config_init = np.zeros(ik_config_inf.shape[0])

        elapased_time = time.time() - t
        print 'Total inference time: ', str(elapased_time)

        return reward, ik_config_inf, config_inf, obj_val_inf, \
                ik_config_init, config_init, obj_val_init, \
                res_info, elapased_time 

    
    def ucb(self, voxel_grid, obj_size, pooling=False):
        print 'UCB:'
        if self.total_action_num < self.strategies_num:
            print 'Execute initial actions...'
            self.cur_action_id = self.total_action_num
            self.cur_act_reward, self.cur_ik_config_inf, self.cur_config_inf,\
                    self.cur_val_inf, self.cur_ik_config_init, self.cur_config_init,\
                    self.cur_val_init, self.res_info, self.cur_inf_time = \
                        self.grasp_strategies(
                                            self.al_strategies[self.cur_action_id], 
                                            voxel_grid, obj_size, pooling)
        else:
            max_score = -1.
            max_action_id = -1
            for i in xrange(self.strategies_num):
                action_num = len(self.rewards[i])
                exploration_score = np.sqrt(2 * 
                            np.log(self.total_action_num) / action_num)
                score = self.avg_rewards[i] + exploration_score 
                # print 'strategy:', self.al_strategies[i]
                # print 'avg_reward:', self.avg_rewards[i]
                # print 'exploration_score:', exploration_score
                # print 'score:', score
                if score > max_score:
                    max_score = score
                    max_action_id = i

            print 'max_score:', max_score
            print 'max_action_id:', max_action_id
            print 'max strategy:', self.al_strategies[max_action_id]
            self.cur_act_reward, self.cur_ik_config_inf, self.cur_config_inf,\
                    self.cur_val_inf, self.cur_ik_config_init, self.cur_config_init,\
                    self.cur_val_init, self.res_info, self.cur_inf_time = \
                        self.grasp_strategies(
                                            self.al_strategies[max_action_id], 
                                            voxel_grid, obj_size, pooling)

            self.cur_action_id = max_action_id

        print 'self.cur_action_id:', self.cur_action_id
        print 'self.cur_act_reward:', self.cur_act_reward
    
        return  self.cur_act_reward, self.cur_ik_config_inf, \
                self.cur_config_inf, self.cur_val_inf, \
                self.cur_ik_config_init, self.cur_config_init, \
                self.cur_val_init, self.res_info

    
    def alternation(self, voxel_grid, obj_size, pooling=False):
        print 'Alternation:'
        self.cur_action_id = self.total_action_num % self.strategies_num
        # self.cur_action_id = 1
        self.cur_act_reward, self.cur_ik_config_inf, self.cur_config_inf,\
                self.cur_val_inf, self.cur_ik_config_init, self.cur_config_init,\
                self.cur_val_init, self.res_info, self.cur_inf_time = \
                    self.grasp_strategies(
                                        self.al_strategies[self.cur_action_id], 
                                        voxel_grid, obj_size, pooling)
        print 'self.cur_action_id:', self.cur_action_id
        print 'self.cur_act_reward:', self.cur_act_reward
    
        return  self.cur_act_reward, self.cur_ik_config_inf, \
                self.cur_config_inf, self.cur_val_inf, \
                self.cur_ik_config_init, self.cur_config_init, \
                self.cur_val_init, self.res_info


    def active_data_update(self):
        action_num = len(self.rewards[self.cur_action_id])
        self.avg_rewards[self.cur_action_id] = (self.avg_rewards[self.cur_action_id] * 
                            action_num + self.cur_act_reward) / (action_num + 1.)
        self.actions.append(self.cur_action_id)
        self.rewards[self.cur_action_id].append(self.cur_act_reward)
        self.inf_time[self.cur_action_id].append(self.cur_inf_time)
        self.total_action_num += 1
        print 'self.actions:', self.actions
        print 'self.rewards:', self.rewards
        print 'self.avg_rewards:', self.avg_rewards 
        print 'self.total_action_num:', self.total_action_num
        print 'self.inf_time:', self.inf_time

        al_file = h5py.File(self.al_file_path, 'r+')
        al_file['total_action_num'][()] = self.total_action_num
        al_file['al_strategies'][()] = self.al_strategies
        del al_file['actions'] 
        al_file.create_dataset('actions', data=self.actions)
        for i in xrange(len(self.al_strategies)):
            del al_file['rewards_strategy_' + str(i)] 
            al_file.create_dataset('rewards_strategy_' + str(i), 
                                    data=self.rewards[i])
        al_file['avg_rewards'][()] = self.avg_rewards
        for i in xrange(len(self.al_strategies)):
            del al_file['inf_time_strategy_' + str(i)] 
            al_file.create_dataset('inf_time_strategy_' + str(i), 
                                    data=self.inf_time[i])
        al_file.close()

 
    def incremental_train_grasp_clf(self, batch_id):
        is_train = True
        dropout = False
        update_voxel_enc = False
        logs_path_train = self.tf_logs_path + '/al_net_incrmt_train_' + \
                                    'batch_' + str(batch_id)
        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter(logs_path_train, 
                                                graph=self.tf_sess.graph)
        training_iter = 5
        # learning_rates = {0:0.0001, 3:0.00001}
        learning_rates = {0:0.00001}
        learning_rate = None
        start_time = time.time()
        iter_num = 0
        binary_threshold = 0.5
        grasp_configs, grasp_voxel_grids, grasp_obj_sizes, grasp_labels = \
        self.adl.active_batch(batch_id * self.batch_size, 
                                (batch_id + 1) * self.batch_size)

        while iter_num < training_iter:
            if iter_num in learning_rates:
                learning_rate = learning_rates[iter_num]

            # Don't need to feed the learning rate of voxel ae, 
            # since only the optimizer in grasp_network is used.
            feed_dict = {self.grasp_net.voxel_ae.is_train: (is_train and update_voxel_enc),
                        self.grasp_net.voxel_ae.partial_voxel_grids: grasp_voxel_grids,
                        self.grasp_net.is_train: is_train,
                        self.grasp_net.holder_config: grasp_configs,
                        self.grasp_net.holder_obj_size: grasp_obj_sizes,
                        self.grasp_net.holder_labels: grasp_labels,
                        self.grasp_net.learning_rate: learning_rate,
                        }
            if dropout:
                feed_dict[self.grasp_net.keep_prob] = 0.9

            [batch_suc_prob, loss, _, train_summary] = \
                                            self.tf_sess.run([
                                                    self.grasp_net.grasp_net_res['suc_prob'], 
                                                    self.grasp_net.grasp_net_res['loss'],
                                                    self.grasp_net.grasp_net_res['opt_loss'],
                                                    self.grasp_net.grasp_net_res['train_summary'], ],
                                                    feed_dict=feed_dict)
            summary_writer.add_summary(train_summary, iter_num)

            print 'iter_num:', iter_num
            print 'learning_rate:', learning_rate
            print 'loss:', loss
            print 'suc_num:', np.sum(batch_suc_prob > 0.5)
            print 'true suc_num:', np.sum(grasp_labels)
            batch_suc_prob[batch_suc_prob > binary_threshold] = 1.
            batch_suc_prob[batch_suc_prob <= binary_threshold] = 0.
            prfc = precision_recall_fscore_support(grasp_labels, batch_suc_prob)
            print 'precision, recall, fscore, support:'
            print prfc

            iter_num += 1

        batch_models_path = self.active_models_path + 'batch_' \
                            + str(batch_id) + '_models'
        grasp_model_path = batch_models_path + '/batch_' \
                           + str(batch_id) + '_incrmt_net' 
        self.saver.save(self.tf_sess, grasp_model_path)
 
        elapsed_time = time.time() - start_time
        print 'Grasp net incremental training time: ', elapsed_time


    def retrain_grasp_clf(self, batch_id):
        is_train = True
        dropout = False
        update_voxel_enc = False
        logs_path_train = self.tf_logs_path + '/al_net_retrain_' + \
                                    'batch_' + str(batch_id)
        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter(logs_path_train, 
                                                graph=self.tf_sess.graph)
        training_epochs = 5
        # learning_rates = {0:0.0001, 3:0.00001}
        learning_rates = {0:0.00001}
        # training_epochs = 10
        # learning_rates = {0:0.0001, 5:0.00001}
        learning_rate = None
        start_time = time.time()
        iter_num = 0
        binary_threshold = 0.5

        # Reset the active data loader by recounting the samples 
        # number and setting the epochs number to be zero.
        self.adl.count_samples()
        self.adl.epochs_completed = 0
        print 'epochs_completed:', self.adl.epochs_completed
        print 'self.adl.num_samples:', self.adl.num_samples
        print 'self.adl.num_spv_samples:', self.adl.num_spv_samples
        print 'self.adl.num_act_samples:', self.adl.num_act_samples
        self.adl.shuffle(is_shuffle=True)
        while self.adl.epochs_completed < training_epochs:
            if self.adl.epochs_completed in learning_rates:
                learning_rate = learning_rates[self.adl.epochs_completed]

            grasp_configs, grasp_voxel_grids, grasp_obj_sizes, grasp_labels = \
                    self.adl.next_batch(self.batch_size)
            # Don't need to feed the learning rate of voxel ae, 
            # since only the optimizer in grasp_network is used.
            feed_dict = {self.grasp_net.voxel_ae.is_train: (is_train and update_voxel_enc),
                        self.grasp_net.voxel_ae.partial_voxel_grids: grasp_voxel_grids,
                        self.grasp_net.is_train: is_train,
                        self.grasp_net.holder_config: grasp_configs,
                        self.grasp_net.holder_obj_size: grasp_obj_sizes,
                        self.grasp_net.holder_labels: grasp_labels,
                        self.grasp_net.learning_rate: learning_rate,
                        }
            if dropout:
                feed_dict[self.grasp_net.keep_prob] = 0.9

            [batch_suc_prob, loss, _, train_summary] = \
                                            self.tf_sess.run([
                                                    self.grasp_net.grasp_net_res['suc_prob'], 
                                                    self.grasp_net.grasp_net_res['loss'],
                                                    self.grasp_net.grasp_net_res['opt_loss'],
                                                    self.grasp_net.grasp_net_res['train_summary'], ],
                                                    feed_dict=feed_dict)
            summary_writer.add_summary(train_summary, iter_num)

            if iter_num % 10 == 0:
                print 'epochs_completed:', self.adl.epochs_completed
                print 'iter_num:', iter_num
                print 'learning_rate:', learning_rate
                print 'loss:', loss
                print 'suc_num:', np.sum(batch_suc_prob > 0.5)
                print 'true suc_num:', np.sum(grasp_labels)
                batch_suc_prob[batch_suc_prob > binary_threshold] = 1.
                batch_suc_prob[batch_suc_prob <= binary_threshold] = 0.
                prfc = precision_recall_fscore_support(grasp_labels, batch_suc_prob)
                print 'precision, recall, fscore, support:'
                print prfc

            iter_num += 1

        batch_models_path = self.active_models_path + 'batch_' \
                            + str(batch_id) + '_models'
        grasp_model_path = batch_models_path + '/batch_' \
                           + str(batch_id) + '_retrain_net' 
        self.saver.save(self.tf_sess, grasp_model_path)
 
        elapsed_time = time.time() - start_time
        print 'Grasp net retraining time: ', elapsed_time


    def incremental_train_mdn_prior(self, batch_id):
        is_train = True
        update_voxel_enc = False
        logs_path_train = self.tf_logs_path + '/al_mdn_incrmt_train_' + \
                                    'batch_' + str(batch_id)
        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter(logs_path_train, 
                                                graph=self.tf_sess.graph)
        training_iter = 5
        # learning_rates = {0:0.0001, 3:0.00001}
        learning_rates = {0:0.00001}
        learning_rate = None
        start_time = time.time()
        iter_num = 0
        grasp_configs, grasp_voxel_grids, grasp_obj_sizes, grasp_labels = \
        self.mdn_adl.active_batch(batch_id * self.batch_size, 
                                (batch_id + 1) * self.batch_size)
        empty_configs = np.zeros(np.array(grasp_configs).shape) 
        empty_obj_sizes = np.zeros(np.array(grasp_obj_sizes).shape)
        while iter_num < training_iter:
            if iter_num in learning_rates:
                learning_rate = learning_rates[iter_num]

            feed_dict = {self.prior_net.voxel_ae.is_train: (is_train and update_voxel_enc),
                        self.prior_net.voxel_ae.partial_voxel_grids: grasp_voxel_grids,
                        self.prior_net.is_train: is_train,
                        self.prior_net.holder_config: grasp_configs,
                        self.prior_net.holder_obj_size: grasp_obj_sizes,
                        self.prior_net.learning_rate: learning_rate,
                        # Seems the MDN Adam optimizer requires the grasp net holders.
                        # I do not know exactly why yet. But I have verified the grasp net 
                        # holders don't affect the MDN training.
                        self.grasp_net.is_train: is_train,
                        self.grasp_net.holder_config: empty_configs,
                        self.grasp_net.holder_obj_size: empty_obj_sizes,
                        }

            [loss, _, train_summary] = self.tf_sess.run([self.prior_net.prior_net_res['loss'],
                                                 self.prior_net.prior_net_res['opt_loss'],
                                                 self.prior_net.prior_net_res['train_summary'], ],
                                                 feed_dict=feed_dict)
            summary_writer.add_summary(train_summary, iter_num)

            print 'iter_num:', iter_num
            print 'learning_rate:', learning_rate
            print 'loss:', loss
            iter_num += 1

        batch_models_path = self.active_models_path + 'batch_' \
                            + str(batch_id) + '_models'
        prior_model_path = batch_models_path + '/batch_' \
                           + str(batch_id) + '_incrmt_mdn' 
        self.prior_saver.save(self.tf_sess, prior_model_path)
 
        elapsed_time = time.time() - start_time
        print 'MDN prior incremental training time: ', elapsed_time
       
   
    def retrain_mdn_prior(self, batch_id):
        is_train = True
        update_voxel_enc = False
        logs_path_train = self.tf_logs_path + '/al_mdn_retrain_' + \
                                    'batch_' + str(batch_id)
        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter(logs_path_train, 
                                                graph=self.tf_sess.graph)
        training_epochs = 5
        # learning_rates = {0:0.0001, 3:0.00001}
        learning_rates = {0:0.00001}
        # training_epochs = 10
        # learning_rates = {0:0.0001, 5:0.00001}
        learning_rate = None
        start_time = time.time()
        iter_num = 0

        # Reset the active data loader by recounting the samples 
        # number and setting the epochs number to be zero.
        self.mdn_adl.count_samples()
        self.mdn_adl.epochs_completed = 0
        print 'epochs_completed:', self.mdn_adl.epochs_completed
        print 'self.mdn_adl.num_samples:', self.mdn_adl.num_samples
        print 'self.mdn_adl.num_spv_samples:', self.mdn_adl.num_spv_samples
        print 'self.mdn_adl.num_act_samples:', self.mdn_adl.num_act_samples
        self.mdn_adl.shuffle(is_shuffle=True)
        while self.mdn_adl.epochs_completed < training_epochs:
            if self.mdn_adl.epochs_completed in learning_rates:
                learning_rate = learning_rates[self.mdn_adl.epochs_completed]

            grasp_configs, grasp_voxel_grids, grasp_obj_sizes, grasp_labels = \
                self.mdn_adl.next_batch(self.batch_size)

            empty_configs = np.zeros(np.array(grasp_configs).shape) 
            empty_obj_sizes = np.zeros(np.array(grasp_obj_sizes).shape)

            feed_dict = {self.prior_net.voxel_ae.is_train: (is_train and update_voxel_enc),
                        self.prior_net.voxel_ae.partial_voxel_grids: grasp_voxel_grids,
                        self.prior_net.is_train: is_train,
                        self.prior_net.holder_config: grasp_configs,
                        self.prior_net.holder_obj_size: grasp_obj_sizes,
                        self.prior_net.learning_rate: learning_rate,
                        self.grasp_net.is_train: is_train,
                        self.grasp_net.holder_config: empty_configs,
                        self.grasp_net.holder_obj_size: empty_obj_sizes,
                        }

            [loss, _, train_summary] = self.tf_sess.run([self.prior_net.prior_net_res['loss'],
                                                 self.prior_net.prior_net_res['opt_loss'],
                                                 self.prior_net.prior_net_res['train_summary'], ],
                                                 feed_dict=feed_dict)
            summary_writer.add_summary(train_summary, iter_num)

            if iter_num % 10 == 0:
                print 'epochs_completed:', self.mdn_adl.epochs_completed
                print 'iter_num:', iter_num
                print 'learning_rate:', learning_rate
                print 'loss:', loss
            iter_num += 1
 
        batch_models_path = self.active_models_path + 'batch_' \
                            + str(batch_id) + '_models'
        prior_model_path = batch_models_path + '/batch_' \
                           + str(batch_id) + '_retrain_mdn' 
        self.prior_saver.save(self.tf_sess, prior_model_path)
             
        elapsed_time = time.time() - start_time
        print 'MDN prior retraining time: ', elapsed_time


    def retrain_suc_mdn_prior(self, batch_id):
        is_train = True
        update_voxel_enc = False
        logs_path_train = self.tf_logs_path + '/al_suc_mdn_retrain_' + \
                                    'batch_' + str(batch_id)
        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter(logs_path_train, 
                                                graph=self.tf_sess.graph)
        training_epochs = 5
        # learning_rates = {0:0.0001, 3:0.00001}
        learning_rates = {0:0.00001}
        # training_epochs = 10
        # learning_rates = {0:0.0001, 5:0.00001}
        learning_rate = None
        start_time = time.time()
        iter_num = 0

        # Reset the active data loader by recounting the samples 
        # number and setting the epochs number to be zero.
        self.suc_mdn_adl.count_samples()
        self.suc_mdn_adl.epochs_completed = 0
        print 'epochs_completed:', self.suc_mdn_adl.epochs_completed
        print 'self.suc_mdn_adl.num_samples:', self.suc_mdn_adl.num_samples
        print 'self.suc_mdn_adl.num_spv_samples:', self.suc_mdn_adl.num_spv_samples
        print 'self.suc_mdn_adl.num_act_samples:', self.suc_mdn_adl.num_act_samples
        self.suc_mdn_adl.shuffle(is_shuffle=True)
        while self.suc_mdn_adl.epochs_completed < training_epochs:
            if self.suc_mdn_adl.epochs_completed in learning_rates:
                learning_rate = learning_rates[self.suc_mdn_adl.epochs_completed]

            grasp_configs, grasp_voxel_grids, grasp_obj_sizes, grasp_labels = \
                self.suc_mdn_adl.next_batch(self.batch_size)

            empty_configs = np.zeros(np.array(grasp_configs).shape) 
            empty_obj_sizes = np.zeros(np.array(grasp_obj_sizes).shape)

            feed_dict = {self.suc_prior_net.voxel_ae.is_train: (is_train and update_voxel_enc),
                        self.suc_prior_net.voxel_ae.partial_voxel_grids: grasp_voxel_grids,
                        self.suc_prior_net.is_train: is_train,
                        self.suc_prior_net.holder_config: grasp_configs,
                        self.suc_prior_net.holder_obj_size: grasp_obj_sizes,
                        self.suc_prior_net.learning_rate: learning_rate,
                        # I have to feed these placeholders for the prior and grasp 
                        # net to run the suc prior net. I have verified these 
                        # placeholders do not affect the correctness of the suc prior 
                        # net training. 
                        self.prior_net.is_train: is_train,
                        self.prior_net.holder_config: empty_configs,
                        self.prior_net.holder_obj_size: empty_obj_sizes,
                        self.grasp_net.is_train: is_train,
                        self.grasp_net.holder_config: empty_configs,
                        self.grasp_net.holder_obj_size: empty_obj_sizes,
                        }

            [loss, _, train_summary] = self.tf_sess.run([self.suc_prior_net.prior_net_res['loss'],
                                                 self.suc_prior_net.prior_net_res['opt_loss'],
                                                 self.suc_prior_net.prior_net_res['train_summary'], ],
                                                 feed_dict=feed_dict)
            summary_writer.add_summary(train_summary, iter_num)

            if iter_num % 10 == 0:
                print 'epochs_completed:', self.suc_mdn_adl.epochs_completed
                print 'iter_num:', iter_num
                print 'learning_rate:', learning_rate
                print 'loss:', loss
            iter_num += 1
 
        batch_models_path = self.active_models_path + 'batch_' \
                            + str(batch_id) + '_models'
        suc_prior_path = batch_models_path + '/batch_' \
                           + str(batch_id) + '_retrain_suc_mdn' 
        self.suc_prior_saver.save(self.tf_sess, suc_prior_path)
             
        elapsed_time = time.time() - start_time
        print 'MDN prior retraining time: ', elapsed_time


    def refit_gmm_prior(self, batch_id):
        start_time = time.time()
        print self.gmm_prior_model.weights_
        print self.gmm_prior_model.means_
        self.gmm_prior_model = mixture.GaussianMixture(
                                    n_components=self.gmm_num_components, 
                                    covariance_type='full', random_state=0, 
                                    # init_params='kmeans', n_init=1)
                                    init_params='kmeans', n_init=5)

        self.adl.count_samples()
        grasp_configs = self.adl.load_grasp_configs()
        self.gmm_prior_model.fit(grasp_configs)
        batch_models_path = self.active_models_path + 'batch_' \
                            + str(batch_id) + '_models'
        prior_model_path = batch_models_path + '/batch_' \
                            + str(batch_id) + '_gmm_prior' 
        pickle.dump(self.gmm_prior_model, open(prior_model_path, 'wb'))
        print 'After refitting:'
        print self.gmm_prior_model.weights_
        print self.gmm_prior_model.means_
        elapsed_time = time.time() - start_time
        print 'Prior refit time: ', elapsed_time


    def active_model_update(self, batch_id):
        print 'Active model update for batch:', batch_id
        batch_models_path = self.active_models_path + 'batch_' \
                            + str(batch_id) + '_models'
        if not os.path.exists(batch_models_path):
            os.makedirs(batch_models_path)
        batches_num = batch_id + 1
        if batch_id != 0 and batches_num % self.active_update_batches == 0:
            # raw_input('Retrain clf')
            self.retrain_grasp_clf(batch_id)
        else:
            # raw_input('Inc train clf')
            self.incremental_train_grasp_clf(batch_id)
        if self.prior_name == 'MDN':
            if batch_id != 0 and batches_num % self.active_update_batches == 0:
                # raw_input('Retrain mdn')
                self.retrain_mdn_prior(batch_id)
                # raw_input('Retrain suc mdn')
                self.retrain_suc_mdn_prior(batch_id)
            else:
                # raw_input('Inc train mdn')
                self.incremental_train_mdn_prior(batch_id)
        elif self.prior_name == 'GMM':
            # self.refit_gmm_prior(batch_id)
            if batch_id != 0 and batches_num % self.active_update_batches == 0:
                # raw_input('Retrain gmm')
                self.refit_gmm_prior(batch_id)
                # raw_input('Retrain suc gmm')
                self.refit_suc_gmm_prior(batch_id)
        self.update_last_batch_id(batch_id)


if __name__ == '__main__':
    grasp_net_model_path = pkg_path + '/models/grasp_al_net/' + \
                       'grasp_net_freeze_enc_2_sets.ckpt'
    # grasp_net_model_path = pkg_path + '/models/grasp_al_net/' + \
    #                    'grasp_net_freeze_enc_2_sets_dropout_90.ckpt'
    prior_model_path = pkg_path + '/models/grasp_al_prior/prior_2_sets'
    al_file_path = '/mnt/tars_data/al_grasp_queries_sim/al_data.h5' 
    active_models_path = '/mnt/tars_data/multi_finger_sim_data/active_models/'
    gal = GraspActiveLearner(grasp_net_model_path, prior_model_path, 
                            al_file_path, active_models_path) 
    #gal.refit_gmm_prior(-1)
    # print gal.config_explore_sample()

   # test_data_path = '/mnt/tars_data/gazebo_al_grasps/test/' + \
   #                 'merged_grasp_data_6_16_and_6_18.h5'
   # data_file = h5py.File(test_data_path, 'r')
   # grasp_id = 0
   # grasp_config_obj_key = 'grasp_' + str(grasp_id) + '_config_obj'
   # grasp_full_config = data_file[grasp_config_obj_key][()] 
   # preshape_config_idx = list(xrange(8)) + [10, 11] + \
   #                        [14, 15] + [18, 19]
   # grasp_preshape_config = grasp_full_config[preshape_config_idx]
   # grasp_sparse_voxel_key = 'grasp_' + str(grasp_id) + '_sparse_voxel'
   # sparse_voxel_grid = data_file[grasp_sparse_voxel_key][()]
   # obj_dim_key = 'grasp_' + str(grasp_id) + '_dim_w_h_d'
   # obj_size = data_file[obj_dim_key][()]
   # grasp_label_key = 'grasp_' + str(grasp_id) + '_label'
   # grasp_label = data_file[grasp_label_key][()]
   # 
   # voxel_grid_full_dim = [32, 32, 32]
   # voxel_grid = np.zeros(tuple(voxel_grid_full_dim))
   # voxel_grid_index = sparse_voxel_grid.astype(int)
   # voxel_grid[voxel_grid_index[:, 0], voxel_grid_index[:, 1],
   #             voxel_grid_index[:, 2]] = 1
   # voxel_grid = np.expand_dims(voxel_grid, -1)

   # data_file.close()

   # # # inf_res = gal.grasp_strategies('grasp_suc', voxel_grid, obj_size) 
   # # # print 'inf_res:', inf_res
   # # # print 'grasp_preshape_config:', grasp_preshape_config
   # # # print 'True grasp label:', grasp_label

   # # # gal.exe_init_actions(voxel_grid, obj_size)
   # for i in xrange(10):
   #     gal.ucb(voxel_grid, obj_size)


