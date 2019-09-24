#!/usr/bin/env python
import rospy
import numpy as np
import pickle
import os
import time
import cv2
import matplotlib.pyplot as plt
import roslib.packages as rp
from geometry_msgs.msg import Pose, Quaternion, PoseStamped
from prob_grasp_planner.msg import VisualInfo, HandConfig
from prob_grasp_planner.srv import *
import tf
import h5py
from scipy.optimize import fmin_bfgs
from scipy.optimize import fmin_l_bfgs_b
from scipy.optimize import minimize

class GraspPgmInfer:
    def __init__(self, pgm_grasp_type=True):
        self.pgm_grasp_type = pgm_grasp_type

        self.palm_loc_dof_dim = 3
        self.palm_dof_dim = 6
        self.finger_joints_dof_dim = 8
        self.config_dim = self.palm_dof_dim + self.finger_joints_dof_dim 
        self.isrr_limit = False #True
        self.setup_config_limits()
        self.hand_config_frame_id = None

        pkg_path = rp.get_pkg_dir('prob_grasp_planner') 

        self.test_sim_gaus = True
        self.pca_model_path = pkg_path + '/models/grasp_type_planner/' + \
                              'train_models/pca/pca.model'
        self.train_model_path = pkg_path + '/models/grasp_type_planner' + \
                                '/train_models/classifiers/'
        self.prior_path = pkg_path + '/models/grasp_type_planner' + \
                          '/train_models/priors_all/'
        self.non_type_prior = False #True
        self.load_learned_models()
        
        #Use the computer tars or not
        self.tars = False
        if self.tars:
            self.grasp_config_log_save_path = '/media/kai/logs/multi_finger_sim_data/grad_des_ls_log/'
        else:
            self.grasp_config_log_save_path = '/dataspace/data_kai/logs/multi_finger_sim_data/grad_des_ls_log/'
        self.iter_total_num = 1000#500

        self.log_inf = True
        self.reg_log_prior = 1.
        #self.reg_log_prior = 0.5
        #self.reg_log_prior = 0.1
        #Regularization for log likelihood, this is mainly used to 
        #test the inference with only prior.
        self.reg_log_lkh = 1.

        self.voxel_grid_dim = [20, 20, 20]
        self.voxel_size = [0.01, 0.01, 0.01]

        self.al_strategy = None


    def load_learned_models(self):
        #Load PCA model
        self.pca_model = pickle.load(open(self.pca_model_path, 'rb'))

        #Load logistic regression models and priors.
        if self.non_type_prior:
            self.all_grasp_priors_model = pickle.load(open(self.prior_path + 'all_type_gmm.model', 'rb'))
            if self.pgm_grasp_type:
                self.power_grasp_clf_model = pickle.load(open(self.train_model_path + 'power_clf.model', 'rb'))
                self.prec_grasp_clf_model = pickle.load(open(self.train_model_path + 'prec_clf.model', 'rb'))
                self.power_grasp_priors_model = self.all_grasp_priors_model
                self.prec_grasp_priors_model = self.all_grasp_priors_model 
            else:
                self.all_grasp_clf_model = pickle.load(open(self.train_model_path + 'all_type_clf.model', 'rb'))
        else:
            if self.pgm_grasp_type:
                self.power_grasp_clf_model = pickle.load(open(self.train_model_path + 'power_clf.model', 'rb'))
                self.prec_grasp_clf_model = pickle.load(open(self.train_model_path + 'prec_clf.model', 'rb'))
                self.power_grasp_priors_model = pickle.load(open(self.prior_path + 'power_gmm.model', 'rb'))
                self.prec_grasp_priors_model = pickle.load(open(self.prior_path + 'prec_gmm.model', 'rb'))
                #print 'power weights:', self.power_grasp_priors_model.weights_
                #print 'power means:', self.power_grasp_priors_model.means_
                #print 'power covariances:', self.power_grasp_priors_model.covariances_
                #print 'prec weights:', self.prec_grasp_priors_model.weights_
                #print 'prec means:', self.prec_grasp_priors_model.means_
                #print 'prec covariances:', self.prec_grasp_priors_model.covariances_
            else:
                self.all_grasp_clf_model = pickle.load(open(self.train_model_path + 'all_type_clf.model', 'rb'))
                self.all_grasp_priors_model = pickle.load(open(self.prior_path + 'all_type_gmm.model', 'rb'))
                #print 'all weights:', self.all_grasp_priors_model.weights_
                #print 'all means:', self.all_grasp_priors_model.means_
                #print 'all covariances:', self.all_grasp_priors_model.covariances_

        #print 'covariances determinant:'
        #for i in xrange(len(self.power_grasp_priors_model.covariances_)):
        #    print 'power:'
        #    print self.power_grasp_priors_model.weights_[i]
        #    print np.linalg.det(self.power_grasp_priors_model.covariances_[i])
        #    print self.power_grasp_priors_model.means_[i]
        #    print 'precision:'
        #    print self.prec_grasp_priors_model.weights_[i]
        #    print np.linalg.det(self.prec_grasp_priors_model.covariances_[i])
        #    print self.prec_grasp_priors_model.means_[i]

    def setup_joint_angle_limits(self):
        '''
        Initializes a number of constants determing the joint limits for allegro
        TODO: Automate this by using a URDF file and allow hand to be specified at launch
        '''
        self.index_joint_0_lower = -0.59
        self.index_joint_0_upper = 0.57
        self.middle_joint_0_lower = -0.59
        self.middle_joint_0_upper = 0.57
        self.ring_joint_0_lower = -0.59
        self.ring_joint_0_upper = 0.57
        
        self.index_joint_1_lower = -0.296
        self.index_joint_1_upper = 0.71
        self.middle_joint_1_lower = -0.296
        self.middle_joint_1_upper = 0.71
        self.ring_joint_1_lower = -0.296
        self.ring_joint_1_upper = 0.71
        
        self.thumb_joint_0_lower = 0.363
        self.thumb_joint_0_upper = 1.55
        self.thumb_joint_1_lower = -0.205
        self.thumb_joint_1_upper = 1.263

        if not self.isrr_limit:
            self.index_joint_0_sample_lower = self.index_joint_0_lower 
            self.index_joint_0_sample_upper = self.index_joint_0_upper 
            self.middle_joint_0_sample_lower = self.middle_joint_0_lower
            self.middle_joint_0_sample_upper = self.middle_joint_0_upper
            self.ring_joint_0_sample_lower = self.ring_joint_0_lower 
            self.ring_joint_0_sample_upper = self.ring_joint_0_upper

            self.index_joint_1_sample_lower = self.index_joint_1_lower 
            self.index_joint_1_sample_upper = self.index_joint_1_upper 
            self.middle_joint_1_sample_lower = self.middle_joint_1_lower 
            self.middle_joint_1_sample_upper = self.middle_joint_1_upper
            self.ring_joint_1_sample_lower = self.ring_joint_1_lower 
            self.ring_joint_1_sample_upper = self.ring_joint_1_upper 

            self.thumb_joint_0_sample_lower = self.thumb_joint_0_lower 
            self.thumb_joint_0_sample_upper = self.thumb_joint_0_upper 
            self.thumb_joint_1_sample_lower = self.thumb_joint_1_lower 
            self.thumb_joint_1_sample_upper = self.thumb_joint_1_upper 

        else:
            #Set up joint limits for isrr paper
            self.index_joint_0_middle = (self.index_joint_0_lower + self.index_joint_0_upper) * 0.5
            self.middle_joint_0_middle = (self.middle_joint_0_lower + self.middle_joint_0_upper) * 0.5
            self.ring_joint_0_middle = (self.ring_joint_0_lower + self.ring_joint_0_upper) * 0.5
            self.index_joint_1_middle = (self.index_joint_1_lower + self.index_joint_1_upper) * 0.5
            self.middle_joint_1_middle = (self.middle_joint_1_lower + self.middle_joint_1_upper) * 0.5
            self.ring_joint_1_middle = (self.ring_joint_1_lower + self.ring_joint_1_upper) * 0.5
            self.thumb_joint_0_middle = (self.thumb_joint_0_lower + self.thumb_joint_0_upper) * 0.5
            self.thumb_joint_1_middle = (self.thumb_joint_1_lower + self.thumb_joint_1_upper) * 0.5

            self.index_joint_0_range = self.index_joint_0_upper - self.index_joint_0_lower
            self.middle_joint_0_range = self.middle_joint_0_upper - self.middle_joint_0_lower
            self.ring_joint_0_range = self.ring_joint_0_upper - self.ring_joint_0_lower
            self.index_joint_1_range = self.index_joint_1_upper - self.index_joint_1_lower
            self.middle_joint_1_range = self.middle_joint_1_upper - self.middle_joint_1_lower
            self.ring_joint_1_range = self.ring_joint_1_upper - self.ring_joint_1_lower
            self.thumb_joint_0_range = self.thumb_joint_0_upper - self.thumb_joint_0_lower
            self.thumb_joint_1_range = self.thumb_joint_1_upper - self.thumb_joint_1_lower

            self.first_joint_lower_limit = 0.5
            self.first_joint_upper_limit = 0.5
            self.second_joint_lower_limit = 0.5
            self.second_joint_upper_limit = 0.

            self.thumb_1st_joint_lower_limit = 0.
            self.thumb_1st_joint_upper_limit = 1.0
            self.thumb_2nd_joint_lower_limit = 0.5
            self.thumb_2nd_joint_upper_limit = 0.5

            self.index_joint_0_sample_lower = self.index_joint_0_middle - self.first_joint_lower_limit * self.index_joint_0_range
            self.index_joint_0_sample_upper = self.index_joint_0_middle + self.first_joint_upper_limit * self.index_joint_0_range
            self.middle_joint_0_sample_lower = self.middle_joint_0_middle - self.first_joint_lower_limit * self.middle_joint_0_range
            self.middle_joint_0_sample_upper = self.middle_joint_0_middle + self.first_joint_upper_limit * self.middle_joint_0_range
            self.ring_joint_0_sample_lower = self.ring_joint_0_middle - self.first_joint_lower_limit * self.ring_joint_0_range
            self.ring_joint_0_sample_upper = self.ring_joint_0_middle + self.first_joint_upper_limit * self.ring_joint_0_range

            self.index_joint_1_sample_lower = self.index_joint_1_middle - self.second_joint_lower_limit * self.index_joint_1_range
            self.index_joint_1_sample_upper = self.index_joint_1_middle + self.second_joint_upper_limit * self.index_joint_1_range
            self.middle_joint_1_sample_lower = self.middle_joint_1_middle - self.second_joint_lower_limit * self.middle_joint_1_range
            self.middle_joint_1_sample_upper = self.middle_joint_1_middle + self.second_joint_upper_limit * self.middle_joint_1_range
            self.ring_joint_1_sample_lower = self.ring_joint_1_middle - self.second_joint_lower_limit * self.ring_joint_1_range
            self.ring_joint_1_sample_upper = self.ring_joint_1_middle + self.second_joint_upper_limit * self.ring_joint_1_range

            self.thumb_joint_0_sample_lower = self.thumb_joint_0_middle - self.thumb_1st_joint_lower_limit * self.thumb_joint_0_range
            self.thumb_joint_0_sample_upper = self.thumb_joint_0_middle + self.thumb_1st_joint_upper_limit * self.thumb_joint_0_range
            self.thumb_joint_1_sample_lower = self.thumb_joint_1_middle - self.thumb_2nd_joint_lower_limit * self.thumb_joint_1_range
            self.thumb_joint_1_sample_upper = self.thumb_joint_1_middle + self.thumb_2nd_joint_upper_limit * self.thumb_joint_1_range


    def setup_config_limits(self):
        '''
        Set up the limits for grasp preshape configurations.
        '''
        self.preshape_config_lower_limit = np.zeros(self.config_dim)
        #self.preshape_config_lower_limit[:self.palm_dof_dim] = np.array([-2., -2., 0., -np.pi, -np.pi, -np.pi])
        self.preshape_config_lower_limit[:self.palm_dof_dim] = np.array([-1., -1., -2., -np.pi, -np.pi, -np.pi])

        self.preshape_config_upper_limit = np.zeros(self.config_dim)
        #two_pi = 2 * np.pi
        #self.preshape_config_upper_limit[:self.palm_dof_dim] = np.array([2., 2., 2., np.pi, np.pi, np.pi])
        self.preshape_config_upper_limit[:self.palm_dof_dim] = np.array([1., 1., 0.5, np.pi, np.pi, np.pi])

        self.setup_joint_angle_limits()

        self.preshape_config_lower_limit[self.palm_dof_dim] = self.index_joint_0_sample_lower
        self.preshape_config_lower_limit[self.palm_dof_dim + 1] = self.index_joint_1_sample_lower
        self.preshape_config_lower_limit[self.palm_dof_dim + 2] = self.middle_joint_0_sample_lower
        self.preshape_config_lower_limit[self.palm_dof_dim + 3] = self.middle_joint_1_sample_lower
        self.preshape_config_lower_limit[self.palm_dof_dim + 4] = self.ring_joint_0_sample_lower
        self.preshape_config_lower_limit[self.palm_dof_dim + 5] = self.ring_joint_1_sample_lower
        self.preshape_config_lower_limit[self.palm_dof_dim + 6] = self.thumb_joint_0_sample_lower
        self.preshape_config_lower_limit[self.palm_dof_dim + 7] = self.thumb_joint_1_sample_lower

        self.preshape_config_upper_limit[self.palm_dof_dim] = self.index_joint_0_sample_upper
        self.preshape_config_upper_limit[self.palm_dof_dim + 1] = self.index_joint_1_sample_upper
        self.preshape_config_upper_limit[self.palm_dof_dim + 2] = self.middle_joint_0_sample_upper
        self.preshape_config_upper_limit[self.palm_dof_dim + 3] = self.middle_joint_1_sample_upper
        self.preshape_config_upper_limit[self.palm_dof_dim + 4] = self.ring_joint_0_sample_upper
        self.preshape_config_upper_limit[self.palm_dof_dim + 5] = self.ring_joint_1_sample_upper
        self.preshape_config_upper_limit[self.palm_dof_dim + 6] = self.thumb_joint_0_sample_upper
        self.preshape_config_upper_limit[self.palm_dof_dim + 7] = self.thumb_joint_1_sample_upper

    def setup_palm_pose_limits(self, config_init):
        '''
        Set up the limits for the palm pose of grasp preshape configurations
        from the intialization.
        '''
        #pos_range = 0.05
        #ort_range = 0.05 * np.pi
        pos_range = 0.5
        ort_range = 0.5 * np.pi
        #pos_range = float('inf')
        #ort_range = float('inf')
        lower_limit_range = -np.array([pos_range, pos_range, pos_range, ort_range, ort_range, ort_range])
        upper_limit_range = np.array([pos_range, pos_range, pos_range, ort_range, ort_range, ort_range])
        self.preshape_config_lower_limit[:self.palm_dof_dim] = config_init[:self.palm_dof_dim] + lower_limit_range
        self.preshape_config_upper_limit[:self.palm_dof_dim] = config_init[:self.palm_dof_dim] + upper_limit_range

    def project_config_all(self, q):
        '''
        Project the preshape configuration into the valid range.
        '''
        q_proj = np.copy(q)
        two_pi = 2 * np.pi
        for i in xrange(self.palm_loc_dof_dim):
            if q_proj[i] < self.preshape_config_lower_limit[i]:
                q_proj[i] = self.preshape_config_lower_limit[i]
            if q_proj[i] > self.preshape_config_upper_limit[i]:
                q_proj[i] = self.preshape_config_upper_limit[i]

        q_proj[self.palm_loc_dof_dim:] %= two_pi
        for i in range(self.palm_loc_dof_dim, self.config_dim):
            if q_proj[i] > np.pi:
                q_proj[i] -= two_pi
            if q_proj[i] < self.preshape_config_lower_limit[i]:
                q_proj[i] = self.preshape_config_lower_limit[i]
            if q_proj[i] > self.preshape_config_upper_limit[i]:
                q_proj[i] = self.preshape_config_upper_limit[i]

        return q_proj

    def project_config(self, q):
        '''
        Project the preshape hand joint angles into the valid range.
        '''
        q_proj = np.copy(q)
        two_pi = 2 * np.pi
        q_proj[self.palm_dof_dim:] %= two_pi
        for i in range(self.palm_dof_dim, self.config_dim):
            if q_proj[i] > np.pi:
                q_proj[i] -= two_pi
            if q_proj[i] < self.preshape_config_lower_limit[i]:
                q_proj[i] = self.preshape_config_lower_limit[i]
            if q_proj[i] > self.preshape_config_upper_limit[i]:
                q_proj[i] = self.preshape_config_upper_limit[i]

        return q_proj


    def convert_preshape_to_full_config(self, preshape_config):
        '''
        Convert preshape grasp configuration to full grasp configuration by filling zeros for 
        uninferred finger joints.
        '''
        hand_config = HandConfig()
        hand_config.palm_pose.header.frame_id = self.hand_config_frame_id
        hand_config.palm_pose.pose.position.x, hand_config.palm_pose.pose.position.y, \
                hand_config.palm_pose.pose.position.z = preshape_config[:self.palm_loc_dof_dim]    
    
        palm_euler = preshape_config[self.palm_loc_dof_dim:self.palm_dof_dim] 
        palm_quaternion = tf.transformations.quaternion_from_euler(palm_euler[0], palm_euler[1], palm_euler[2])
        #hand_config.palm_pose.pose.orientation = palm_quaternion
        hand_config.palm_pose.pose.orientation.x, hand_config.palm_pose.pose.orientation.y, \
                hand_config.palm_pose.pose.orientation.z, hand_config.palm_pose.pose.orientation.w = palm_quaternion 

        hand_config.hand_joint_state.name = ['index_joint_0','index_joint_1','index_joint_2', 'index_joint_3',
                   'middle_joint_0','middle_joint_1','middle_joint_2', 'middle_joint_3',
                   'ring_joint_0','ring_joint_1','ring_joint_2', 'ring_joint_3',
                   'thumb_joint_0','thumb_joint_1','thumb_joint_2', 'thumb_joint_3']
        hand_config.hand_joint_state.position = [preshape_config[self.palm_dof_dim], preshape_config[self.palm_dof_dim + 1], 0., 0.,
                                                preshape_config[self.palm_dof_dim + 2], preshape_config[self.palm_dof_dim + 3], 0., 0.,
                                                preshape_config[self.palm_dof_dim + 4], preshape_config[self.palm_dof_dim + 5], 0., 0.,
                                                preshape_config[self.palm_dof_dim + 6], preshape_config[self.palm_dof_dim + 7], 0., 0.]

        return hand_config

    def convert_full_to_preshape_config(self, hand_config):
        '''
        Convert full grasp configuration to preshape grasp configuration by deleting uninferred joint
        angles.
        '''
        palm_quaternion = (hand_config.palm_pose.pose.orientation.x, hand_config.palm_pose.pose.orientation.y,
                hand_config.palm_pose.pose.orientation.z, hand_config.palm_pose.pose.orientation.w) 
        palm_euler = tf.transformations.euler_from_quaternion(palm_quaternion)

        preshape_config = [hand_config.palm_pose.pose.position.x, hand_config.palm_pose.pose.position.y,
                hand_config.palm_pose.pose.position.z, palm_euler[0], palm_euler[1], palm_euler[2],
                hand_config.hand_joint_state.position[0], hand_config.hand_joint_state.position[1],
                hand_config.hand_joint_state.position[4], hand_config.hand_joint_state.position[5],
                hand_config.hand_joint_state.position[8], hand_config.hand_joint_state.position[9],
                hand_config.hand_joint_state.position[12], hand_config.hand_joint_state.position[13]]

        return np.array(preshape_config)
    
    def get_logistic_clf(self, grasp_type):
        logistic = None
        if grasp_type == 'all':
            logistic = self.all_grasp_clf_model
        elif grasp_type == 'power':
            logistic = self.power_grasp_clf_model
        elif grasp_type == 'prec':
            logistic = self.prec_grasp_clf_model
        else:
            print 'Wrong grasp type to get logistic regression model!'
        return logistic

    def compute_clf_suc_prob(self, grasp_type, latent_voxel, grasp_config):
        '''
        Compute the classifier (i.e. logistic regression) success probability.
        '''
        logistic = self.get_logistic_clf(grasp_type)
        grasp = np.concatenate((latent_voxel, grasp_config))
        clf_suc_prob = logistic.predict_proba([grasp])[0, 1]
        return clf_suc_prob

    def compute_clf_log_suc_prob(self, grasp_type, latent_voxel, grasp_config):
        '''
        Compute the classifier (i.e. logistic regression) success probability.
        '''
        log_suc_prob = np.log(self.compute_clf_suc_prob(grasp_type, latent_voxel, grasp_config))
        log_suc_prob *= self.reg_log_lkh
        return log_suc_prob 

    def compute_clf_suc_prob_grad(self, grasp_type, latent_voxel, grasp_config):
        '''
        Compute the classifier (i.e. logistic regression) success probability 
        gradient with respect to grasp configuration.
        '''
        logistic = self.get_logistic_clf(grasp_type)
        grasp = np.concatenate((latent_voxel, grasp_config))
        clf_suc_prob = logistic.predict_proba([grasp])[0, 1]
        clf_suc_prob_grad = clf_suc_prob * (1 - clf_suc_prob) * logistic.coef_[0, -len(grasp_config):] 
        return clf_suc_prob_grad #, clf_suc_prob

    def compute_clf_log_suc_prob_grad(self, grasp_type, latent_voxel, grasp_config):
        '''
        Compute the classifier (i.e. logistic regression) log success probability 
        gradient with respect to grasp configuration.
        '''
        logistic = self.get_logistic_clf(grasp_type)
        grasp = np.concatenate((latent_voxel, grasp_config))
        clf_suc_prob = logistic.predict_proba([grasp])[0, 1]
        clf_log_prob_grad = (1 - clf_suc_prob) * logistic.coef_[0, -len(grasp_config):] 
        clf_log_prob_grad *= self.reg_log_lkh
        return clf_log_prob_grad #, clf_suc_prob

    def get_prior_model(self, grasp_type):
        prior_model = None
        if grasp_type == 'all':
            prior_model = self.all_grasp_priors_model
        elif grasp_type == 'power':
            prior_model = self.power_grasp_priors_model
        elif grasp_type == 'prec':
            prior_model = self.prec_grasp_priors_model
        else:
            print 'Wrong grasp type to get grasp priors!'
        return prior_model

    def compute_grasp_prior(self, grasp_type, grasp_config):
        '''
        Compute the grasp configuration prior.
        '''
        prior_model = self.get_prior_model(grasp_type)
        log_prior = prior_model.score_samples([grasp_config])[0]
        prior = np.exp(log_prior)
        return prior

    def compute_grasp_log_prior(self, grasp_type, grasp_config):
        '''
        Compute the grasp configuration prior.
        '''
        prior_model = self.get_prior_model(grasp_type)
        #print prior_model
        #print prior_model.weights_
        #print prior_model.means_
        #print 'grasp_config:', grasp_config
        log_prior = prior_model.score_samples([grasp_config])[0]
        #print 'log_prior:', log_prior
        #print 'reg_log_prior:', self.reg_log_prior
        log_prior *= self.reg_log_prior
        return log_prior

    def compute_grasp_prior_grad(self, grasp_type, grasp_config):
        '''
        Compute the grasp configuration prior gradient with respect
        to grasp configuration.
        '''
        prior_model = self.get_prior_model(grasp_type)
        weighted_log_prob = prior_model._estimate_weighted_log_prob(np.array([grasp_config]))[0]
        weighted_prob = np.exp(weighted_log_prob)
        #weighted_prob can also be computed by: 
        #multivariate_normal(mean=g.means_[i], cov=g.covariances_[i], allow_singular=True)
        grad = np.zeros(len(grasp_config))
        for i, w in enumerate(prior_model.weights_):
            grad += -weighted_prob[i] * np.matmul(np.linalg.inv(prior_model.covariances_[i]), \
                    (grasp_config - prior_model.means_[i]))
        grad *= self.reg_log_prior
        return grad

    def compute_grasp_log_prior_grad(self, grasp_type, grasp_config):
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
        prior_model = self.get_prior_model(grasp_type)
        #_estimate_weighted_log_prob dimension: (n_samples, n_components)
        weighted_log_prob = prior_model._estimate_weighted_log_prob(np.array([grasp_config]))[0]
        max_wlp = np.max(weighted_log_prob)
        #Dim: n_components
        wlp_minus_max = weighted_log_prob - max_wlp

        #Dim: n_config
        p_x_prime = np.zeros(len(grasp_config))
        for i in xrange(prior_model.weights_.shape[0]):
            #log of inverse of covariance matrix multiply distance (x - mean)
            #Dim: n_config 
            inv_sigma_dist = np.matmul(np.linalg.inv(prior_model.covariances_[i]), \
                                (grasp_config - prior_model.means_[i]))
            p_x_prime += -np.exp(wlp_minus_max[i]) * inv_sigma_dist

        prior = np.sum(np.exp(wlp_minus_max))
        grad = p_x_prime / prior 

        grad *= self.reg_log_prior
        return grad

    
    def compute_d_prob_d_config(self, grasp_type, latent_voxel, grasp_config):
        '''
        Compute gradients d(p(y=1 | theta, o', g=1, w) * p(theta | g=0)) / d theta.
        '''
        #print self.compute_grasp_prior_grad(grasp_type, grasp_config).shape
        #print self.compute_clf_suc_prob(grasp_type, latent_voxel, grasp_config)
        #print self.compute_grasp_prior(grasp_type, grasp_config)
        #print self.compute_clf_suc_prob_grad(grasp_type, latent_voxel, grasp_config).shape
        
        prior = self.compute_grasp_prior(grasp_type, grasp_config)
        prior_grad = self.compute_grasp_prior_grad(grasp_type, grasp_config)
        clf_suc_prob = self.compute_clf_suc_prob(grasp_type, latent_voxel, grasp_config)
        clf_suc_prob_grad = self.compute_clf_suc_prob_grad(grasp_type, latent_voxel, grasp_config)
        d_prob_d_config = prior_grad * clf_suc_prob + prior * clf_suc_prob_grad

        obj_prob = self.compute_clf_suc_prob(grasp_type, latent_voxel, grasp_config) * \
                    self.compute_grasp_prior(grasp_type, grasp_config)

        #Gradient checking
        grad_check = False
        if grad_check:
            num_grad = self.compute_num_grad(self.compute_grasp_prior, grasp_type, None, grasp_config)
            grad_diff = prior_grad - num_grad
            print 'prior_grad:', prior_grad
            print 'num_grad:', num_grad
            print 'Gradient difference:', grad_diff
            print 'Gradient difference abs mean:', np.mean(abs(grad_diff))
            print 'Gradient difference abs mean percentage:',  np.mean(abs(grad_diff)) / np.mean(abs(prior_grad))
            print '-----------------------------'

            num_grad = self.compute_num_grad(self.compute_clf_suc_prob, grasp_type, latent_voxel, grasp_config)
            grad_diff = clf_suc_prob_grad - num_grad
            print 'clf_suc_prob_grad:', clf_suc_prob_grad
            print 'num_grad:', num_grad
            print 'Gradient difference:', grad_diff
            print 'Gradient difference abs mean:', np.mean(abs(grad_diff))
            print 'Gradient difference abs mean percentage:',  np.mean(abs(grad_diff)) / np.mean(abs(clf_suc_prob_grad))
            print '+++++++++++++++++++++++++++++'

            num_grad = self.compute_num_grad(self.compute_obj_prob, grasp_type, latent_voxel, grasp_config)
            grad_diff = d_prob_d_config - num_grad
            print 'd_prob_d_config:', d_prob_d_config
            print 'num_grad:', num_grad
            print 'Gradient difference:', grad_diff
            print 'Gradient difference abs mean:', np.mean(abs(grad_diff))
            print 'Gradient difference abs mean percentage:',  np.mean(abs(grad_diff)) / np.mean(abs(d_prob_d_config))

            print '~~~~~~~~~~~~~~~~~~~~~~~~~~~'
            print 'prior:', prior
            print 'clf_suc_prob:', clf_suc_prob
            print 'obj_prob:', obj_prob
            print '################################################################'

        return d_prob_d_config, obj_prob

    def compute_num_grad(self, func, grasp_type, latent_voxel, grasp_config):
        '''
        Compute numerical gradients d(p(y=1 | theta, o', g=1, w) * p(theta | g=0)) / d theta.
        '''
        eps = 10**-4
        grad = np.zeros(len(grasp_config))
        for i in xrange(len(grasp_config)):
            grasp_config_plus = np.copy(grasp_config)
            grasp_config_plus[i] += eps
            if latent_voxel is None:
                obj_prob_plus = func(grasp_type, grasp_config_plus)
            else:
                obj_prob_plus = func(grasp_type, latent_voxel, grasp_config_plus)
            grasp_config_minus = np.copy(grasp_config)
            grasp_config_minus[i] -= eps
            if latent_voxel is None:
                obj_prob_minus = func(grasp_type, grasp_config_minus)
            else:
                obj_prob_minus = func(grasp_type, latent_voxel, grasp_config_minus)
            #print 'grasp_config_plus:', grasp_config_plus
            #print 'grasp_config_minus:', grasp_config_minus
            #print 'obj_prob_plus:', obj_prob_plus
            #print 'obj_prob_minus:', obj_prob_minus
            ith_grad = (obj_prob_plus - obj_prob_minus) / (2. * eps)
            grad[i] = ith_grad
        return grad

    def compute_obj_prob(self, grasp_type, latent_voxel, grasp_config):
        '''
        Compute probability of objective function: p(y=2 | theta, o', g=1, w) * p(theta | g=0). 
        '''
        obj_prob = self.compute_clf_suc_prob(grasp_type, latent_voxel, grasp_config) * \
                    self.compute_grasp_prior(grasp_type, grasp_config)
        return obj_prob

    def compute_log_d_prob_d_config(self, grasp_type, latent_voxel, grasp_config):
        '''
        Compute gradients d(p(y=1 | theta, o', g=1, w) * p(theta | g=0)) / d theta.
        '''
        #prior = self.compute_grasp_prior(grasp_type, grasp_config)
        #prior_grad = self.compute_grasp_prior_grad(grasp_type, grasp_config)
        #log_prior_grad = prior_grad / prior

        log_prior_grad = self.compute_grasp_log_prior_grad(grasp_type, grasp_config)

        #clf_suc_prob = self.compute_clf_suc_prob(grasp_type, latent_voxel, grasp_config)
        #clf_suc_prob_grad = self.compute_clf_suc_prob_grad(grasp_type, latent_voxel, grasp_config)
        #log_clf_prob_grad = clf_suc_prob_grad / clf_suc_prob
        
        clf_log_prob_grad = self.compute_clf_log_suc_prob_grad(grasp_type, latent_voxel, grasp_config)

        d_prob_d_config = log_prior_grad + clf_log_prob_grad

        #obj_prob = self.compute_log_obj_prob(grasp_type, latent_voxel, grasp_config)

        clf_log_prob = self.compute_clf_log_suc_prob(grasp_type, latent_voxel, grasp_config)
        log_prior = self.compute_grasp_log_prior(grasp_type, grasp_config)
        #print 'clf_log_prob:', clf_log_prob
        #print 'log_prior:', log_prior
        obj_prob = clf_log_prob + log_prior 
        #obj_prob = log_prior 
        #obj_prob = clf_log_prob


        #Gradient checking
        grad_check = False #True 
        if grad_check:
            num_grad = self.compute_num_grad(self.compute_grasp_log_prior, grasp_type, None, grasp_config)
            grad_diff = log_prior_grad - num_grad
            print 'log_prior_grad:', log_prior_grad
            print 'num_grad:', num_grad
            print 'Gradient difference:', grad_diff
            print 'Gradient difference abs mean:', np.mean(abs(grad_diff))
            print 'Gradient difference abs mean percentage:',  np.mean(abs(grad_diff)) / np.mean(abs(log_prior_grad))
            print '-----------------------------'

            num_grad = self.compute_num_grad(self.compute_clf_log_suc_prob, grasp_type, latent_voxel, grasp_config)
            grad_diff = clf_log_prob_grad - num_grad
            print 'clf_log_prob_grad:', clf_log_prob_grad
            print 'num_grad:', num_grad
            print 'Gradient difference:', grad_diff
            print 'Gradient difference abs mean:', np.mean(abs(grad_diff))
            print 'Gradient difference abs mean percentage:',  np.mean(abs(grad_diff)) / np.mean(abs(clf_log_prob_grad))
            print '+++++++++++++++++++++++++++++'

            num_grad = self.compute_num_grad(self.compute_log_obj_prob, grasp_type, latent_voxel, grasp_config)
            grad_diff = d_prob_d_config - num_grad
            print 'd_prob_d_config:', d_prob_d_config
            print 'num_grad:', num_grad
            print 'Gradient difference:', grad_diff
            print 'Gradient difference abs mean:', np.mean(abs(grad_diff))
            print 'Gradient difference abs mean percentage:',  np.mean(abs(grad_diff)) / np.mean(abs(d_prob_d_config))
            print '~~~~~~~~~~~~~~~~~~~~~~~~~~~'
            #print 'prior:', prior
            #print 'clf_suc_prob:', clf_suc_prob
            print 'obj_prob:', obj_prob
            print '################################################################'

        return d_prob_d_config, obj_prob, clf_log_prob, log_prior

    def compute_log_obj_prob(self, grasp_type, latent_voxel, grasp_config):
        '''
        Compute probability of objective function: p(y=2 | theta, o', g=1, w) * p(theta | g=0). 
        '''
        clf_log_prob = self.compute_clf_log_suc_prob(grasp_type, latent_voxel, grasp_config)
        log_prior = self.compute_grasp_log_prior(grasp_type, grasp_config)
        #print 'grasp_config:', grasp_config
        #print 'clf_log_prob:', clf_log_prob
        #print 'log_prior:', log_prior
        obj_prob = clf_log_prob + log_prior 
        #print 'obj_prob:', obj_prob
        #obj_prob = log_prior 
        #obj_prob = clf_log_prob
        return obj_prob

    def find_learning_rate_bt(self, alpha, grasp_type, latent_voxel, q, suc_prob, grad_q, 
                                line_search_log=None, use_talor=False):
        '''
        Backtracking line search to find the learning rate.
        '''
        t = time.time()
        iter_num = -1
        #alpha = 0.001
        tao = 0.5
        beta = 0.001#0.1
        l = 0
        iter_limit = 100
        q_new = q + alpha * grad_q
        if line_search_log is not None:
            line_search_log.writelines('q_new: ' + str(q_new))
            line_search_log.writelines('\n')
        q_new = self.project_config(q_new)
        if line_search_log is not None:
            line_search_log.writelines('q_new after projection: ' + str(q_new))
            line_search_log.writelines('\n')
        if self.log_inf:
            suc_prob_new = self.compute_log_obj_prob(grasp_type, latent_voxel, q_new)
        else:
            suc_prob_new = self.compute_obj_prob(grasp_type, latent_voxel, q_new)
        talor_1st_order = beta * alpha * np.inner(grad_q, grad_q)
        #Double check the mean is the right thing to do or not?
        talor_1st_order = np.mean(talor_1st_order)
        if line_search_log is not None:
            line_search_log.writelines('use_talor: ' + str(use_talor))
            line_search_log.writelines('\n')
        #print suc_prob_new, suc_prob, talor_1st_order
        #print type(suc_prob_new), type(suc_prob), type(use_talor), type(talor_1st_order)
        while suc_prob_new <= suc_prob + use_talor * talor_1st_order:
        #while suc_prob_new <= suc_prob:
            if line_search_log is not None:
                line_search_log.writelines('l: ' + str(l))
                line_search_log.writelines('\n')
                line_search_log.writelines('suc_prob_new: ' + str(suc_prob_new))
                line_search_log.writelines('\n')
                line_search_log.writelines('suc_prob: ' + str(suc_prob))
                line_search_log.writelines('\n')
                line_search_log.writelines('talor_1st_order: ' + str(talor_1st_order))
                line_search_log.writelines('\n')
                line_search_log.writelines('alpha: ' + str(alpha))
                line_search_log.writelines('\n')
            alpha *= tao
            q_new = q + alpha * grad_q
            if line_search_log is not None:
                line_search_log.writelines('q_new: ' + str(q_new))
                line_search_log.writelines('\n')
            q_new = self.project_config(q_new)
            if line_search_log is not None:
                line_search_log.writelines('q_new after projection: ' + str(q_new))
                line_search_log.writelines('\n')
            if self.log_inf:
                suc_prob_new = self.compute_log_obj_prob(grasp_type, latent_voxel, q_new)
            else:
                suc_prob_new = self.compute_obj_prob(grasp_type, latent_voxel, q_new)
            talor_1st_order = beta * alpha * np.inner(grad_q, grad_q)
            if l > iter_limit:
                if line_search_log is not None:
                    line_search_log.writelines('********* Can not find alpha in ' + str(iter_limit) + ' iters')
                    line_search_log.writelines('\n')
                alpha = 0.
                break
            l += 1
        if line_search_log is not None:
            line_search_log.writelines('Line search time: ' + str(time.time() - t))
            line_search_log.writelines('\n')
        #print (suc_prob_new > suc_prob), alpha
        return alpha

    def gd_inf_one_type(self, grasp_type, object_voxel, init_hand_config, 
                            save_grad_to_log=False, object_id=None, grasp_id=None):
        '''
        Gradient descent inference with line search for one grasp type. 
        '''
        #self.hand_config_frame_id = init_hand_config.palm_pose.header.frame_id
        #q = self.convert_full_to_preshape_config(init_hand_config)
        q = init_hand_config
        #self.hand_config_frame_id = 'grasp_object'
        #print 'init_config:', q

        #self.setup_palm_pose_limits(q)

        voxel_grid_dim = object_voxel.shape
        voxel_num = voxel_grid_dim[0] * voxel_grid_dim[1] * voxel_grid_dim[2]  
        grasp_1d_voxel = np.reshape(object_voxel, voxel_num)
        latent_voxel = self.pca_model.transform([grasp_1d_voxel])[0]

        if save_grad_to_log: 
            config_grad_path = self.grasp_config_log_save_path + 'object_' + str(object_id) \
                                    + '_grasp_' + str(grasp_id) + '_' + grasp_type + '/'
            if not os.path.exists(config_grad_path):
                os.makedirs(config_grad_path)
            log_file_path = config_grad_path + 'gradient_descent_log'
            log_file = open(log_file_path, 'w')
            line_search_log_file_path = config_grad_path + 'line_search_log'
            line_search_log = open(line_search_log_file_path, 'w')
        else:
            line_search_log = None
        
        t = time.time()
        
        suc_probs = []
        if self.log_inf:
            clf_log_probs = []
            log_priors = []
        #iter_total_num = 100
        delta = 10**-8
        #delta = 10**-3
        use_talor = True
        #if grasp_id % 2 == 1:
        #    use_talor = 1.
        
        #save_grad = False
        #if object_id % 10 != 0:
        #    save_grad = False

        q_learn_rate = 0.0001
        bt_rate_scale = 2. #1.2
        for iter_num in xrange(self.iter_total_num):
            #print 'iter:', iter_num
            if self.log_inf:
                #grad_q, suc_prob = self.compute_log_d_prob_d_config(grasp_type, latent_voxel, q)
                grad_q, suc_prob, clf_log_prob, log_prior = self.compute_log_d_prob_d_config(grasp_type, latent_voxel, q)
                clf_log_probs.append(clf_log_prob)
                log_priors.append(log_prior)
            else:
                grad_q, suc_prob = self.compute_d_prob_d_config(grasp_type, latent_voxel, q)
            suc_probs.append(suc_prob)
            grad_norm = np.linalg.norm(grad_q)
            if save_grad_to_log: 
                log_file.writelines('iter: ' + str(iter_num))
                log_file.writelines('\n')
                log_file.writelines('q: ' + str(q))
                log_file.writelines('\n')
                log_file.writelines('grad_q: ' + str(grad_q))
                log_file.writelines('\n')
                log_file.writelines('norm(grad_q): ' + str(grad_norm))
                log_file.writelines('\n')
                log_file.writelines('suc_prob: ' + str(suc_prob))
                log_file.writelines('\n')
                log_file.writelines('clf_log_prob: ' + str(clf_log_prob))
                log_file.writelines('\n')
                log_file.writelines('log_prior: ' + str(log_prior))
                log_file.writelines('\n')
            #Stop if gradient is too small
            if grad_norm < delta:
                if save_grad_to_log: 
                    log_file.writelines('Gradient too small, stop iteration!\n')
                break
           
            if save_grad_to_log: 
                line_search_log.writelines('iter: ' + str(iter_num))
                line_search_log.writelines('\n')
            #Scale the previous backtracking line search learning rate and use it 
            #as the initial learning rate of the current backtracking line search.
            bt_learn_rate = bt_rate_scale * q_learn_rate
            q_learn_rate = self.find_learning_rate_bt(bt_learn_rate, grasp_type, latent_voxel, q, suc_prob, 
                                                        grad_q, line_search_log, use_talor)
            if save_grad_to_log: 
                line_search_log.writelines('######################################################')
                line_search_log.writelines('\n')
            if save_grad_to_log: 
                log_file.writelines('q_learn_rate: ' + str(q_learn_rate))
                log_file.writelines('\n')
            if q_learn_rate == 0.:
                if save_grad_to_log: 
                    log_file.writelines('Alpha is zero, stop iteration.')
                    log_file.writelines('\n')
                break
            q_update = q_learn_rate * grad_q
            q_update = q + q_update
            if save_grad_to_log: 
                log_file.writelines('q: ' + str(q_update))
                log_file.writelines('\n')
            q_update = self.project_config(q_update)
            if save_grad_to_log: 
                log_file.writelines('q after projection: ' + str(q_update))
                log_file.writelines('\n')
            q_update_proj = q_update - q
            if np.linalg.norm(q_update_proj) < delta:
                if save_grad_to_log: 
                    log_file.writelines('q_update_proj too small, stop iteration.')
                    log_file.writelines('\n')
                break
            q = q_update
        
        suc_probs = np.array(suc_probs)
        if save_grad_to_log: 
            plt.figure()
            plt.plot(suc_probs, label='suc')
            if self.log_inf:
                plt.plot(np.array(clf_log_probs), label='clf')
                plt.plot(np.array(log_priors), label='prior')
                print 'suc_probs:', suc_probs
                print 'clf_log_probs:', clf_log_probs
                print 'log_priors:', log_priors
            plt.ylabel('Suc Probalities')
            plt.xlabel('Iteration')
            plt.legend(loc="lower right")
            plt.savefig(config_grad_path + 'suc_prob.png')
            plt.cla()
            plt.clf()
            plt.close()
        
        elapased_time = time.time() - t
        if save_grad_to_log: 
            log_file.writelines('Total inference time: ' + str(elapased_time))
            log_file.writelines('\n')
            log_file.close()
            line_search_log.close()
        #else:
        #    print 'Total inference time: ', str(elapased_time)
        print 'Total inference time: ', str(elapased_time)
        print 'iter_num:', iter_num

        full_grasp_config = self.convert_preshape_to_full_config(q)

        print init_hand_config 
        print full_grasp_config, suc_probs[-1], suc_probs[0]

        return full_grasp_config, suc_probs[-1], suc_probs[0]
        
        #return q, suc_probs[-1], suc_probs[0]

    def gradient_descent_inf(self, object_voxel, init_hand_config, 
                            save_grad_to_log=False, object_id=None, grasp_id=None):
    #def gradient_descent_inf(self, object_voxel, grasp_config, infer_grasp_type=False,
    #                        save_grad_to_log=False, object_id=None, grasp_id=None):
        self.hand_config_frame_id = init_hand_config.palm_pose.header.frame_id
        grasp_config = self.convert_full_to_preshape_config(init_hand_config)
        if infer_grasp_type:
            power_inf_result = self.gd_inf_one_type('power', object_voxel, grasp_config, 
                                                save_grad_to_log, object_id, grasp_id)
            prec_inf_result = self.gd_inf_one_type('prec', object_voxel, grasp_config, 
                                                save_grad_to_log, object_id, grasp_id)
            if power_inf_result[1] >= prec_inf_result[1]:
                inf_result = power_inf_result
            else:
                inf_result = prec_inf_result
        else:
            inf_result = self.gd_inf_one_type('all', object_voxel, grasp_config, 
                                                save_grad_to_log, object_id, grasp_id)
        return inf_result


    def grasp_obj_bfgs(self, grasp_config, grasp_type, latent_voxel):
        '''
        Objective function for lbfgs/bfgs optimizer.
        '''
        #print grasp_type
        #print latent_voxel
        #print grasp_config
        obj_prob = self.compute_log_obj_prob(grasp_type, latent_voxel, grasp_config)
        print -obj_prob
        #raw_input('wait')
        return -obj_prob
 

    def grasp_grad_bfgs(self, grasp_config, grasp_type, latent_voxel):
        '''
        Derivative function for lbfgs/bfgs optimizer.
 
        '''
        grad_q, _, _, _ = self.compute_log_d_prob_d_config(grasp_type, latent_voxel, grasp_config)
        return -grad_q
 

    def grasp_clf_obj_bfgs(self, grasp_config, grasp_type, latent_voxel):
        '''
        Objective function for lbfgs/bfgs optimizer.
        '''
        obj_prob = self.compute_clf_suc_prob(grasp_type, latent_voxel, grasp_config)
        return -obj_prob
 

    def grasp_clf_grad_bfgs(self, grasp_config, grasp_type, latent_voxel):
        '''
        Derivative function for lbfgs/bfgs optimizer.
 
        '''
        suc_prob_grad = self.compute_clf_suc_prob_grad(
                                 grasp_type, latent_voxel, grasp_config)
        return -suc_prob_grad


    def quasi_newton_lbfgs_inf(self, grasp_type, object_voxel, init_hand_config, 
                               init_frame_id=None, bfgs=True):
        '''
        Quasi Newton inference with bfgs/lbfgs update. 
        '''
        t = time.time()
        #self.hand_config_frame_id = init_hand_config.palm_pose.header.frame_id
        #q_init = self.convert_full_to_preshape_config(init_hand_config)

        self.hand_config_frame_id = init_frame_id
        q_init = init_hand_config

        voxel_grid_dim = object_voxel.shape
        voxel_num = voxel_grid_dim[0] * voxel_grid_dim[1] * voxel_grid_dim[2]  
        grasp_1d_voxel = np.reshape(object_voxel, voxel_num)
        latent_voxel = self.pca_model.transform([grasp_1d_voxel])[0]

        #self.setup_palm_pose_limits(q_init)
        init_suc_prob = self.grasp_obj_bfgs(q_init, grasp_type, latent_voxel)
        print 'init_suc_prob:', init_suc_prob

        opt_method = 'L-BFGS-B'
        if bfgs:
            opt_method = 'BFGS'

        #opt_method = 'TNC'

        bnds = []
        #for i in range(self.config_dim):
        #    bnds.append((self.preshape_config_lower_limit[i], self.preshape_config_upper_limit[i]))

        #for i in range(self.config_dim):
        #    bnds.append((-float('inf'), float('inf')))

        for i in xrange(self.palm_dof_dim):
            bnds.append((-float('inf'), float('inf')))
        for i in xrange(self.palm_dof_dim, self.config_dim):
            bnds.append((self.preshape_config_lower_limit[i], self.preshape_config_upper_limit[i]))

        #Notice this is for gradient descent, not ascent.
        opt_res = minimize(self.grasp_obj_bfgs, q_init, jac=self.grasp_grad_bfgs, 
                                        args=(grasp_type, latent_voxel,), method=opt_method, bounds=bnds)
        print 'opt_res with constraints:', opt_res

        #opt_res = minimize(self.grasp_obj_bfgs, q_init, jac=self.grasp_grad_bfgs, 
        #                                args=(grasp_type, latent_voxel,), method=opt_method)

        #print 'opt_res without constraints:', opt_res
        #opt_config = opt_res.x
        #proj_res = self.project_config(opt_config)
        #res_valid = np.allclose(opt_config, proj_res)
        #
        #if (not res_valid):
        #    opt_res = minimize(self.grasp_obj_bfgs, opt_config, jac=self.grasp_grad_bfgs, 
        #                                    args=(grasp_type, latent_voxel,), method=opt_method, bounds=bnds)

        #    print 'opt_res with constraints:', opt_res

        #    opt_config = opt_res.x
        #    proj_res = self.project_config(opt_config)
        #    res_valid = np.allclose(opt_config, proj_res)
        #    print 'opt_res is valid with constraints:', res_valid 

        full_grasp_config = self.convert_preshape_to_full_config(opt_res.x)
        #Lift up the grasp to avoid collision
        #full_grasp_config.palm_pose.pose.position.x -= 0.3
        
        init_suc_prob = self.grasp_obj_bfgs(q_init, grasp_type, latent_voxel)

        elapased_time = time.time() - t
        print 'Total inference time: ', str(elapased_time)

        #print full_grasp_config, -opt_res.fun, -init_suc_prob 

        return full_grasp_config, -opt_res.fun, -init_suc_prob 

    def voxel_gen_client(self, scene_cloud, object_frame_id):
        '''
        Service client to get voxel from pointcloud in object frame.
        
        Args:
            scene_cloud.

        Returns:
            Voxelgrid.
        '''
        rospy.loginfo('Waiting for service gen_inference_voxel.')
        rospy.wait_for_service('gen_inference_voxel')
        rospy.loginfo('Calling service gen_inference_voxel.')
        try:
            gen_voxel_proxy = rospy.ServiceProxy('gen_inference_voxel', GenInfVoxel)
            gen_voxel_request = GenInfVoxelRequest()
            gen_voxel_request.scene_cloud = scene_cloud
            gen_voxel_request.voxel_dim = self.voxel_grid_dim
            gen_voxel_request.voxel_size = self.voxel_size
            gen_voxel_request.object_frame_id = object_frame_id

            gen_voxel_response = gen_voxel_proxy(gen_voxel_request) 
            sparse_voxel_grid = np.reshape(gen_voxel_response.voxel_grid, 
                                [len(gen_voxel_response.voxel_grid) / 3, 3])
            return sparse_voxel_grid
        except rospy.ServiceException, e:
            rospy.loginfo('Service gen_inference_voxel call failed: %s'%e)
        rospy.loginfo('Service gen_inference_voxel is executed.')

    ###################Active Learning for Grasp PGM################

    def compute_active_entropy_obj(self, grasp_type, latent_voxel, grasp_config):
        '''
        Compute the max entropy for active learning of the generative model, 
        including the likelihood and prior.
        '''
        #prior = self.compute_grasp_prior(grasp_type, grasp_config)
        log_prior = self.compute_grasp_log_prior(grasp_type, grasp_config)
        prior = np.exp(log_prior)

        suc_prob = self.compute_clf_suc_prob(grasp_type, latent_voxel, grasp_config)
        log_suc_prob = np.log(suc_prob)
        suc_entropy = - suc_prob * prior * (log_suc_prob + log_prior)

        failure_prob = 1 - suc_prob
        log_failure_prob = np.log(failure_prob)
        failure_entropy = - failure_prob * prior * (log_failure_prob + log_prior)

        entropy = suc_entropy + failure_entropy
        return entropy 


    def compute_active_entropy_grad(self, grasp_type, latent_voxel, grasp_config):
        '''
        Compute the max entropy gradient for active learning of the generative model, 
        including the likelihood and prior.
        '''
        log_prior = self.compute_grasp_log_prior(grasp_type, grasp_config)
        prior = np.exp(log_prior)
        prior_grad = self.compute_grasp_prior_grad(grasp_type, grasp_config)

        suc_prob = self.compute_clf_suc_prob(grasp_type, latent_voxel, grasp_config)
        log_suc_prob = np.log(suc_prob)
        suc_prob_grad = self.compute_clf_suc_prob_grad(
                                 grasp_type, latent_voxel, grasp_config)
        suc_entropy_grad = -(suc_prob_grad * prior + suc_prob * prior_grad) * \
                           (1 + log_suc_prob + log_prior)
        suc_entropy = - suc_prob * prior * (log_suc_prob + log_prior)

        failure_prob = 1 - suc_prob
        log_failure_prob = np.log(failure_prob)
        failure_prob_grad = -suc_prob_grad 
        failure_entropy_grad = -(failure_prob_grad * prior + failure_prob * prior_grad) * \
                           (1 + log_failure_prob + log_prior)
        failure_entropy = - failure_prob * prior * (log_failure_prob + log_prior)
       
        entropy_grad = suc_entropy_grad + failure_entropy_grad
        entropy = suc_entropy + failure_entropy

        #Gradient checking
        grad_check = True 
        if grad_check:
            num_grad = self.compute_num_grad(self.compute_active_entropy_obj, 
                                             grasp_type, latent_voxel, grasp_config)
            grad_diff = entropy_grad - num_grad
            print 'entropy_grad:', entropy_grad
            print 'num_grad:', num_grad
            print 'Gradient difference:', grad_diff
            print 'Gradient difference abs mean:', np.mean(abs(grad_diff))
            print 'Gradient difference abs mean percentage:',  np.mean(abs(grad_diff)) / np.mean(abs(entropy_grad))
            print '~~~~~~~~~~~~~~~~~~~~~~~~~~~'

            #print 'entropy_grad:', entropy_grad
            print '################################################################'

        return entropy_grad, entropy


    def compute_likelihood_entropy_obj(self, grasp_type, latent_voxel, grasp_config):
        '''
        Compute the max entropy for active learning of the likelihood.
        '''
        suc_prob = self.compute_clf_suc_prob(grasp_type, latent_voxel, grasp_config)
        log_suc_prob = np.log(suc_prob)
        suc_entropy = - suc_prob * log_suc_prob

        failure_prob = 1 - suc_prob
        log_failure_prob = np.log(failure_prob)
        failure_entropy = - failure_prob * log_failure_prob

        entropy = suc_entropy + failure_entropy
        return entropy 


    def compute_likelihood_entropy_grad(self, grasp_type, latent_voxel, grasp_config):
        '''
        Compute the max entropy gradient for active learning of the likelihood.
        '''
        suc_prob = self.compute_clf_suc_prob(grasp_type, latent_voxel, grasp_config)
        log_suc_prob = np.log(suc_prob)
        suc_prob_grad = self.compute_clf_suc_prob_grad(
                                 grasp_type, latent_voxel, grasp_config)
        suc_entropy_grad = -suc_prob_grad * (1 + log_suc_prob)
        suc_entropy = - suc_prob * log_suc_prob

        failure_prob = 1 - suc_prob
        log_failure_prob = np.log(failure_prob)
        failure_prob_grad = -suc_prob_grad 
        failure_entropy_grad = -failure_prob_grad * (1 + log_failure_prob)
        failure_entropy = - failure_prob * log_failure_prob
       
        entropy_grad = suc_entropy_grad + failure_entropy_grad
        entropy = suc_entropy + failure_entropy

        #Gradient checking
        grad_check = True 
        if grad_check:
            num_grad = self.compute_num_grad(self.compute_likelihood_entropy_obj, 
                                             grasp_type, latent_voxel, grasp_config)
            grad_diff = entropy_grad - num_grad
            print 'entropy_grad:', entropy_grad
            print 'num_grad:', num_grad
            print 'Gradient difference:', grad_diff
            print 'Gradient difference abs mean:', np.mean(abs(grad_diff))
            print 'Gradient difference abs mean percentage:',  np.mean(abs(grad_diff)) / np.mean(abs(entropy_grad))
            print '~~~~~~~~~~~~~~~~~~~~~~~~~~~'

            #print 'entropy_grad:', entropy_grad
            print '################################################################'

        return entropy_grad, entropy


    def compute_likelihood_uncertainty_obj(self, grasp_type, latent_voxel, grasp_config):
        '''
        Compute the uncertainty for active learning of the likelihood.
        '''
        suc_prob = self.compute_clf_suc_prob(grasp_type, latent_voxel, grasp_config)
        if suc_prob <= 0.5:
            uncertainty = suc_prob
        else:
            uncertainty = 1. - suc_prob
        return uncertainty 


    def compute_likelihood_uncertainty_grad(self, grasp_type, latent_voxel, grasp_config):
        '''
        Compute the uncertainty gradient for active learning of the likelihood.
        '''
        #TODO: set suc_prob as one parameter so that we can only compute suc_prob once 
        #for the objective function and the gradient function.
        suc_prob = self.compute_clf_suc_prob(grasp_type, latent_voxel, grasp_config)
        suc_prob_grad = self.compute_clf_suc_prob_grad(
                                 grasp_type, latent_voxel, grasp_config)
        if suc_prob < 0.5:
            uncertainty_grad = suc_prob_grad
        elif suc_prob > 0.5:
            uncertainty_grad = -suc_prob_grad
        else:
            uncertainty_grad = 0.
        return uncertainty_grad


    def sample_grasp_config(self, grasp_type):
        prior_model = self.get_prior_model(grasp_type)
        # Set the random_state of the GMM to be None so that it 
        # generates different random samples
        if prior_model.random_state != None:
            prior_model.random_state = None
        return prior_model.sample()


    def active_obj_bfgs(self, grasp_config, grasp_type, latent_voxel):
        '''
        Objective function for lbfgs/bfgs optimizer of active learning max entropy.
        '''
        #entropy = self.compute_active_entropy_obj(grasp_type, latent_voxel, grasp_config)
        #entropy = self.compute_likelihood_entropy_obj(grasp_type, latent_voxel, grasp_config)
        #return -entropy
        if self.al_strategy == 'clf_uncertainty':
            uncertainty = self.compute_likelihood_uncertainty_obj(grasp_type, latent_voxel, grasp_config)
            obj_val = -uncertainty
        elif self.al_strategy == 'prior':
            obj_val = self.compute_grasp_prior(grasp_type, grasp_config)
        elif self.al_strategy == 'grasp_success':
            obj_val = self.grasp_obj_bfgs(grasp_config, grasp_type, latent_voxel)
        elif self.al_strategy == 'grasp_clf_success':
            obj_val = self.grasp_clf_obj_bfgs(grasp_config, grasp_type, latent_voxel)
        return obj_val


    def active_grad_bfgs(self, grasp_config, grasp_type, latent_voxel):
        '''
        Derivative function for lbfgs/bfgs optimizer of active learning max entropy.
        '''
        #entropy_grad, _ = self.compute_active_entropy_grad(grasp_type, latent_voxel, grasp_config)
        #entropy_grad, _ = self.compute_likelihood_entropy_grad(grasp_type, latent_voxel, grasp_config)
        #return -entropy_grad
        if self.al_strategy == 'clf_uncertainty':
            uncertainty_grad = self.compute_likelihood_uncertainty_grad(grasp_type, latent_voxel, grasp_config)
            obj_grad = -uncertainty_grad
        elif self.al_strategy == 'prior':
            obj_grad = self.compute_grasp_prior_grad(grasp_type, grasp_config)
        elif self.al_strategy == 'grasp_success':
            obj_grad = self.grasp_grad_bfgs(grasp_config, grasp_type, latent_voxel)
        elif self.al_strategy == 'grasp_clf_success':
            obj_grad = self.grasp_clf_grad_bfgs(grasp_config, grasp_type, latent_voxel)
        return obj_grad


if __name__ == '__main__':
    #power_data_path = '/media/kai/multi_finger_sim_data_complete_v4/'
    #voxel_path = power_data_path + 'power_grasps/power_align_failure_grasps.h5'
    prec_data_path = '/mnt/tars_data/multi_finger_sim_data_precision_1/'
    #voxel_path = prec_data_path + 'prec_grasps/prec_align_suc_grasps.h5'
    voxel_path = prec_data_path + 'prec_grasps/prec_align_failure_grasps.h5'
    #voxel_path = '/dataspace/data_kai/test_grasp_inf/power_failure_grasp_voxel_data.h5'
    grasp_voxel_file = h5py.File(voxel_path, 'r')
    grasp_voxel_grids = grasp_voxel_file['grasp_voxel_grids'][()] 
    grasp_configs = grasp_voxel_file['grasp_configs_obj'][()]
    grasp_pgm_inf = GraspPgmInfer(pgm_grasp_type=True) 
    #print grasp_pgm_inf.gradient_descent_inf(object_voxel=grasp_voxel_grids[0],
    #                        grasp_config=grasp_configs[0], infer_grasp_type=True,
    #                        save_grad_to_log=True, object_id=0, grasp_id=0)

    #print grasp_pgm_inf.gd_inf_one_type('prec', grasp_voxel_grids[0], grasp_configs[0],
    #                        save_grad_to_log=False, object_id=-1, grasp_id=0)
    ##print grasp_pgm_inf.quasi_newton_lbfgs_inf('prec', grasp_voxel_grids[5], grasp_configs[5], bfgs=False)
    print grasp_pgm_inf.quasi_newton_lbfgs_inf('prec', grasp_voxel_grids[0], grasp_configs[0], bfgs=False)

    #grasp_config = np.array([-0.04588782,  0.01481504,  0.01793477,  2.20800101,  0.36448444, -0.2931122,
    #          0.2331929,   0.13870653, -0.21761251,  0.21749901,  0.08889905, -0.00225458,
    #            0.53184325,  0.20105151])
    #print grasp_pgm_inf.gd_inf_one_type('prec', grasp_voxel_grids[0], grasp_config,
    #                        save_grad_to_log=True, object_id=-1, grasp_id=0)
    #print grasp_pgm_inf.quasi_newton_lbfgs_inf('prec', grasp_voxel_grids[0], grasp_config, bfgs=False)
    #print grasp_pgm_inf.quasi_newton_lbfgs_inf('prec', grasp_voxel_grids[0], grasp_config, bfgs=True)

    print grasp_configs[4]

    grasp_voxel_file.close()

