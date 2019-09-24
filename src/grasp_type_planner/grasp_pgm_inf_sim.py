#!/usr/bin/env python
import numpy as np
import pickle
import os
import time
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import roslib.packages as rp
from geometry_msgs.msg import Pose, Quaternion, PoseStamped
from prob_grasp_planner.msg import VisualInfo, HandConfig
import tf
import h5py
import roslib.packages as rp
from scipy.optimize import fmin_bfgs
from scipy.optimize import fmin_l_bfgs_b
from scipy.optimize import minimize
from scipy.interpolate import griddata
from scipy.stats import multivariate_normal
from matplotlib import cm
from matplotlib import gridspec

class GraspPgmInferSim:
    def __init__(self, pgm_grasp_type=True):
        self.pgm_grasp_type = pgm_grasp_type

        self.isrr_limit = False #True
        self.hand_config_frame_id = None

        pkg_path = rp.get_pkg_dir('prob_grasp_planner') 

        self.pca_model_path = pkg_path + '/train_models_sim/pca/pca.model'
        self.train_model_path = pkg_path + '/train_models_sim/classifiers/'
        self.prior_path = pkg_path + '/train_models_sim/priors/'

        self.non_type_prior = True
        self.load_learned_models()
        
        #Use the computer tars or t
        self.tars = True #False
        if self.tars:
            self.grasp_config_log_save_path = '/media/kai/logs/multi_finger_sim_data/grad_des_ls_log/'
        else:
            self.grasp_config_log_save_path = '/dataspace/data_kai/logs/multi_finger_sim_data/grad_des_ls_log/'
        self.iter_total_num = 1000#500

        self.log_inf = True
        #The regularization value is related with both the 
        #data (voxel, config) dimension and the number of GMM components. 
        self.reg_log_prior = 0.1 #1.
        #self.reg_log_prior = 0.01
        #self.reg_log_prior = 0.05
        #Regularization for log likelihood, this is mainly used to 
        #test the inference with only prior.
        self.reg_log_lkh = 1.

        self.q_max = 2.
        self.q_min = -2.

    def load_learned_models(self):
        #Load PCA model
        self.pca_model = pickle.load(open(self.pca_model_path, 'rb'))

        #Load logistic regression models and priors.
        if self.non_type_prior:
                self.all_grasp_priors_model = pickle.load(open(self.prior_path + 'all_type_gmm.model', 'rb'))
                self.power_grasp_priors_model = self.all_grasp_priors_model
                self.prec_grasp_priors_model = self.all_grasp_priors_model 
                self.power_grasp_clf_model = pickle.load(open(self.train_model_path + 'power_clf.model', 'rb'))
                self.prec_grasp_clf_model = pickle.load(open(self.train_model_path + 'prec_clf.model', 'rb'))
                self.all_grasp_clf_model = pickle.load(open(self.train_model_path + 'all_type_clf.model', 'rb'))
        else:
            if self.pgm_grasp_type:
                self.power_grasp_clf_model = pickle.load(open(self.train_model_path + 'power_clf.model', 'rb'))
                self.prec_grasp_clf_model = pickle.load(open(self.train_model_path + 'prec_clf.model', 'rb'))
                self.power_grasp_priors_model = pickle.load(open(self.prior_path + 'power_gmm.model', 'rb'))
                self.prec_grasp_priors_model = pickle.load(open(self.prior_path + 'prec_gmm.model', 'rb'))
            else:
                self.all_grasp_clf_model = pickle.load(open(self.train_model_path + 'all_type_clf.model', 'rb'))
                self.all_grasp_priors_model = pickle.load(open(self.prior_path + 'all_type_gmm.model', 'rb'))

    def project_config(self, q):
        '''
        Project the preshape configuration into the valid range.
        '''
        q_proj = np.copy(q)
        q_proj = np.clip(q_proj, self.q_min, self.q_max)
        return q_proj
    
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
        log_prior = prior_model.score_samples([grasp_config])[0]
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
        grad_check = False 
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
        #print 'clf_log_prob:', clf_log_prob
        #print 'log_prior:', log_prior
        obj_prob = clf_log_prob + log_prior 
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
        q = init_hand_config
        #self.setup_palm_pose_limits(q)
        #voxel_grid_dim = object_voxel.shape
        #voxel_num = voxel_grid_dim[0] * voxel_grid_dim[1] * voxel_grid_dim[2]  
        #grasp_1d_voxel = np.reshape(object_voxel, voxel_num)
        #latent_voxel = self.pca_model.transform([grasp_1d_voxel])[0]

        latent_voxel = self.pca_model.transform([object_voxel])[0]

        #latent_voxel = object_voxel

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
            configs = [q[0]]
        #iter_total_num = 100
        delta = 10**-8
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

            if save_grad_to_log:
                configs.append(q[0])
            
        suc_probs = np.array(suc_probs)
        if save_grad_to_log: 
            plt.figure()
            plt.plot(suc_probs, label='suc')
            if self.log_inf:
                plt.plot(np.array(clf_log_probs), label='clf')
                plt.plot(np.array(log_priors), label='prior')
                #print 'suc_probs:', suc_probs
                #print 'clf_log_probs:', clf_log_probs
                #print 'log_priors:', log_priors
            plt.ylabel('Suc Probalities')
            plt.xlabel('Iteration')
            plt.legend(loc="lower right")
            plt.savefig(config_grad_path + 'suc_prob.png')
            plt.cla()
            plt.clf()
            plt.close()

            #print 'configs:', configs
            #self.plot_inf(grasp_type, latent_voxel, configs, config_grad_path + 'inf_result.png')
        
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

        #full_grasp_config = self.convert_preshape_to_full_config(q)
        #print full_grasp_config, suc_probs[-1], suc_probs[0]
        
        return q, suc_probs[-1], suc_probs[0]
        
    def gradient_descent_inf(self, object_voxel, init_hand_config, 
                            save_grad_to_log=False, object_id=None, grasp_id=None):
        #self.hand_config_frame_id = init_hand_config.palm_pose.header.frame_id
        #grasp_config = self.convert_full_to_preshape_config(init_hand_config)
        grasp_config = init_hand_config
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

    def get_obj_bfgs(self, grasp_config, grasp_type, latent_voxel):
        '''
        Objective function for lbfgs/bfgs optimizer.
        '''
        obj_prob = self.compute_log_obj_prob(grasp_type, latent_voxel, grasp_config)
        return -obj_prob
 

    def get_grad_bfgs(self, grasp_config, grasp_type, latent_voxel):
        '''
        Derivative function for lbfgs/bfgs optimizer.
 
        '''
        grad_q, _, _, _ = self.compute_log_d_prob_d_config(grasp_type, latent_voxel, grasp_config)
        return -grad_q
 
    def quasi_newton_lbfgs_inf(self, grasp_type, object_voxel, init_hand_config, bfgs=True):
        '''
        Quasi Newton inference with bfgs/lbfgs update. 
        '''
        t = time.time()
        #self.hand_config_frame_id = init_hand_config.palm_pose.header.frame_id
        #q_init = self.convert_full_to_preshape_config(init_hand_config)

        q_init = init_hand_config

        #q_init = np.zeros(14)

        #self.hand_config_frame_id = init_frame_id
        #q_init = init_hand_config
        #voxel_grid_dim = object_voxel.shape
        #voxel_num = voxel_grid_dim[0] * voxel_grid_dim[1] * voxel_grid_dim[2]  
        #grasp_1d_voxel = np.reshape(object_voxel, voxel_num)
        #latent_voxel = self.pca_model.transform([grasp_1d_voxel])[0]

        latent_voxel = self.pca_model.transform([object_voxel])[0]

        #latent_voxel = object_voxel

        #self.setup_palm_pose_limits(q_init)

        opt_method = 'L-BFGS-B'
        if bfgs:
            opt_method = 'BFGS'
        bnds = []
        for i in range(len(q_init)):
            bnds.append((self.q_min, self.q_max))

        #Notice this is for gradient descent, not ascent.
        opt_res = minimize(self.get_obj_bfgs, q_init, jac=self.get_grad_bfgs, 
                                        args=(grasp_type, latent_voxel,), method=opt_method, bounds=bnds)
        print 'opt_res:', opt_res
        #full_grasp_config = self.convert_preshape_to_full_config(opt_res.x)
        full_grasp_config = opt_res.x
        
        init_suc_prob = self.get_obj_bfgs(q_init, grasp_type, latent_voxel)

        elapased_time = time.time() - t
        print 'Total inference time: ', str(elapased_time)

        print full_grasp_config, -opt_res.fun, -init_suc_prob 
        return full_grasp_config, -opt_res.fun, -init_suc_prob 

    def plot_contour(self, x, y, z, plot_range):
        #https://stackoverflow.com/questions/26999145/
        #matplotlib-making-2d-gaussian-contours-with-transparent-outermost-layer
        plot_range += 0.1
        xi = np.linspace(-plot_range,plot_range,100)
        yi = np.linspace(-plot_range,plot_range,100)
        ## grid the data.
        zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
        # contour the gridded data, plotting dots at the randomly spaced data points.
        CS = plt.contour(xi,yi,zi,6,linewidths=0.5,colors=np.random.rand(3,1))
        #CS = plt.contour(xi,yi,zi,6,linewidths=0.5,colors='k')
        #CS = plt.contourf(xi,yi,zi,15,cmap=plt.cm.jet)
        #CS = plt.contourf(xi,yi,zi,6,cmap=cm.Greys_r)

    def plot_data_gen_gaus(self):
        npts = 1000
        plot_range = 2.
        xy = np.random.uniform(-plot_range, plot_range, (npts, 2))
        x = xy[:, 0]
        y = xy[:, 1]

        suc_data_mean = [0.5, 1.]
        suc_data_covar = 2. * np.array([[0.5, 0.2], [0.2, 0.5]])
        gaus = multivariate_normal(mean=suc_data_mean, cov=suc_data_covar)
        z = gaus.pdf(xy)
        self.plot_contour(x, y, z, plot_range)

        fail_data_mean = [0.5, -1.]
        fail_data_covar = 2. * np.array([[0.5, 0.2], [0.2, 0.5]])
        gaus = multivariate_normal(mean=fail_data_mean, cov=fail_data_covar)
        z = gaus.pdf(xy)
        self.plot_contour(x, y, z, plot_range)

    def plot_prior(self, grasp_type):
        prior_model = self.get_prior_model(grasp_type)
        weights = prior_model.weights_
        means = prior_model.means_
        covariances = prior_model.covariances_
        plot_range = 2 
        for i in xrange(weights.shape[0]):
            gaus = multivariate_normal(mean=means[i], cov=covariances[i])
            x = np.linspace(-plot_range, plot_range, 100)
            y = weights[i] * gaus.pdf(x)
            plt.plot(y, x, color=np.random.rand(3,1))
        plt.ylim(-plot_range, plot_range)
        plt.xlabel('density')
        #plt.ylabel('config')

    def plot_inf(self, grasp_type, latent_voxel, configs, fig_name):
        np.random.seed(1234)
        #plt.figure()
        fig = plt.figure(figsize=(8, 6)) 
        gs = gridspec.GridSpec(1, 2, width_ratios=[2.5, 1.5]) 
        #plt.subplot(121)
        plt.subplot(gs[0])
        plot_range = 2.
        #Plot ground truth Gaussians
        self.plot_data_gen_gaus()

        #Plot priors
        #If prior is 1D, contours can not be used to plot the prior.
        #Can plot as a few points. 
        #prior_model = self.get_prior_model(grasp_type)
        #mean = prior_model.means_[0, 0]
        #voxel_mean = 0.5
        #plt.plot([voxel_mean], [mean], marker='*', markersize=15, color='m')

        #Plot classifier line
        logistic = self.get_logistic_clf(grasp_type)
        b = logistic.intercept_[0]
        w = logistic.coef_[0]
        x1 = -plot_range
        y1 = (w[0] * x1 + b ) / -w[1]
        x2 = plot_range
        y2 = (w[0] * x2 + b ) / -w[1]
        plt.plot([x1, x2], [y1, y2], color='r')

        #Plot inference grasp configs
        voxels = [latent_voxel] * len(configs)
        plt.plot(voxels, configs, color='b', linestyle=':', alpha=0.5)
        plt.scatter(voxels, configs, c=np.linspace(0., 1., len(configs)), s=15.)
        #plt.scatter(voxels, configs, c=np.random.rand(len(configs)), s=10.)
        plt.legend(loc="lower right")
        plt.xlim(-plot_range, plot_range)
        plt.ylim(-plot_range, plot_range)
        plt.xlabel('voxel')
        plt.ylabel('config')
        #plt.show()

        #Plot prior in another sub figure
        #plt.subplot(122)
        plt.subplot(gs[1])
        self.plot_prior(grasp_type)

        #plt.subplots_adjust(wspace=0.3)

        plt.savefig(fig_name)
        plt.close()

    def plot_prob_curves(self, grasp_type='power'):
        grasp_res = 100
        grasp_min = -5.0
        grasp_max = 5.0
        grasp_range = grasp_max-grasp_min
        X = np.array(range(grasp_res))*((grasp_range)/(grasp_res-1)) + grasp_min
        latent_voxel = [0.5]
        L = []
        P = []
        J = []
        for x in X:
            theta = [x]
            l = self.compute_clf_suc_prob(grasp_type, latent_voxel, theta)
            p = self.compute_grasp_prior(grasp_type, theta)
            j = l*p
            L.append(l)
            P.append(p)
            J.append(j)
            print 'L[',x,'] =', l
            print 'P[',x,'] =', p
            print 'J[',x,'] =', j

        plt.figure()
        plt.title('Grasp Success')
        plt.ylabel('Probability')
        plt.xlabel('Theta')
        plt.xlim((grasp_min,grasp_max))
        plt.plot(X,L,label='Likelihood')
        plt.plot(X,P,label='Prior')
        plt.plot(X,J,label='Posterior')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    grasp_pgm_inf = GraspPgmInferSim(pgm_grasp_type=True) 
    #grasp_voxel_grid = [-0.5]
    #init_config = [-2.]
    #grasp_voxel_grid = [-100000000.5]
    #init_config = [-200000000.]
    #grasp_voxel_grid = [0.5]
    #init_config = [-200000000.]
    #grasp_voxel_grid = [0.5]
    #init_config = [-50.]
    voxel_dim = 800 #15
    grasp_config_dim = 14
    data_dim = voxel_dim + grasp_config_dim
    grasp_voxel_grid = [-2.5] * voxel_dim
    init_config = [-20.] * grasp_config_dim

    print '##################GD##################'
    print grasp_pgm_inf.gd_inf_one_type('prec', grasp_voxel_grid, init_config,
                            save_grad_to_log=True, object_id=-1, grasp_id=0)
    print '***************'
    print grasp_pgm_inf.gd_inf_one_type('power', grasp_voxel_grid, init_config,
                            save_grad_to_log=True, object_id=-1, grasp_id=0)
    print '################LBFGS#################'
    print grasp_pgm_inf.quasi_newton_lbfgs_inf('prec', grasp_voxel_grid, init_config, bfgs=False)
    print '***************'
    print grasp_pgm_inf.quasi_newton_lbfgs_inf('power', grasp_voxel_grid, init_config, bfgs=False)
    print '################BFGS#################'
    print grasp_pgm_inf.quasi_newton_lbfgs_inf('prec', grasp_voxel_grid, init_config, bfgs=True)
    print '***************'
    print grasp_pgm_inf.quasi_newton_lbfgs_inf('power', grasp_voxel_grid, init_config, bfgs=True)

