import tensorflow
from tensorflow_probability import distributions as tensorflowd
import numpy as np
from grasp_success_network import GraspSuccessNetwork
from grasp_prior_network import GraspPriorNetwork
import roslib.packages as rp
pkg_path = rp.get_pkg_dir('prob_grasp_planner')
import sys
sys.path.append(pkg_path + '/src/grasp_common_library')
import grasp_common_functions as gcf
from set_config_limits import SetConfigLimits
from scipy.optimize import minimize
import pickle
import h5py
import time
from sklearn.metrics import precision_recall_fscore_support
from sklearn import mixture
import os
from scipy.stats import multivariate_normal
import rospy
import tf
from sensor_msgs.msg import JointState


class GraspVoxelInference:
    '''
    Grasp inference.
    '''

    def __init__(self, grasp_net_model_path=None, prior_model_path=None, 
                    gmm_model_path=None, vis_preshape=False,
                    virtual_hand_parent_tf=''):
        self.grasp_net_model_path = grasp_net_model_path
        self.prior_model_path = prior_model_path
        self.gmm_model_path = gmm_model_path

        self.config_limits = SetConfigLimits()

        self.reg_log_prior = 0.5 # 0.1 # 1.
        #Regularization for log likelihood, this is mainly used to 
        #test the inference with only prior.
        self.reg_log_lkh = 1.

        self.load_grasp_model()
        self.load_gmm_prior()

        self.vis_preshape = vis_preshape
        if self.vis_preshape:
            self.tf_br = tf.TransformBroadcaster()
            self.js_pub = rospy.Publisher('/virtual_hand/allegro_hand_right/joint_states', 
                                          JointState, queue_size=1)
            self.preshape_config = None
        self.virtual_hand_parent_tf = virtual_hand_parent_tf

        
    def load_grasp_net(self, grasp_net_model_path):
        update_voxel_enc = False
        self.grasp_net = GraspSuccessNetwork(update_voxel_enc)
        is_train = False
        self.grasp_net.grasp_net_train_test(train_mode=is_train)

        self.config_suc_grad = tensorflow.gradients(self.grasp_net.grasp_net_res['suc_prob'], 
                                        self.grasp_net.holder_config)
        saver = tensorflow.train.Saver()
        saver.restore(self.tensorflow_sess, grasp_net_model_path)


    def load_prior_net(self, prior_net_model_path):
        update_voxel_enc = False
        self.prior_net = GraspPriorNetwork(update_voxel_enc)
        is_train = False
        # self.prior_net.prior_net_train_test(train_mode=is_train)
        self.prior_net.build_prior_network(voxel_ae=self.grasp_net.voxel_ae)
        prior_saver = tensorflow.train.Saver(tensorflow.get_collection(
                                    tensorflow.GraphKeys.GLOBAL_VARIABLES, 'prior_net'))
        # saver = tensorflow.train.Saver()
        prior_saver.restore(self.tensorflow_sess, prior_net_model_path)


    def load_grasp_model(self):
        tensorflow_config = tensorflow.ConfigProto(log_device_placement=False)
        tensorflow_config.gpu_options.allow_growth = True
        self.tensorflow_sess = tensorflow.Session(config=tensorflow_config)
        # self.tensorflow_sess = tensorflow.Session()

        self.load_grasp_net(self.grasp_net_model_path)
        print 'Loading prior from: ', self.prior_model_path
        #self.prior_model = pickle.load(open(self.prior_model_path, 'rb'))
        self.load_prior_net(self.prior_model_path)
        self.build_mixture_model()
        tensorflow.get_default_graph().finalize()


    def load_gmm_prior(self):
        print 'Loading GMM prior from: ', self.gmm_model_path
        self.gmm_prior_model = pickle.load(open(self.gmm_model_path, 'rb'))
        self.seperate_gmm_comp()


    def seperate_gmm_comp(self):
        if self.gmm_prior_model.means_[0, 2] >= self.gmm_prior_model.means_[1, 2]:
            self.gmm_overhead_model = multivariate_normal(mean=self.gmm_prior_model.means_[0], 
                                                        cov=self.gmm_prior_model.covariances_[0]) 
            self.gmm_side_model = multivariate_normal(mean=self.gmm_prior_model.means_[1], 
                                                        cov=self.gmm_prior_model.covariances_[1]) 
        else:
            self.gmm_overhead_model = multivariate_normal(mean=self.gmm_prior_model.means_[1], 
                                                        cov=self.gmm_prior_model.covariances_[1]) 
            self.gmm_side_model = multivariate_normal(mean=self.gmm_prior_model.means_[0], 
                                                        cov=self.gmm_prior_model.covariances_[0]) 

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
        [suc_prob] = self.tensorflow_sess.run(
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
        [suc_prob, config_gradient] = self.tensorflow_sess.run(
                [self.grasp_net.grasp_net_res['suc_prob'], self.config_suc_grad], 
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
        log_config_grad = config_grad / suc_prob
        return log_config_grad


    def build_mixture_model(self):
        k = self.prior_net.num_components 
        n = self.prior_net.config_dim
        self.config_holder = tensorflow.placeholder(tensorflow.float32, [None, n],
                                            name='config_holder')
        self.locs_holder = tensorflow.placeholder(tensorflow.float32, [None, k, n], 
                                            name='locs_holder')
        self.scales_holder = tensorflow.placeholder(tensorflow.float32, [None, k, n], 
                                            name='scales_holder')
        self.logits_holder = tensorflow.placeholder(tensorflow.float32, [None, k], 
                                            name='logits_holder')

        mix_first_locs = tensorflow.transpose(self.locs_holder, [1, 0, 2])
        mix_first_scales = tensorflow.transpose(self.scales_holder, [1, 0, 2])
        cat = tensorflowd.Categorical(logits=self.logits_holder)
        mix_components = []
        for loc, scale in zip(tensorflow.unstack(mix_first_locs), tensorflow.unstack(mix_first_scales)):
        # for loc, scale in zip(mix_first_locs, mix_first_scales):
            normal = tensorflowd.MultivariateNormalDiag(loc=loc, scale_diag=scale)
            mix_components.append(normal)
        mixture = tensorflowd.Mixture(cat=cat, components=mix_components)

        self.prior_prob = mixture.prob(self.config_holder)
        self.prior_prob_grad = tensorflow.gradients(self.prior_prob, self.config_holder)
        self.prior_log_prob = mixture.log_prob(self.config_holder)
        self.prior_log_prob_grad = tensorflow.gradients(self.prior_log_prob, self.config_holder)
        self.prior_sample = mixture.sample()
        self.seperate_comp_sample = [mix_components[0].sample(), 
                                     mix_components[1].sample()]


    def get_prior_mixture(self, grasp_voxel_grid, grasp_obj_size):
        feed_dict = {
            self.prior_net.voxel_ae.is_train: False,
            self.prior_net.voxel_ae.partial_voxel_grids: [grasp_voxel_grid],
            self.prior_net.is_train: False,
            self.prior_net.holder_obj_size: [grasp_obj_size],
        }
        [self.locs, self.scales, self.logits] = self.tensorflow_sess.run(
                        [self.prior_net.prior_net_res['locs'],
                        self.prior_net.prior_net_res['scales'],
                        self.prior_net.prior_net_res['logits']], 
                        feed_dict=feed_dict)


    def compute_grasp_log_prior(self, grasp_config):
        '''
        Compute the grasp configuration log prior.
        '''
        feed_dict = {self.locs_holder: self.locs, 
                    self.scales_holder: self.scales,
                    self.logits_holder: self.logits,
                    self.config_holder: [grasp_config]}
        [log_prior] = self.tensorflow_sess.run([self.prior_log_prob], 
                                        feed_dict=feed_dict)  
        return log_prior[0]


    def compute_grasp_prior(self, grasp_config):
        '''
        Compute the grasp configuration prior.
        '''
        feed_dict = {self.locs_holder: self.locs, 
                    self.scales_holder: self.scales,
                    self.logits_holder: self.logits,
                    self.config_holder: [grasp_config]}
        [prior] = self.tensorflow_sess.run([self.prior_prob], 
                                        feed_dict=feed_dict)  
        return prior[0]


    def grasp_log_prior_grad(self, grasp_config):
        '''
        Compute the grasp configuration log prior gradient with respect
        to grasp configuration.
        '''
        feed_dict = {self.locs_holder: self.locs, 
                    self.scales_holder: self.scales,
                    self.logits_holder: self.logits,
                    self.config_holder: [grasp_config]}
        [log_prior_grad] = self.tensorflow_sess.run([self.prior_log_prob_grad], 
                                        feed_dict=feed_dict)  
        return log_prior_grad[0][0]


    def grasp_prior_grad(self, grasp_config):
        '''
        Compute the grasp configuration prior gradient with respect
        to grasp configuration.
        '''
        feed_dict = {self.locs_holder: self.locs, 
                    self.scales_holder: self.scales,
                    self.logits_holder: self.logits,
                    self.config_holder: [grasp_config]}
        [prior_grad] = self.tensorflow_sess.run([self.prior_prob_grad], 
                                        feed_dict=feed_dict)  
        return prior_grad[0][0]


    def compute_grasp_log_prior_gmm(self, grasp_config):
        '''
        Compute the grasp configuration prior.
        '''
        log_prior = self.gmm_prior_model.score_samples([grasp_config])[0]
        return log_prior


    def compute_grasp_log_prior_gmm(self, grasp_config):
        '''
        Compute the grasp configuration prior.
        '''
        log_prior = self.gmm_prior_model.score_samples([grasp_config])[0]
        prior = np.exp(log_prior)
        return prior


    def grasp_log_prior_grad_gmm(self, grasp_config):
        '''
        Compute the grasp configuration log prior gradient with respect
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


    def compute_num_grad(self, func, grasp_config, 
                        grasp_voxel_grid, grasp_obj_size):
        eps = 10**-5
        grad = np.zeros(len(grasp_config))
        for i in xrange(len(grasp_config)):
            grasp_config_plus = np.copy(grasp_config)
            grasp_config_plus[i] += eps
            if grasp_voxel_grid is not None:
                obj_prob_plus = func(grasp_config_plus, 
                            grasp_voxel_grid, grasp_obj_size)
            else:
                obj_prob_plus = func(grasp_config_plus)
            grasp_config_minus = np.copy(grasp_config)
            grasp_config_minus[i] -= eps
            if grasp_voxel_grid is not None:
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


    def grasp_log_posterior(self, grasp_config, grasp_voxel_grid,
                            grasp_obj_size, prior_name):
        log_suc_prob = self.grasp_clf_log_suc_prob(grasp_config, grasp_voxel_grid,
                            grasp_obj_size)
        if prior_name == 'GMM':
            log_prior = self.compute_grasp_log_prior_gmm(grasp_config)
        elif prior_name == 'MDN':
            log_prior = self.compute_grasp_log_prior(grasp_config)
        elif prior_name == 'Constraint':
            log_prior = 0.

        reg_log_suc_prob = log_suc_prob * self.reg_log_lkh
        reg_log_prior = log_prior * self.reg_log_prior
        log_posterior = reg_log_suc_prob + reg_log_prior
        # print 'reg_log_suc_prob:', reg_log_suc_prob 
        # print 'reg_log_prior:', reg_log_prior
        # print 'log_posterior:', log_posterior
        return np.float64(log_posterior)


    def grasp_norm_posterior(self, grasp_config, grasp_voxel_grid,
                            grasp_obj_size, prior_name):
        suc_prob = self.pred_clf_suc_prob(grasp_config, 
                                    grasp_voxel_grid, grasp_obj_size)
        if prior_name == 'GMM':
            log_prior = self.compute_grasp_log_prior_gmm(grasp_config)
        elif prior_name == 'MDN':
            log_prior = self.compute_grasp_log_prior(grasp_config)
        elif prior_name == 'Constraint':
            log_prior = 0.
        reg_suc_prob = 0.5 * suc_prob
        sigmoid_log_prior = self.sigmoid(log_prior)
        reg_log_prior = 0.5 * sigmoid_log_prior
        log_posterior = reg_suc_prob + reg_log_prior
        # print 'reg_suc_prob:', reg_suc_prob 
        # print 'reg_log_prior:', reg_log_prior
        # print 'log_posterior:', log_posterior
        return np.float64(log_posterior)


    def grasp_neg_log_posterior(self, grasp_config, grasp_voxel_grid,
                            grasp_obj_size, prior_name):
        return -self.grasp_log_posterior(grasp_config, 
                        grasp_voxel_grid, grasp_obj_size, prior_name)
        # return -self.grasp_norm_posterior(grasp_config, 
        #                 grasp_voxel_grid, grasp_obj_size, use_gmm)


    def grasp_log_posterior_grad(self, grasp_config, grasp_voxel_grid,
                            grasp_obj_size, prior_name):
        clf_log_suc_grad = self.grasp_clf_log_suc_grad(grasp_config, grasp_voxel_grid,
                            grasp_obj_size)
        if prior_name == 'GMM':
            log_prior_grad = self.grasp_log_prior_grad_gmm(grasp_config)
        elif prior_name == 'MDN':
            log_prior_grad = self.grasp_log_prior_grad(grasp_config)
        elif prior_name == 'Constraint':
            log_prior_grad = 0.
        reg_log_suc_grad = clf_log_suc_grad * self.reg_log_lkh
        reg_log_prior_grad = log_prior_grad * self.reg_log_prior
        log_post_grad = reg_log_suc_grad + reg_log_prior_grad
        # print 'reg_log_suc_grad', reg_log_suc_grad
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

            num_grad = self.compute_num_grad(self.grasp_log_posterior,
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


    def grasp_norm_posterior_grad(self, grasp_config, grasp_voxel_grid,
                            grasp_obj_size, prior_name):
        suc_grad, suc_prob = self.compute_clf_config_grad(grasp_config, 
                                        grasp_voxel_grid, grasp_obj_size)
        if prior_name == 'GMM':
            log_prior = self.compute_grasp_log_prior_gmm(grasp_config) 
            log_prior_grad = self.grasp_log_prior_grad_gmm(grasp_config)
        elif prior_name == 'MDN':
            log_prior = self.compute_grasp_log_prior(grasp_config) 
            log_prior_grad = self.grasp_log_prior_grad(grasp_config)
        elif prior_name == 'Constraint':
            log_prior = 0.
            log_prior_grad = 0.
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


    def grasp_neg_log_post_grad(self, grasp_config, grasp_voxel_grid,
                            grasp_obj_size, prior_name):
        return  -self.grasp_log_posterior_grad(grasp_config, 
                        grasp_voxel_grid, grasp_obj_size, prior_name)
        # return  -self.grasp_norm_posterior_grad(grasp_config, 
        #                 grasp_voxel_grid, grasp_obj_size, use_gmm)


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


    def sigmoid_log_prior(self, grasp_config):
        log_prior = self.compute_grasp_log_prior(grasp_config) 
        return self.sigmoid(log_prior)
           

    def sample_grasp_config_gmm(self):
        # Set the random_state of the GMM to be None so that it 
        # generates different random samples
        if self.gmm_prior_model.random_state != None:
            self.gmm_prior_model.random_state = None
        return self.gmm_prior_model.sample()


    def sample_grasp_config(self):
        feed_dict = {self.locs_holder: self.locs, 
                    self.scales_holder: self.scales,
                    self.logits_holder: self.logits}
        [sample] = self.tensorflow_sess.run([self.prior_sample], 
                                    feed_dict=feed_dict)  
        return sample
 

    def sample_config_gmm_explicit(self, grasp_type):
        # return self.gmm_prior_model.sample()
        if grasp_type == 'overhead':
            sample = self.gmm_overhead_model.rvs() 
        elif grasp_type == 'side':
            sample = self.gmm_side_model.rvs() 
        else:
            print 'Wrong grasp type for GMM sampling!'
            return None
        return sample


    def sample_config_explicit(self, grasp_type):
        feed_dict = {self.locs_holder: self.locs, 
                    self.scales_holder: self.scales,
                    self.logits_holder: self.logits}
        print 'self.locs: ', self.locs
        if self.locs[0, 0, 2] >= self.locs[0, 1, 2]:
            overhead_idx = 0
            side_idx = 1
        else:
            overhead_idx = 1
            side_idx = 0
        if grasp_type == 'overhead':
            [sample] = self.tensorflow_sess.run([self.seperate_comp_sample[overhead_idx]], 
                                        feed_dict=feed_dict)  
        elif grasp_type == 'side':
            [sample] = self.tensorflow_sess.run([self.seperate_comp_sample[side_idx]], 
                                        feed_dict=feed_dict)  
        else:
            print 'Wrong grasp type for MDN sampling!'
            return None
        return sample


    def mdn_mean_explicit(self, grasp_voxel_grid, grasp_obj_size, grasp_type):
        self.locs, self.scales, self.logits = [None] * 3
        self.get_prior_mixture(grasp_voxel_grid, grasp_obj_size)
        print 'self.locs: ', self.locs
        if self.locs[0, 0, 2] >= self.locs[0, 1, 2]:
            overhead_idx = 0
            side_idx = 1
        else:
            overhead_idx = 1
            side_idx = 0
        if grasp_type == 'overhead':
            comp_mean = self.locs[0, overhead_idx, :]
        elif grasp_type == 'side':
            comp_mean = self.locs[0, side_idx, :]
        else:
            print 'Wrong grasp type for MDN mean!'
            return None
        return comp_mean


    def max_grasp_suc_bfgs(self, grasp_voxel_grid, grasp_obj_size,
                            prior_name=False, cfg_init=None, grasp_type='unknown',
                            bfgs=False, ): 
        t = time.time()
        self.locs, self.scales, self.logits = [None] * 3
        self.get_prior_mixture(grasp_voxel_grid, grasp_obj_size)

        print 'cfg_init: ', cfg_init
        print 'grasp_type: ', grasp_type
        if prior_name == 'GMM':
            if grasp_type == 'unknown':
                config_sample, _ = self.sample_grasp_config_gmm()
                config_init = config_sample[0].astype('float64')
            else:
                config_init = self.sample_config_gmm_explicit(grasp_type)
        elif prior_name == 'MDN':
            if grasp_type == 'unknown':
                config_sample = self.sample_grasp_config() 
            else:
                config_sample = self.sample_config_explicit(grasp_type)
            print 'config_sample: ', config_sample
            config_init = config_sample[0]
        elif prior_name == 'Constraint':
            config_init = cfg_init
        else:
            print 'Wrong prior name for inference!'

        print 'config_init:', config_init

        opt_method = 'L-BFGS-B'
        if bfgs:
            opt_method = 'BFGS'

        bnds = []
        if prior_name == 'Constraint':
            for i in xrange(self.config_limits.palm_loc_dof_dim):
                bnds.append((config_init[i] - 0.05, config_init[i] + 0.05))
            for i in xrange(self.config_limits.palm_loc_dof_dim, self.config_limits.palm_dof_dim):
                bnds.append((config_init[i] - 0.05 * np.pi, config_init[i] + 0.05 * np.pi))
            for i in xrange(self.config_limits.palm_dof_dim, self.config_limits.config_dim):
                bnds.append((self.config_limits.isrr_config_lower_limit[i], 
                             self.config_limits.isrr_config_upper_limit[i]))

        else:
            for i in xrange(self.config_limits.config_dim):
                bnds.append((self.config_limits.preshape_config_lower_limit[i], 
                             self.config_limits.preshape_config_upper_limit[i]))
        bnds = np.array(bnds).astype('float64')
        print 'bnds:', bnds
        
        res_info = {'inf_log_prior':-1, 'inf_suc_prob':-1,
                    'init_log_prior':-1, 'init_suc_prob':-1}
        opt_res = minimize(self.grasp_neg_log_posterior, config_init, 
                            jac=self.grasp_neg_log_post_grad, 
                            args=(grasp_voxel_grid, grasp_obj_size, prior_name,), 
                            method=opt_method, bounds=bnds)
        print opt_res
        obj_val_inf = -opt_res.fun
        config_inf = opt_res.x

        print 'obj_val_inf:', obj_val_inf
        inf_suc_prob = self.pred_clf_suc_prob(config_inf, 
                                grasp_voxel_grid, grasp_obj_size)
        if prior_name == 'GMM':
            inf_log_prior = self.compute_grasp_log_prior_gmm(config_inf)
        elif prior_name == 'MDN':
            inf_log_prior = self.compute_grasp_log_prior(config_inf)
        elif prior_name == 'Constraint':
            inf_log_prior = 0.
        res_info['inf_suc_prob'] = inf_suc_prob
        res_info['inf_log_prior'] = inf_log_prior
        print 'inf_suc_prob:', inf_suc_prob
        print 'inf_suc_log_prob:', np.log(inf_suc_prob)
        print 'inf_log_prior:', inf_log_prior

        obj_val_init = self.grasp_log_posterior(config_init, 
                                        grasp_voxel_grid, grasp_obj_size, prior_name)
        # obj_val_init = self.grasp_norm_posterior(config_init, 
        #                                 grasp_voxel_grid, grasp_obj_size)
        print 'obj_val_init:', obj_val_init
        init_suc_prob = self.pred_clf_suc_prob(config_init, 
                                grasp_voxel_grid, grasp_obj_size)
        if prior_name == 'GMM':
            init_log_prior = self.compute_grasp_log_prior_gmm(config_init)
        elif prior_name == 'MDN':
            init_log_prior = self.compute_grasp_log_prior(config_init)
        elif prior_name == 'Constraint':
            init_log_prior = 0.
        res_info['init_suc_prob'] = init_suc_prob
        res_info['init_log_prior'] = init_log_prior
        print 'init_suc_prob:', init_suc_prob
        print 'init_suc_log_prob:', np.log(init_suc_prob)
        print 'init_log_prior:', init_log_prior

        elapased_time = time.time() - t
        print 'Total inference time: ', str(elapased_time)

        return config_inf, obj_val_inf, \
                config_init, obj_val_init, res_info 


if __name__ == '__main__':
    grasp_net_model_path = pkg_path + '/models/grasp_al_net/' + \
                       'grasp_net_freeze_enc_10_sets.ckpt'
    prior_model_path = pkg_path + '/models/grasp_al_prior/prior_net_freeze_enc_10_sets.ckpt'
    gmm_model_path = pkg_path + '/models/grasp_al_prior/gmm_10_sets'
    ginf = GraspVoxelInference(grasp_net_model_path, prior_model_path, gmm_model_path) 

    test_data_path = '/mnt/tars_data/gazebo_al_grasps/test/' + \
                    'merged_grasp_data_6_16_and_6_18.h5'
    data_file = h5py.File(test_data_path, 'r')
    grasp_id = 100
    grasp_config_obj_key = 'grasp_' + str(grasp_id) + '_config_obj'
    grasp_full_config = data_file[grasp_config_obj_key][()] 
    preshape_config_idx = list(xrange(8)) + [10, 11] + \
                           [14, 15] + [18, 19]
    grasp_preshape_config = grasp_full_config[preshape_config_idx]
    grasp_sparse_voxel_key = 'grasp_' + str(grasp_id) + '_sparse_voxel'
    sparse_voxel_grid = data_file[grasp_sparse_voxel_key][()]
    obj_dim_key = 'grasp_' + str(grasp_id) + '_dim_w_h_d'
    obj_size = data_file[obj_dim_key][()]
    grasp_label_key = 'grasp_' + str(grasp_id) + '_label'
    grasp_label = data_file[grasp_label_key][()]
    
    voxel_grid_full_dim = [32, 32, 32]
    voxel_grid = np.zeros(tuple(voxel_grid_full_dim))
    voxel_grid_index = sparse_voxel_grid.astype(int)
    voxel_grid[voxel_grid_index[:, 0], voxel_grid_index[:, 1],
                voxel_grid_index[:, 2]] = 1
    voxel_grid = np.expand_dims(voxel_grid, -1)

    data_file.close()
    
    ginf.max_grasp_suc_bfgs(voxel_grid, obj_size, prior_name='MDN')
   
