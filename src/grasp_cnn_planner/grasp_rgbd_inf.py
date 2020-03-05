import os
import time
import cv2
import numpy as np
from grasp_rgbd_patches_net import GraspRgbdPatchesNet
from grasp_rgbd_config_net import GraspRgbdConfigNet
import matplotlib.pyplot as plt
from gen_rgbd_images import GenRgbdImage
import roslib.packages as rp
from compute_finger_tip_location import ComputeFingerTipPose
from prob_grasp_planner.msg import VisualInfo, HandConfig
from geometry_msgs.msg import Pose, Quaternion, PoseStamped
import tf
from scipy.optimize import fmin_bfgs
from scipy.optimize import fmin_l_bfgs_b
from scipy.optimize import minimize
import rospy
from sklearn import mixture
import pickle

class GraspInf:
    def __init__(self, config_net=False, use_hd=True):
        self.use_hd = use_hd
        self.isrr_limit = False #True
        self.palm_loc_dof_dim = 3
        self.palm_dof_dim = 6
        self.finger_joints_dof_dim = 8
        self.theta_dim = self.palm_dof_dim + self.finger_joints_dof_dim 
        self.setup_config_limits()
        self.fingers_num = 4
        self.gen_rgbd = GenRgbdImage() 
        self.config_net = config_net
        if not self.config_net:
            self.grasp_net = GraspRgbdPatchesNet()
            self.compute_finger_tip_loc = ComputeFingerTipPose()
        else:
            self.grasp_net = GraspRgbdConfigNet() 

        self.grasp_net.init_net_inf()
        
        # 3 dof location of the palm + 3 dof orientation of the palm (4 parameters for quaternion, 
        # 3 for Euler angles) + 1st two joints of the thumb + 1st joint of other three fingers
        # Other joint angles are fixed for grasp preshape inference.
        self.hand_config_eps = np.zeros(self.theta_dim) 
        # epsilon of the palm location. Unit is meter.
        self.hand_config_eps[:self.palm_loc_dof_dim] = 0.01
        # epsilon of the palm orientation. 
        # Use Euler angles instead of Quarternion to represent the palm orientation?
        self.hand_config_eps[self.palm_loc_dof_dim:self.palm_dof_dim] = np.pi * 0.01
        # epsilon of finger joints
        self.hand_config_eps[self.palm_dof_dim:] = np.pi * 0.01
        self.hand_config_frame_id = None
        self.grasp_config_log_save_path = '/data_space/data_kai/logs/multi_finger_exp_data/grad_des_ls_log/'
        self.iter_total_num = 100#500

        self.listener = tf.TransformListener()

        self.log_inf = True

        self.reg_log_prior = 1.
        self.load_grasp_prior()

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

        #print 'convert_preshape_to_full_config:'
        #print preshape_config
        #print hand_config
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

        #print 'convert_full_to_preshape_config:'
        #print hand_config
        #print preshape_config
        return np.array(preshape_config)

    def get_grasp_config_patches(self, rgbd_image, theta):
        hand_config = self.convert_preshape_to_full_config(theta)
        self.compute_finger_tip_loc.set_up_input(hand_config.palm_pose, hand_config.hand_joint_state, use_hd=self.use_hd) 
        self.compute_finger_tip_loc.proj_finger_palm_locs_to_img()
        palm_image_loc = self.compute_finger_tip_loc.palm_image_loc
        finger_tip_image_locs = self.compute_finger_tip_loc.finger_tip_image_locs
        palm_patch, finger_tip_patches = self.gen_rgbd.get_finger_palm_patches(rgbd_image, palm_image_loc, finger_tip_image_locs, 
                                                #object_id=req.object_id, grasp_id=req.grasp_id, 
                                                #object_name=req.object_name, grasp_label=req.grasp_success_label,
                                                save=False)
        return palm_patch, finger_tip_patches

    def compute_d_patch_d_theta(self, rgbd_image, theta, theta_idx, theta_idx_eps):
        '''
        Compute the gradient of the palm and finger patches with respect to one dimension of 
        given theta (hand configuration).
        '''
        theta_eps = np.zeros(theta.shape)
        theta_eps[theta_idx] = theta_idx_eps
        theta_plus_eps = theta + theta_eps
        palm_patch_theta_plus, finger_tip_patches_theta_plus = self.get_grasp_config_patches(rgbd_image, theta_plus_eps) 
        theta_minus_eps = theta - theta_eps
        palm_patch_theta_minus, finger_tip_patches_theta_minus = self.get_grasp_config_patches(rgbd_image, theta_minus_eps)
        d_palm_patch_d_theta_idx = (palm_patch_theta_plus - palm_patch_theta_minus) / (2 * theta_idx_eps)
        d_finger_patches_d_theta_idx = (np.array(finger_tip_patches_theta_plus) - \
                                        np.array(finger_tip_patches_theta_minus)) / (2 * theta_idx_eps) 
        return d_palm_patch_d_theta_idx, d_finger_patches_d_theta_idx

    def compute_d_prob_d_theta(self, rgbd_image, theta):
        '''
        Compute the gradients of the grasp success probablity with respect to grasp
        configuration parameters theta.
        '''
        d_prob_d_palm_patch, d_prob_d_finger_patches, suc_prob = \
                                    self.get_d_prob_d_patches_and_suc_prob(rgbd_image, theta)
        #theta_dim = hand_config.shape[0]
        #theta = hand_config[:7]
        d_prob_d_theta = np.zeros(self.theta_dim)
        for i in xrange(self.theta_dim):
            d_palm_patch_d_theta_i, d_finger_patches_d_theta_i = \
                    self.compute_d_patch_d_theta(rgbd_image, theta, i, self.hand_config_eps[i])
            d_prob_d_theta_i_palm = np.multiply(d_prob_d_palm_patch, d_palm_patch_d_theta_i)
            d_prob_d_theta_i = np.sum(d_prob_d_theta_i_palm)
            for j, d_f_j_patch_d_theta_i in enumerate(d_finger_patches_d_theta_i):
                d_prob_d_theta_i_f_j = np.multiply(d_prob_d_finger_patches[j], d_f_j_patch_d_theta_i)
                d_prob_d_theta_i += np.sum(d_prob_d_theta_i_f_j)
            d_prob_d_theta[i] = d_prob_d_theta_i 
        return d_prob_d_theta, suc_prob

    def compute_d_prob_d_theta_config_net(self, theta, grasp_rgbd_patch):
        '''
        Compute the gradients of the grasp success probablity with respect to grasp
        configuration parameters theta using the rgbd + configuration net.
        '''
        d_prob_d_theta, suc_prob = self.grasp_net.get_config_gradients(grasp_rgbd_patch, theta)
        suc_prob_2 = self.get_config_suc_prob_config_net(theta, grasp_rgbd_patch)

        #Gradient checking
        grad_check = True 
        if grad_check:
            print 'suc_prob:', suc_prob
            print 'suc_prob_2', suc_prob_2
            num_grad = self.compute_num_grad(self.get_config_suc_prob_config_net, 
                                            theta, grasp_rgbd_patch)
            grad_diff = d_prob_d_theta - num_grad
            print 'd_prob_d_theta:', d_prob_d_theta
            print 'num_grad:', num_grad
            print 'Gradient difference:', grad_diff
            print 'Gradient difference abs mean:', np.mean(abs(grad_diff))
            print 'Gradient difference abs mean percentage:',  np.mean(abs(grad_diff)) / np.mean(abs(d_prob_d_theta))
            print 'Gradient difference abs percentage:', grad_diff / d_prob_d_theta
            print 'Gradient percentage:', num_grad / d_prob_d_theta
            #print '~~~~~~~~~~~~~~~~~~~~~~~~~~~'

            #print 'd_prob_d_theta:', d_prob_d_theta
            print '################################################################'

        return d_prob_d_theta, suc_prob
        #return 0.25*d_prob_d_theta, suc_prob
        #return num_grad, suc_prob


    def get_d_prob_d_patches_and_suc_prob(self, rgbd_image, theta):
        '''
        Get the gradients of grasp success probability with respect the grasp patches and 
        the grasp sucess probability.
        '''
        palm_patch, finger_tip_patches = self.get_grasp_config_patches(rgbd_image, theta)
        d_prob_d_palm_patch, d_prob_d_finger_patches, suc_prob = \
                self.grasp_net.get_rgbd_gradients(palm_patch, finger_tip_patches)
        #print d_prob_d_palm_patch, d_prob_d_finger_patches, suc_prob
        return d_prob_d_palm_patch, d_prob_d_finger_patches, suc_prob

    def get_config_suc_prob(self, theta, rgbd_image):
        '''
        Get the grasp sucess probability.
        '''
        palm_patch, finger_tip_patches = self.get_grasp_config_patches(rgbd_image, theta)
        suc_prob = self.grasp_net.get_suc_prob(palm_patch, finger_tip_patches)
        return suc_prob


    def get_config_suc_prob_config_net(self, theta, grasp_rgbd_patch):
        '''
        Get the grasp sucess probability using rgbd + configuration net.
        '''
        suc_prob = self.grasp_net.get_suc_prob(grasp_rgbd_patch, theta)
        return suc_prob


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
        self.preshape_config_lower_limit = np.zeros(self.theta_dim)
        #self.preshape_config_lower_limit[:self.palm_dof_dim] = np.array([-5., -5., -5., -np.pi, -np.pi, -np.pi])
        self.preshape_config_lower_limit[:self.palm_dof_dim] = np.array([-1., -1., -2., -np.pi, -np.pi, -np.pi])

        self.preshape_config_upper_limit = np.zeros(self.theta_dim)
        #two_pi = 2 * np.pi
        #self.preshape_config_upper_limit[:self.palm_dof_dim] = np.array([5., 5., 5., np.pi, np.pi, np.pi])
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
        #pos_range = 0.02
        #ort_range = 0.02 * np.pi
        #pos_range = 0.2
        #ort_range = 0.5 * np.pi
        pos_range = 0.05
        ort_range = 0.05 * np.pi
        lower_limit_range = -np.array([pos_range, pos_range, pos_range, ort_range, ort_range, ort_range])
        upper_limit_range = np.array([pos_range, pos_range, pos_range, ort_range, ort_range, ort_range])
        self.preshape_config_lower_limit[:self.palm_dof_dim] = config_init[:self.palm_dof_dim] + lower_limit_range
        self.preshape_config_upper_limit[:self.palm_dof_dim] = config_init[:self.palm_dof_dim] + upper_limit_range

    def project_config(self, q):
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
        for i in range(self.palm_loc_dof_dim, self.theta_dim):
            if q_proj[i] > np.pi:
                q_proj[i] -= two_pi
            if q_proj[i] < self.preshape_config_lower_limit[i]:
                q_proj[i] = self.preshape_config_lower_limit[i]
            if q_proj[i] > self.preshape_config_upper_limit[i]:
                q_proj[i] = self.preshape_config_upper_limit[i]

        return q_proj

    def find_learning_rate_bt(self, rgbd, q, suc_prob, grad_q, 
                                line_search_log=None, use_talor=False):
        '''
        Backtracking line search to find the learning rate.
        '''
        t = time.time()
        iter_num = -1
        #alpha = 0.0005
        alpha = 0.001
        #alpha = 0.01
        #alpha = 0.05
        tao = 0.5
        beta = 0.1
        l = 0
        #iter_limit = 100
        #iter_limit = 10
        iter_limit = 20
        q_new = q + alpha * grad_q
        if line_search_log is not None:
            line_search_log.writelines('q_new: ' + str(q_new))
            line_search_log.writelines('\n')
        q_new = self.project_config(q_new)
        if line_search_log is not None:
            line_search_log.writelines('q_new after projection: ' + str(q_new))
            line_search_log.writelines('\n')
        #_, _, suc_prob_new = self.get_d_prob_d_patches_and_suc_prob(rgbd, q_new)
        if not self.config_net:
            suc_prob_new = self.get_config_suc_prob(q_new, rgbd)
        else:
            suc_prob_new = self.get_config_suc_prob_config_net(q_new, rgbd)
        talor_1st_order = beta * alpha * np.inner(grad_q, grad_q)
        #Double check the mean is the right thing to do or not?
        talor_1st_order = np.mean(talor_1st_order)
        if line_search_log is not None:
            line_search_log.writelines('use_talor: ' + str(use_talor))
            line_search_log.writelines('\n')
        #print suc_prob_new, suc_prob, talor_1st_order
        #print type(suc_prob_new), type(suc_prob), type(use_talor), type(talor_1st_order)
        #while suc_prob_new <= suc_prob + use_talor * talor_1st_order:
        while suc_prob_new <= suc_prob:
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
            #_, _, suc_prob_new = self.get_d_prob_d_patches_and_suc_prob(rgbd, q_new)
            if not self.config_net:
                suc_prob_new = self.get_config_suc_prob(q_new, rgbd)
            else:
                suc_prob_new = self.get_config_suc_prob_config_net(q_new, rgbd)
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

    def gradient_descent_inf(self, rgbd_image, init_hand_config, 
                            save_grad_to_log=False, object_id=None, grasp_id=None):
        '''
        Gradient descent inference with line search. 
        '''
        #self.compute_finger_tip_loc.set_up_input(init_hand_config.palm_pose, 
        #                                            init_hand_config.hand_joint_state, use_hd=self.use_hd) 

        self.hand_config_frame_id = init_hand_config.palm_pose.header.frame_id
        #self.joints_num_per_finger = len(init_hand_config.hand_joint_state.position) / self.fingers_num

        q = self.convert_full_to_preshape_config(init_hand_config)
        self.setup_palm_pose_limits(q)

        if save_grad_to_log: 
            config_grad_path = self.grasp_config_log_save_path + 'object_' + str(object_id) \
                                    + '_grasp_' + str(grasp_id) + '/'
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
        #iter_total_num = 100
        delta = 10**-8
        use_talor = 0.
        #if grasp_id % 2 == 1:
        #    use_talor = 1.
        
        #save_grad = False
        #if object_id % 10 != 0:
        #    save_grad = False

        for iter_num in xrange(self.iter_total_num):
            print 'iter:', iter_num
            if not self.config_net:
                grad_q, suc_prob = self.compute_d_prob_d_theta(rgbd_image, q)#, save_grad, object_id, iter_num)
            else:
                grad_q, suc_prob = self.compute_d_prob_d_theta_config_net(q, rgbd_image)
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
            #Stop if gradient is too small
            if grad_norm < delta:
                if save_grad_to_log: 
                    log_file.writelines('Gradient too small, stop iteration!\n')
                break
           
            if save_grad_to_log: 
                line_search_log.writelines('iter: ' + str(iter_num))
                line_search_log.writelines('\n')
            q_learn_rate = self.find_learning_rate_bt(rgbd_image, q, suc_prob, grad_q, line_search_log, use_talor)
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
            plt.plot(suc_probs)
            plt.ylabel('Suc Probalities')
            plt.xlabel('Iteration')
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

        full_grasp_config = self.convert_preshape_to_full_config(q)
        return full_grasp_config, suc_probs[-1], suc_probs[0]

   
    def get_config_d_prob_d_theta(self, q, rgbd_image):
        '''
        Derivative function for lbfgs/bfgs optimizer.
 
        '''
        grad_q, _ = self.compute_d_prob_d_theta(rgbd_image, q)
        return grad_q

    def quasi_newton_lbfgs_inf(self, rgbd_image, init_hand_config, bfgs=False, 
                            save_grad_to_log=False, object_id=None, grasp_id=None):
        '''
        Quasi Newton inference with bfgs/lbfgs update. 
        '''
        t = time.time()
        q_init = self.convert_full_to_preshape_config(init_hand_config)
        opt_method = 'L-BFGS-B'
        if bfgs:
            opt_method = 'BFGS'
        bnds = []
        for i in range(self.theta_dim):
            bnds.append((self.preshape_config_lower_limit[i], self.preshape_config_upper_limit[i]))

        opt_res = minimize(self.get_config_suc_prob, q_init, jac=self.get_config_d_prob_d_theta, 
                                        args=(rgbd_image,), method=opt_method, bounds=bnds)
        print 'opt_res:', opt_res
        full_grasp_config = self.convert_preshape_to_full_config(opt_res.x)
        
        init_suc_prob = self.get_config_suc_prob(q_init, rgbd_image)

        elapased_time = time.time() - t
        print 'Total inference time: ', str(elapased_time)

        return full_grasp_config, opt_res.fun, init_suc_prob 

    def trans_palm_pose(self, target_frame, palm_pose):
        trans_pose = None
        rate = rospy.Rate(100.0)
        while (not rospy.is_shutdown()) and trans_pose is None:
            try:
                #self.listener.waitForTransform(target_frame, kinect_frame_id, rospy.Time.now(), rospy.Duration(4.0))
                trans_pose = self.listener.transformPose(target_frame, palm_pose)
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
            rate.sleep()
        return trans_pose

    ###################Active Learning for Config-Net################

    def compute_num_grad(self, func, grasp_config, rgbd):
        '''
        Compute numerical gradients d(p(y=1 | theta, o', g=1, w) * p(theta | g=0)) / d theta.
        '''
        eps = 10**-4
        grad = np.zeros(len(grasp_config))
        for i in xrange(len(grasp_config)):
            grasp_config_plus = np.copy(grasp_config)
            grasp_config_plus[i] += eps
            obj_prob_plus = func(grasp_config_plus, rgbd)
            grasp_config_minus = np.copy(grasp_config)
            grasp_config_minus[i] -= eps
            obj_prob_minus = func(grasp_config_minus, rgbd)
            #print 'grasp_config_plus:', grasp_config_plus
            #print 'grasp_config_minus:', grasp_config_minus
            #print 'obj_prob_plus:', obj_prob_plus
            #print 'obj_prob_minus:', obj_prob_minus
            ith_grad = (obj_prob_plus - obj_prob_minus) / (2. * eps)
            grad[i] = ith_grad
        return grad


    def load_grasp_prior(self):
        self.prior_model = pickle.load(open(self.grasp_net.prior_path + 
                                        'gmm.model', 'rb'))


    def compute_grasp_prior(self, grasp_config):
        '''
        Compute the grasp configuration prior.
        '''
        log_prior = self.prior_model.score_samples([grasp_config])[0]
        prior = np.exp(log_prior)
        return prior


    def compute_grasp_log_prior(self, grasp_config):
        '''
        Compute the grasp configuration prior.
        '''
        log_prior = self.prior_model.score_samples([grasp_config])[0]
        log_prior *= self.reg_log_prior
        return log_prior


    def compute_grasp_prior_grad(self, grasp_config):
        '''
        Compute the grasp configuration prior gradient with respect
        to grasp configuration.
        '''
        weighted_log_prob = self.prior_model._estimate_weighted_log_prob(np.array([grasp_config]))[0]
        weighted_prob = np.exp(weighted_log_prob)
        #weighted_prob can also be computed by: 
        #multivariate_normal(mean=g.means_[i], cov=g.covariances_[i], allow_singular=True)
        grad = np.zeros(len(grasp_config))
        for i, w in enumerate(self.prior_model.weights_):
            grad += -weighted_prob[i] * np.matmul(np.linalg.inv(self.prior_model.covariances_[i]), \
                    (grasp_config - self.prior_model.means_[i]))
        grad *= self.reg_log_prior
        return grad


    def compute_active_entropy_obj(self, grasp_config, rgbd):
        '''
        Compute the max entropy for active learning of the generative model, 
        including the likelihood and prior.
        '''
        log_prior = self.compute_grasp_log_prior(grasp_config)
        prior = np.exp(log_prior)

        suc_prob = self.get_config_suc_prob_config_net(grasp_config, rgbd)
        log_suc_prob = np.log(suc_prob)
        suc_entropy = - suc_prob * prior * (log_suc_prob + log_prior)

        failure_prob = 1 - suc_prob
        log_failure_prob = np.log(failure_prob)
        failure_entropy = - failure_prob * prior * (log_failure_prob + log_prior)

        entropy = suc_entropy + failure_entropy
        return entropy 


    def compute_active_entropy_grad(self, grasp_config, rgbd):
        '''
        Compute the max entropy gradient for active learning of the generative model, 
        including the likelihood and prior.
        '''
        log_prior = self.compute_grasp_log_prior(grasp_config)
        prior = np.exp(log_prior)
        prior_grad = self.compute_grasp_prior_grad(grasp_config)
        print 'prior:', prior
        print 'prior_grad:', prior_grad

        #suc_prob = self.get_config_suc_prob_config_net(grasp_config, rgbd) 
        suc_prob_grad , suc_prob = self.compute_d_prob_d_theta_config_net(grasp_config, rgbd)
        print 'suc_prob:', suc_prob
        print 'suc_prob_grad:', suc_prob_grad 
        log_suc_prob = np.log(suc_prob)
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
                                             grasp_config, rgbd)
            grad_diff = entropy_grad - num_grad
            print 'entropy_grad:', entropy_grad
            print 'num_grad:', num_grad
            print 'Gradient difference:', grad_diff
            print 'Gradient difference abs mean:', np.mean(abs(grad_diff))
            print 'Gradient difference abs mean percentage:',  np.mean(abs(grad_diff)) / np.mean(abs(entropy_grad))
            print '~~~~~~~~~~~~~~~~~~~~~~~~~~~'

            print 'entropy_grad:', entropy_grad
            print '################################################################'

        return entropy_grad, entropy


    def compute_likelihood_entropy_obj(self, grasp_config, rgbd):
        '''
        Compute the max entropy for active learning of the likelihood.
        '''
        suc_prob = self.get_config_suc_prob_config_net(grasp_config, rgbd)

        log_suc_prob = np.log(suc_prob)
        suc_entropy = - suc_prob * log_suc_prob

        failure_prob = 1 - suc_prob
        log_failure_prob = np.log(failure_prob)
        failure_entropy = - failure_prob * log_failure_prob

        entropy = suc_entropy + failure_entropy
        return entropy 


    def compute_likelihood_entropy_grad(self, grasp_config, rgbd):
        '''
        Compute the max entropy gradient for active learning of the likelihood.
        '''
        suc_prob = self.get_config_suc_prob_config_net(grasp_config, rgbd)
        log_suc_prob = np.log(suc_prob)
        suc_prob_grad , suc_prob = self.compute_d_prob_d_theta_config_net(grasp_config, rgbd)
        
        #suc_prob_grad *= 0.75
        
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
                                             grasp_config, rgbd)
            grad_diff = entropy_grad - num_grad
            print 'entropy_grad:', entropy_grad
            print 'num_grad:', num_grad
            print 'Gradient difference:', grad_diff
            print 'Gradient difference abs mean:', np.mean(abs(grad_diff))
            print 'Gradient difference abs mean percentage:',  np.mean(abs(grad_diff)) / np.mean(abs(entropy_grad))
            #print '~~~~~~~~~~~~~~~~~~~~~~~~~~~'

            #print 'entropy_grad:', entropy_grad
            print '################################################################'

        return entropy_grad, entropy


    #def active_obj_bfgs(self, grasp_config, rgbd):
    #    '''
    #    Objective function for lbfgs/bfgs optimizer of active learning max entropy.
    #    '''
    #    #entropy = self.compute_active_entropy_obj(grasp_type, latent_voxel, grasp_config)
    #    entropy = self.compute_likelihood_entropy_obj(rgbd, grasp_config)
    #    return -entropy
 

    #def active_grad_bfgs(self, grasp_config, rgbd):
    #    '''
    #    Derivative function for lbfgs/bfgs optimizer of active learning max entropy.
    #    '''
    #    #entropy_grad, _ = self.compute_active_entropy_grad(grasp_type, latent_voxel, grasp_config)
    #    entropy_grad, _ = self.compute_likelihood_entropy_grad(rgbd, grasp_config)
    #    return -entropy_grad


