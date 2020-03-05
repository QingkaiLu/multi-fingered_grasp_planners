#!/usr/bin/env python
import numpy as np
import os
import time
import cv2
import h5py
import matplotlib.pyplot as plt
import roslib.packages as rp
from geometry_msgs.msg import Pose, Quaternion, PoseStamped
from prob_grasp_planner.msg import VisualInfo, HandConfig
from scipy.optimize import fmin_bfgs
from scipy.optimize import fmin_l_bfgs_b
from scipy.optimize import minimize
import sys
pkg_path = rp.get_pkg_dir('prob_grasp_planner')
sys.path.append(pkg_path + '/src/grasp_type_planner')
from grasp_pgm_inference import GraspPgmInfer
sys.path.append(pkg_path + '/src/grasp_cnn_planner')
#from grasp_rgbd_inf import GraspInf
import rospy
import tf
from sensor_msgs.msg import JointState


class GraspActiveLearner:
    '''
    Active grasp learner.
    '''

    def __init__(self, grasp_planner_name, vis_preshape, 
                 use_hd=True, virtual_hand_parent_tf=''):
        #rospy.init_node('grasp_active_learner')
        #TODO: the grasp type pgm and the grasp cnn class have a lot of common functions, 
        #such as the setup_config_limits() and convert_preshape_to_full_config(), 
        #need to move these common functions into a seperate class.  
        self.grasp_planner_name = grasp_planner_name
        if self.grasp_planner_name == 'grasp_type_pgm':
            self.grasp_model = GraspPgmInfer(pgm_grasp_type=True)
        elif self.grasp_planner_name == 'grasp_config_net':
            self.grasp_model = GraspInf(config_net=True, use_hd=use_hd)

        self.vis_preshape = vis_preshape
        if self.vis_preshape:
            self.tf_br = tf.TransformBroadcaster()
            self.js_pub = rospy.Publisher('/virtual_hand/allegro_hand_right/joint_states', 
                                          JointState, queue_size=1)
            self.preshape_config = None
        self.virtual_hand_parent_tf = virtual_hand_parent_tf


    def pub_preshape_config(self):
        if self.preshape_config is not None:
            for i in xrange(2): 
                preshape_pose = self.preshape_config.palm_pose
                self.tf_br.sendTransform((preshape_pose.pose.position.x, preshape_pose.pose.position.y,
                                          preshape_pose.pose.position.z), (preshape_pose.pose.orientation.x, 
                                          preshape_pose.pose.orientation.y, preshape_pose.pose.orientation.z, 
                                          preshape_pose.pose.orientation.w), 
                                        rospy.Time.now(), '/virtual_hand/allegro_mount', 
                                        '/virtual_hand/' + self.virtual_hand_parent_tf)
                self.js_pub.publish(self.preshape_config.hand_joint_state)
                rospy.sleep(0.5)


    def active_max_uncertainty_lbfgs(self, grasp_type, object_voxel, init_hand_config, 
                               init_frame_id=None, bfgs=True):
        '''
        Max entropy for active learning with bfgs/lbfgs update. 
        '''
        #al_strategy = 'clf_uncertainty'
        #al_strategy = 'prior'
        #al_strategy = 'grasp_success'
        al_strategy = 'grasp_clf_success'
        #al_strategy = 'MAB'
        self.grasp_model.al_strategy = al_strategy

        t = time.time()
        #self.grasp_model.hand_config_frame_id = init_hand_config.palm_pose.header.frame_id
        #q_init = self.grasp_model.convert_full_to_preshape_config(init_hand_config)
        config_sample, _ = self.grasp_model.sample_grasp_config(grasp_type)
        q_init = config_sample[0]
        #q_init = init_hand_config
        print 'q_init:', q_init

        self.grasp_model.hand_config_frame_id = init_frame_id
        
        voxel_grid_dim = object_voxel.shape
        voxel_num = voxel_grid_dim[0] * voxel_grid_dim[1] * voxel_grid_dim[2]  
        grasp_1d_voxel = np.reshape(object_voxel, voxel_num)
        latent_voxel = self.grasp_model.pca_model.transform([grasp_1d_voxel])[0]

        #self.setup_palm_pose_limits(q_init)

        opt_method = 'L-BFGS-B'
        if bfgs:
            opt_method = 'BFGS'

        #opt_method = 'TNC'

        bnds = []
        for i in xrange(self.grasp_model.palm_dof_dim):
            bnds.append((-float('inf'), float('inf')))
        for i in xrange(self.grasp_model.palm_dof_dim, self.grasp_model.config_dim):
            bnds.append((self.grasp_model.preshape_config_lower_limit[i], 
                         self.grasp_model.preshape_config_upper_limit[i]))

        #Notice this is for gradient descent, not ascent.
        opt_res = minimize(self.grasp_model.active_obj_bfgs, q_init, jac=self.grasp_model.active_grad_bfgs, 
                                        args=(grasp_type, latent_voxel,), method=opt_method, bounds=bnds)
        print 'opt_res with constraints:', opt_res

        elapased_time = time.time() - t
        print 'Total inference time: ', str(elapased_time)

        full_grasp_config = self.grasp_model.convert_preshape_to_full_config(opt_res.x)
        
        init_obj_val = self.grasp_model.active_obj_bfgs(q_init, grasp_type, latent_voxel)
        print 'init_obj_val:', init_obj_val
        init_config = self.grasp_model.convert_preshape_to_full_config(q_init)
        
        if al_strategy == 'clf_uncertainty' or al_strategy == 'grasp_success' \
                or al_strategy == 'grasp_clf_success':
            inf_obj_val = -opt_res.fun
            init_obj_val = -init_obj_val
        elif al_strategy == 'prior':
            inf_obj_val = opt_res.fun

        return full_grasp_config, inf_obj_val, init_obj_val, init_config 


    def find_entropy_rate_bt(self, alpha, q, suc_prob, grad_q, 
                             grasp_type=None, latent_voxel=None, 
                             grasp_rgbd_patch=None, line_search_log=None,
                             use_talor=False):
        '''
        Backtracking line search to find the learning rate of max entropy.
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
        q_new = self.grasp_model.project_config(q_new)
        if line_search_log is not None:
            line_search_log.writelines('q_new after projection: ' + str(q_new))
            line_search_log.writelines('\n')
        if self.grasp_planner_name == 'grasp_type_pgm':
            suc_prob_new = self.grasp_model.compute_likelihood_entropy_obj(
                                                grasp_type, latent_voxel, q_new)
        elif self.grasp_planner_name == 'grasp_config_net':
            #suc_prob_new = self.grasp_model.compute_likelihood_entropy_obj(
            #                                    q_new, grasp_rgbd_patch)
            suc_prob_new = self.grasp_model.compute_active_entropy_obj(
                                                q_new, grasp_rgbd_patch)
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
            q_new = self.grasp_model.project_config(q_new)
            if line_search_log is not None:
                line_search_log.writelines('q_new after projection: ' + str(q_new))
                line_search_log.writelines('\n')
            if self.grasp_planner_name == 'grasp_type_pgm':
                suc_prob_new = self.grasp_model.compute_likelihood_entropy_obj(
                                                    grasp_type, latent_voxel, q_new)
            elif self.grasp_planner_name == 'grasp_config_net':
                #suc_prob_new = self.grasp_model.compute_likelihood_entropy_obj(
                #                                    q_new, grasp_rgbd_patch)
                suc_prob_new = self.grasp_model.compute_active_entropy_obj(
                                                    q_new, grasp_rgbd_patch)
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


    def gd_entropy(self, init_hand_config, grasp_type=None, 
                   object_voxel=None, grasp_rgbd_patch=None,  
                   save_grad_to_log=False, object_id=None, grasp_id=None):
        '''
        Gradient descent with line search for max entropy of one grasp type. 
        '''
        self.grasp_model.hand_config_frame_id = init_hand_config.palm_pose.header.frame_id
        q = self.grasp_model.convert_full_to_preshape_config(init_hand_config)
        #q = init_hand_config
        #self.grasp_model.hand_config_frame_id = 'grasp_object'
        #print 'init_config:', q

        #self.grasp_model.setup_palm_pose_limits(q)

        if self.grasp_planner_name == 'grasp_type_pgm':
            voxel_grid_dim = object_voxel.shape
            voxel_num = voxel_grid_dim[0] * voxel_grid_dim[1] * voxel_grid_dim[2]  
            grasp_1d_voxel = np.reshape(object_voxel, voxel_num)
            latent_voxel = self.grasp_model.pca_model.transform([grasp_1d_voxel])[0]

        if save_grad_to_log: 
            config_grad_path = self.grasp_model.grasp_config_log_save_path + 'object_' + str(object_id) \
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
        if self.grasp_model.log_inf:
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
        for iter_num in xrange(self.grasp_model.iter_total_num):
            #print 'iter:', iter_num
            if self.grasp_planner_name == 'grasp_type_pgm':
                grad_q, suc_prob = self.grasp_model.compute_likelihood_entropy_grad(
                                                    grasp_type, latent_voxel, q)
            elif self.grasp_planner_name == 'grasp_config_net':
                #grad_q, suc_prob = self.grasp_model.compute_likelihood_entropy_grad(
                #                                    q, grasp_rgbd_patch)
                grad_q, suc_prob = self.grasp_model.compute_active_entropy_grad(
                                                    q, grasp_rgbd_patch)

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
                #log_file.writelines('clf_log_prob: ' + str(clf_log_prob))
                #log_file.writelines('\n')
                #log_file.writelines('log_prior: ' + str(log_prior))
                #log_file.writelines('\n')
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
            if self.grasp_planner_name == 'grasp_type_pgm':
                q_learn_rate = self.find_entropy_rate_bt(bt_learn_rate, q, suc_prob, grad_q,
                                                         grasp_type=grasp_type, latent_voxel=latent_voxel,  
                                                         line_search_log=line_search_log, use_talor=use_talor)
            elif self.grasp_planner_name == 'grasp_config_net':
                q_learn_rate = self.find_entropy_rate_bt(bt_learn_rate, q, suc_prob, grad_q,
                                                         grasp_rgbd_patch=grasp_rgbd_patch,
                                                         line_search_log=line_search_log, use_talor=use_talor)

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
            q_update = self.grasp_model.project_config(q_update)
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

            if self.vis_preshape:
                self.preshape_config = self.grasp_model.convert_preshape_to_full_config(q)
                self.pub_preshape_config()

        
        suc_probs = np.array(suc_probs)
        if save_grad_to_log: 
            plt.figure()
            plt.plot(suc_probs, label='suc')
            if self.grasp_model.log_inf:
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

        full_grasp_config = self.grasp_model.convert_preshape_to_full_config(q)

        #print init_hand_config 
        #print full_grasp_config, suc_probs[-1], suc_probs[0]

        return full_grasp_config, suc_probs[-1], suc_probs[0]
        
        #return q, suc_probs[-1], suc_probs[0]


if __name__ == '__main__':
    prec_data_path = '/mnt/tars_data/multi_finger_sim_data_precision_1/'
    #voxel_path = prec_data_path + 'prec_grasps/prec_align_suc_grasps.h5'
    voxel_path = prec_data_path + 'prec_grasps/prec_align_failure_grasps.h5'
    #voxel_path = '/dataspace/data_kai/test_grasp_inf/power_failure_grasp_voxel_data.h5'
    grasp_voxel_file = h5py.File(voxel_path, 'r')
    grasp_voxel_grids = grasp_voxel_file['grasp_voxel_grids'][()] 
    grasp_configs = grasp_voxel_file['grasp_configs_obj'][()]
    grasp_active_learner = GraspActiveLearner(grasp_planner_name='grasp_type_pgm', vis_preshape=False)
    grasp_active_learner.active_max_uncertainty_lbfgs('prec', grasp_voxel_grids[4], grasp_configs[4], bfgs=False)
    #print grasp_active_learner.active_max_uncertainty_lbfgs('prec', np.ones(grasp_voxel_grids[4].shape), 
    #                                             np.ones(grasp_configs[4].shape), bfgs=False)
    #print grasp_active_learner.gd_entropy(grasp_configs[0], grasp_type='prec', 
    #                                      object_voxel=grasp_voxel_grids[0],
    #                                      save_grad_to_log=False, object_id=-1, grasp_id=0)
    #print grasp_active_learner.grasp_model.gd_inf_one_type(
    #                        'prec', grasp_voxel_grids[0], grasp_configs[0],
    #                        save_grad_to_log=False, object_id=-1, grasp_id=0)
    #print grasp_configs[0]
    grasp_voxel_file.close()

