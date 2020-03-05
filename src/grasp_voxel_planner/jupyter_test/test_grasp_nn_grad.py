#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import time
import roslib.packages as rp
pkg_path = rp.get_pkg_dir('prob_grasp_planner')
import sys
sys.path.append(pkg_path + '/src/grasp_active_learning')
from grasp_success_network import GraspSuccessNetwork


# In[2]:


update_voxel_enc = False
grasp_net = GraspSuccessNetwork(update_voxel_enc)
is_train = False
grasp_net.grasp_net_train_test(train_mode=is_train)
config_grad = tf.gradients(grasp_net.grasp_net_res['suc_prob'], grasp_net.holder_config)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
tf.get_default_graph().finalize()


# In[6]:


grasp_model_path = pkg_path + '/models/grasp_al_net/grasp_net_freeze_enc.ckpt'
sess = tf.Session()
saver.restore(sess, grasp_model_path)


# In[23]:


def pred_suc(sess, grasp_voxel_grid, grasp_config, grasp_obj_size):
    feed_dict = {
        grasp_net.voxel_ae.is_train: False,
        grasp_net.voxel_ae.partial_voxel_grids: [grasp_voxel_grid],
        grasp_net.is_train: False,
        grasp_net.holder_config: [grasp_config],
        grasp_net.holder_obj_size: [grasp_obj_size],
 }
    [suc_prob] = sess.run([grasp_net.grasp_net_res['suc_prob']], feed_dict=feed_dict)
    return suc_prob[0][0]


# In[37]:


def compute_config_grad(sess, grasp_voxel_grid, grasp_config, grasp_obj_size):
    feed_dict = {
        grasp_net.voxel_ae.is_train: False,
        grasp_net.voxel_ae.partial_voxel_grids: [grasp_voxel_grid],
        grasp_net.is_train: False,
        grasp_net.holder_config: [grasp_config],
        grasp_net.holder_obj_size: [grasp_obj_size],
 }
    [suc_prob, config_gradient] = sess.run([grasp_net.grasp_net_res['suc_prob'], config_grad], feed_dict=feed_dict)
    return config_gradient[0][0], suc_prob[0][0]


# In[39]:


def compute_num_grad(func, sess, grasp_voxel_grid, grasp_config, grasp_obj_size):
    eps = 10**-4
    grad = np.zeros(len(grasp_config))
    for i in xrange(len(grasp_config)):
        grasp_config_plus = np.copy(grasp_config)
        grasp_config_plus[i] += eps
        obj_prob_plus = func(sess, grasp_voxel_grid, grasp_config_plus, grasp_obj_size)
        grasp_config_minus = np.copy(grasp_config)
        grasp_config_minus[i] -= eps
        obj_prob_minus = func(sess, grasp_voxel_grid, grasp_config_minus, grasp_obj_size)
        #print 'grasp_config_plus:', grasp_config_plus
        #print 'grasp_config_minus:', grasp_config_minus
        #print 'obj_prob_plus:', obj_prob_plus
        #print 'obj_prob_minus:', obj_prob_minus
        ith_grad = (obj_prob_plus - obj_prob_minus) / (2. * eps)
        grad[i] = ith_grad
    return grad


# In[41]:




grasp_voxel_grid = 0.5 * np.ones((32, 32, 32, 1))
grasp_config = 0.5 * np.ones(14)
grasp_obj_size = [0.1, 0.1, 0.1]
suc_prob = pred_suc(sess, grasp_voxel_grid, grasp_config, grasp_obj_size)
print suc_prob
config_gradient, suc_prob = compute_config_grad(sess, grasp_voxel_grid, grasp_config, grasp_obj_size)
print config_gradient, suc_prob
config_num_grad = compute_num_grad(pred_suc, sess, grasp_voxel_grid, grasp_config, grasp_obj_size)
print config_num_grad
grad_diff = config_gradient - config_num_grad
print 'config_gradient:', config_gradient
print 'config_num_grad:', config_num_grad
print 'Gradient difference:', grad_diff
print 'Gradient difference abs mean:', np.mean(abs(grad_diff))
print 'Gradient difference abs mean percentage:',  np.mean(abs(grad_diff)) / np.mean(abs(config_gradient))

# grasp_net_test_freeze_path = '/home/qingkai/test_grasp_net/grasp_net_test_freeze.ckpt'
# full_voxel = np.ones((1, 32, 32, 32, 1))
# with tf.Session() as sess:
#     sess.run(init)
#     saver.restore(sess, grasp_net_test_freeze_path)


# In[ ]:




