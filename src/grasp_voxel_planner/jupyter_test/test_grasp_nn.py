#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf
import time
import roslib.packages as rp
pkg_path = rp.get_pkg_dir('prob_grasp_planner')
import sys
sys.path.append(pkg_path + '/src/grasp_active_learning')
from grasp_success_network import GraspSuccessNetwork


# In[ ]:


update_voxel_enc = False
grasp_net = GraspSuccessNetwork(update_voxel_enc)
is_train = True
grasp_net.grasp_net_train_test(train_mode=is_train)
#grasp_net.build_grasp_network()
#grasp_net.voxel_ae.build_voxel_ae()


# In[ ]:


init = tf.global_variables_initializer()
ae_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'voxel_ae'))
#bn_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'batch_normalization'))
#saver_global = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
saver = tf.train.Saver()
# ae_bn_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'voxel_ae') + \
#                 tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'batch_normalization')
#ae_bn_saver = tf.train.Saver(ae_bn_vars)
#grasp_ae_saver = tf.train.Saver(grasp_net.voxel_ae_vars)
tf.get_default_graph().finalize()


# In[ ]:


voxel_ae_path = pkg_path + '/models/voxel_ae/voxel_vae_ae_aug.ckpt'
# full_voxel = np.ones((2, 32, 32, 32, 1))
# config_test = np.ones((2, 14))
# labels_test = np.ones((2, 1))
voxel_test = [np.ones((32, 32, 32, 1)), 0.1 * np.ones((32, 32, 32, 1))]
config_test = [np.ones(14), 0.1 * np.ones(14)] 
labels_test = [[1], [0]]
learning_rate = 0.001
logs_path_train = '/home/qingkai/tf_logs/grasp_net_test'
grasp_net_test_freeze_path = '/home/qingkai/test_grasp_net/grasp_net_test_freeze.ckpt'
#grasp_net_test_update_path = '/home/qingkai/test_grasp_net/grasp_net_test_update.ckpt'
summary_writer = tf.summary.FileWriter(logs_path_train, graph=tf.get_default_graph())

with tf.Session() as sess:
    sess.run(init)
    ae_saver.restore(sess, voxel_ae_path)
    for iter_num in xrange(100):
        # Don't need to feed the learning rate of voxel ae, 
        # since only the optimizer in grasp_network is used.
        feed_dict = {grasp_net.voxel_ae.is_train: (is_train and update_voxel_enc),
                    grasp_net.voxel_ae.holder_voxel_grids: voxel_test,
                    grasp_net.is_train: is_train,
                    grasp_net.holder_config: config_test,
                    grasp_net.holder_labels: labels_test,
                    grasp_net.learning_rate: learning_rate,
                    }
        [suc_prob, loss, _, train_summary] =                                             sess.run([grasp_net.grasp_net_res['suc_prob'], 
                                                    #grasp_net.grasp_net_res['voxel_config_concat'],
                                                    grasp_net.grasp_net_res['loss'],
                                                    grasp_net.grasp_net_res['opt_loss'],
                                                    grasp_net.grasp_net_res['train_summary'], ],
                                                    feed_dict=feed_dict)
        summary_writer.add_summary(train_summary, iter_num)
        print suc_prob
        print loss
    saver.save(sess, grasp_net_test_freeze_path)
    #saver.save(sess, grasp_net_test_update_path)


# In[ ]:


voxel_ae_path = pkg_path + '/models/voxel_ae/voxel_vae_ae_aug.ckpt'
full_voxel = np.ones((1, 32, 32, 32, 1))
with tf.Session() as sess:
    sess.run(init)
    ae_saver.restore(sess, voxel_ae_path)
    feed_dict = {grasp_net.voxel_ae.is_train: False,
                grasp_net.voxel_ae.holder_voxel_grids: full_voxel,}
    [embed_output] = sess.run([grasp_net.voxel_ae.ae_struct_res['embedding']],
                             feed_dict=feed_dict)
    print embed_output
    print np.mean(embed_output)


# In[ ]:


grasp_net_test_freeze_path = '/home/qingkai/test_grasp_net/grasp_net_test_freeze.ckpt'
full_voxel = np.ones((1, 32, 32, 32, 1))
with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, grasp_net_test_freeze_path)
    feed_dict = {grasp_net.voxel_ae.is_train: False,
                grasp_net.voxel_ae.holder_voxel_grids: full_voxel,}
    [embed_output] = sess.run([grasp_net.voxel_ae.ae_struct_res['embedding']],
                             feed_dict=feed_dict)
    print embed_output
    print np.mean(embed_output)


# In[ ]:


grasp_net_test_update_path = '/home/qingkai/test_grasp_net/grasp_net_test_update.ckpt'
full_voxel = np.ones((1, 32, 32, 32, 1))
with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, grasp_net_test_update_path)
    feed_dict = {grasp_net.voxel_ae.is_train: False,
                grasp_net.voxel_ae.holder_voxel_grids: full_voxel,}
    [embed_output] = sess.run([grasp_net.voxel_ae.ae_struct_res['embedding']],
                             feed_dict=feed_dict)
    print embed_output
    print np.mean(embed_output)


# In[ ]:


# voxel_ae_path = pkg_path + '/models/voxel_ae/voxel_vae_ae_aug.ckpt'
# full_voxel = np.ones((1, 32, 32, 32, 1))
# grasp_net_test_path = './grasp_net_test.ckpt'
    
# with tf.Session() as sess:
#     sess.run(init)
# #     ae_saver.restore(sess, voxel_ae_path)
# #     bn_saver.restore(sess, voxel_ae_path)
# #     ae_bn_saver.restore(sess, voxel_ae_path)
# #    saver.restore(sess, voxel_ae_path)

#     grasp_ae_saver.restore(sess, voxel_ae_path)
#     feed_dict = {grasp_net.voxel_ae.is_train: False,
#                 grasp_net.voxel_ae.holder_voxel_grids: full_voxel,}
#     [embed_output] = sess.run([grasp_net.voxel_ae.ae_struct_res['embedding']],
#                              feed_dict=feed_dict)
#     print embed_output
#     print np.mean(embed_output)
    
#     ae_saver.restore(sess, voxel_ae_path)
#     feed_dict = {grasp_net.voxel_ae.is_train: False,
#                 grasp_net.voxel_ae.holder_voxel_grids: full_voxel,}
#     [embed_output] = sess.run([grasp_net.voxel_ae.ae_struct_res['embedding']],
#                              feed_dict=feed_dict)
#     print embed_output
#     print np.mean(embed_output)
    
#     #saver.save(sess, grasp_net_test_path)
# #    saver.restore(sess, grasp_net_test_path)
# #     feed_dict = {grasp_net.voxel_ae.is_train: False,
# #                 grasp_net.voxel_ae.holder_voxel_grids: full_voxel,}

# #     [voxel_recons_output, embed_output] = sess.run([grasp_net.voxel_ae.ae_struct_res['voxel_reconstructed'],
# #                                  grasp_net.voxel_ae.ae_struct_res['embedding']],
# #                                  feed_dict=feed_dict)
#     #print voxel_recons_output
#     #print np.mean(voxel_recons_output)
# #     [embed_output] = sess.run([grasp_net.voxel_ae.ae_struct_res['embedding']],
# #                              feed_dict=feed_dict)
# #     print embed_output
# #     print np.mean(embed_output)


# In[ ]:


# var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
# print len(var)
# print tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'voxel_ae')
# print grasp_net.voxel_ae_vars


# In[ ]:


# var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'voxel_ae')
# print len(var)


# In[ ]:


# var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'grasp_net')
# print len(var)


# In[ ]:


tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'grasp_net_loss')


# In[ ]:


tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'grasp_net_loss/voxel_ae_enc_struct/batch_normalization/beta/Momentum')


# In[ ]:




