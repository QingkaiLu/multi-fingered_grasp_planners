# Load and save grasp model and weights in order to use in C++.

import tensorflow as tf
import numpy as np
import sys
import pdb
# import mcubes
# import trimesh
import os
import argparse

from grasp_success_network import GraspSuccessNetwork

_MODEL_FILE = '/home/markvandermerwe/Downloads/models/grasp_al_net/grasp_net_freeze_enc_5_sets.ckpt'
_SAVE_MODEL_PATH = '/home/markvandermerwe/prob_grasp_planner/models/grasp_success/'
_LOGS_PATH = '/home/markvandermerwe/prob_grasp_planner/models/grasp_success/'

def save_model(model_file, save_model_path, logs_path):

    # Build the grasp network.
    update_voxel_enc = False
    dropout = True
    grasp_net = GraspSuccessNetwork(update_voxel_enc, dropout)
    grasp_net.grasp_net_train_test(train_mode=False)

    # Take gradients.
    palm_pose_gradient = tf.gradients(grasp_net.grasp_net_res['suc_prob'], grasp_net.holder_config, name='palm_pose_gradient')

    pdb.set_trace()
    
    # Restore model from weights.
    saver = tf.train.Saver()
    tf.get_default_graph().finalize()

    with tf.Session() as sess:

        # Restore variables.
        saver.restore(sess, model_file)
        print("Model restored from: ", model_file)

        # is_train = False
        # grasp_voxel_grids = np.zeros((1,32,32,32,1))
        # grasp_configs = np.zeros((1,14))
        # grasp_obj_sizes = np.zeros((1,3))

        # feed_dict = {grasp_net.voxel_ae.is_train: is_train,
        #              grasp_net.voxel_ae.partial_voxel_grids: grasp_voxel_grids,
        #              grasp_net.is_train: is_train,
        #              grasp_net.holder_config: grasp_configs,
        #              grasp_net.holder_obj_size: grasp_obj_sizes,
        #              grasp_net.keep_prob: 1.0
        # }

        pdb.set_Trace()
        # [batch_suc_prob] = sess.run([grasp_net.grasp_net_res['suc_prob'],], 
        #                             feed_dict=feed_dict)
        # print batch_suc_prob

        # grad = sess.run(palm_pose_gradient,
        #                 feed_dict=feed_dict)
        # print grad
        
        # Setup tensorboard. This is to have the session graph for reference.
        f_writer = tf.summary.FileWriter(logs_path, sess.graph)

        # (Re)Save model weights.
        model_ckpt = os.path.join(save_model_path, 'model.ckpt')
        saver.save(sess, model_ckpt)
        print("Saved model to: ", model_ckpt)

        # Save graph.
        graph_file = os.path.join(save_model_path, 'graph.pb')
        tf.train.write_graph(sess.graph, '.', graph_file)
        print("Saved graph to: ", graph_file)

if __name__ == '__main__':
    # Save away.
    save_model(_MODEL_FILE,
               _SAVE_MODEL_PATH,
               _LOGS_PATH)
