import time
import tensorflow as tf
import numpy as np
from grasp_prior_network import GraspPriorNetwork
from grasp_data_loader import GraspDataLoader 
import roslib.packages as rp
pkg_path = rp.get_pkg_dir('prob_grasp_planner')
from sklearn.metrics import precision_recall_fscore_support


def train_prior_net(train_data_path, logs_path, prior_model_path, 
                    voxel_ae_model_path, pretrain_voxel_enc,
                    update_voxel_enc, dropout):
    prior_net = GraspPriorNetwork(update_voxel_enc, dropout)
    is_train = True
    prior_net.prior_net_train_test(train_mode=is_train)

    init = tf.global_variables_initializer()
    # Saver to restore the voxel ae model parameters from pretrained model
    # get_collection with 'voxel_ae' doesn't work, I think it is because
    # now I am using tf layers for the prior MDN. It works for the grasp success
    # network, which uses tf low level stuff to build the network. 
    # ae_saver = tf.train.Saver(tf.get_collection(
    #                             tf.GraphKeys.GLOBAL_VARIABLES, 'voxel_ae'))
    ae_saver = tf.train.Saver(prior_net.voxel_ae_vars)

    saver = tf.train.Saver()
    tf.get_default_graph().finalize()

    batch_size = 64
    logs_path_train = logs_path + '/prior_net_train'
    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path_train, graph=tf.get_default_graph())
     
    #learning_rates = {0:0.0001, 1:0.001, 50:0.00001}
    #learning_rates = {0:0.0001, 10:0.001, 100:0.0001, 150:0.00001}
    #learning_rates = {0:0.001, 40:0.0001, 80:0.00001}
    learning_rates = {0:0.001, 30:0.0001, 60:0.00001}
    #learning_rates = {0:0.001, 400:0.0001, 800:0.00001}
    #learning_rates = {0:0.1, 400:0.01, 800:0.001}
    learning_rate = None
    training_epochs = 90
    grasp_loader = GraspDataLoader(train_data_path)
    start_time = time.time()
    iter_num = 0

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        if pretrain_voxel_enc:
            ae_saver.restore(sess, voxel_ae_model_path)
        # Training cycle
        while grasp_loader.epochs_completed < training_epochs:
            if grasp_loader.epochs_completed in learning_rates:
                learning_rate = learning_rates[grasp_loader.epochs_completed]

            grasp_configs, grasp_voxel_grids, grasp_obj_sizes, grasp_labels = \
                    grasp_loader.next_batch(batch_size)
            # Don't need to feed the learning rate of voxel ae, 
            # since only the optimizer in prior_network is used.
            feed_dict = {prior_net.voxel_ae.is_train: (is_train and update_voxel_enc),
                        prior_net.voxel_ae.partial_voxel_grids: grasp_voxel_grids,
                        prior_net.is_train: is_train,
                        prior_net.holder_config: grasp_configs,
                        prior_net.holder_obj_size: grasp_obj_sizes,
                        # prior_net.holder_labels: grasp_labels,
                        prior_net.learning_rate: learning_rate,
                        }
            if dropout:
                feed_dict[prior_net.keep_prob] = 0.9

            [loss, _, train_summary] = sess.run([prior_net.prior_net_res['loss'],
                                                 prior_net.prior_net_res['opt_loss'],
                                                 prior_net.prior_net_res['train_summary'], ],
                                                 feed_dict=feed_dict)
            summary_writer.add_summary(train_summary, iter_num)

            if iter_num % 10 == 0:
                print 'epochs_completed:', grasp_loader.epochs_completed
                print 'iter_num:', iter_num
                print 'learning_rate:', learning_rate
                print 'loss:', loss
                # prfc = precision_recall_fscore_support(grasp_labels, batch_suc_prob)
                # print 'precision, recall, fscore, support:'
                # print prfc

            iter_num += 1
 
        saver.save(sess, prior_model_path)
         
    elapsed_time = time.time() - start_time
    print 'Total training elapsed_time: ', elapsed_time


def test_prior_net(test_data_path, prior_model_path, 
                    update_voxel_enc, dropout):
    prior_net = GraspPriorNetwork(update_voxel_enc, dropout)
    is_train = False
    prior_net.prior_net_train_test(train_mode=is_train)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    tf.get_default_graph().finalize()

    batch_size = 1 #64
    testing_epochs = 1 
    grasp_loader = GraspDataLoader(test_data_path)
    start_time = time.time()
    iter_num = 0
    loss_all = []
    top_labels = []

    # Launch the graph
    with tf.Session() as sess:
        saver.restore(sess, prior_model_path)
        # Training cycle
        while grasp_loader.epochs_completed < testing_epochs:
            (grasp_configs, grasp_voxel_grids, grasp_obj_sizes, grasp_labels), \
                                grasp_top_labels = grasp_loader.next_batch(
                                                    batch_size, top_label=True)
            # Don't need to feed the learning rate of voxel ae, 
            # since only the optimizer in prior_network is used.
            feed_dict = {prior_net.voxel_ae.is_train: (is_train and update_voxel_enc),
                        prior_net.voxel_ae.partial_voxel_grids: grasp_voxel_grids,
                        prior_net.is_train: is_train,
                        prior_net.holder_config: grasp_configs,
                        prior_net.holder_obj_size: grasp_obj_sizes,
                        # prior_net.holder_labels: grasp_labels,
                        }
            if dropout:
                feed_dict[prior_net.keep_prob] = 0.9

            [loss] = sess.run([prior_net.prior_net_res['loss']],
                                feed_dict=feed_dict)

            print 'iter_num:', iter_num
            print 'loss:', loss
            loss_all.append(loss)

            top_labels.append(grasp_top_labels[0][0])

            iter_num += 1

    print 'Average loss:', np.mean(loss_all) 

    # top_labels = top_labels.astype(bool)
    print len(loss_all), len(top_labels)
    top_loss = np.copy(loss_all)[top_labels]
    print 'top_loss.shape', top_loss.shape
    print 'Average loss of top grasps:', np.mean(top_loss)

    side_labels = np.invert(top_labels)
    side_loss = np.copy(loss_all)[side_labels]
    print 'side_loss.shape', side_loss.shape
    print 'Average loss of side grasps:', np.mean(side_loss)

    elapsed_time = time.time() - start_time
    print 'Total testing elapsed_time: ', elapsed_time


if __name__ == '__main__':
    is_train = False #True
    dropout = False #True
    update_voxel_enc = False #True
    pretrain_voxel_enc = True #False

    if is_train:
        # train_data_path = '/mnt/tars_data/gazebo_al_grasps/train/' + \
        #                 'multi_finger_sim_data_6_14/proc_grasp_data.h5'
        # train_data_path = '/mnt/tars_data/gazebo_al_grasps/train/' + \
        #                 'merged_grasp_data_6_6_and_6_8.h5'
        # train_data_path = '/mnt/tars_data/gazebo_al_grasps/train/' + \
        #     'merged_grasp_data_6_6_and_6_8_and_6_10_and_6_11_and_6_13.h5'
        # train_data_path = '/mnt/tars_data/gazebo_al_grasps/train/' + \
        #     'merged_grasp_data_10_sets.h5'
        train_data_path = '/mnt/tars_data/gazebo_al_grasps/train/' + \
            'merged_suc_grasp_10_sets.h5'
        # logs_path = '/home/qingkai/tf_logs/prior_net_logs'
        logs_path = '/home/qingkai/tf_logs/suc_prior_net_logs'
        if update_voxel_enc:
            if pretrain_voxel_enc:
                prior_model_path = pkg_path + '/models/grasp_al_prior/prior_net_update_enc_pre.ckpt'
            else:
                prior_model_path = pkg_path + '/models/grasp_al_prior/prior_net_update_enc_scratch.ckpt'
        else:
            # prior_model_path = pkg_path + '/models/grasp_al_prior/prior_net_freeze_enc_10_sets.ckpt'
            prior_model_path = pkg_path + '/models/grasp_al_prior/suc_prior_net_freeze_enc_10_sets.ckpt'

        voxel_ae_model_path = pkg_path + '/models/voxel_ae/'

        train_prior_net(train_data_path, logs_path, prior_model_path, 
                        voxel_ae_model_path, pretrain_voxel_enc, 
                        update_voxel_enc, dropout)
    else:
        # test_data_path = '/mnt/tars_data/gazebo_al_grasps/test/' + \
        #                 'merged_grasp_data_6_16_and_6_18.h5'
        # test_data_path = '/mnt/tars_data/gazebo_al_grasps/train/' + \
        #     'merged_suc_grasp_10_sets.h5'

        # test_data_path = '/mnt/tars_data/gazebo_al_grasps/test/' + \
        #                 'merged_grasp_data_6_16_and_6_18.h5'
        test_data_path = '/mnt/tars_data/gazebo_al_grasps/train/' + \
            'merged_grasp_data_10_sets.h5'
        # test_data_path = '/mnt/tars_data/gazebo_al_grasps/train/' + \
        #     'merged_grasp_data_6_6_and_6_8_and_6_10_and_6_11_and_6_13.h5'
        if update_voxel_enc:
            if pretrain_voxel_enc:
                prior_model_path = pkg_path + '/models/grasp_al_prior/prior_net_update_enc_pre.ckpt'
            else:
                prior_model_path = pkg_path + '/models/grasp_al_prior/prior_net_update_enc_scratch.ckpt'
        else:
            # prior_model_path = pkg_path + '/models/grasp_al_prior/prior_net_freeze_enc_10_sets.ckpt'
            prior_model_path = pkg_path + '/models/grasp_al_prior/suc_prior_net_freeze_enc_10_sets.ckpt'

        test_prior_net(test_data_path, prior_model_path, 
                        update_voxel_enc, dropout)

