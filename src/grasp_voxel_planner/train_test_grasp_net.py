import time
import tensorflow as tf
import numpy as np
from grasp_success_network import GraspSuccessNetwork
from grasp_data_loader import GraspDataLoader 
import roslib.packages as rp
pkg_path = rp.get_pkg_dir('prob_grasp_planner')
from sklearn.metrics import precision_recall_fscore_support


def train_grasp_net(train_data_path, logs_path, grasp_model_path, 
                    voxel_ae_model_path, update_voxel_enc, dropout):
    grasp_net = GraspSuccessNetwork(update_voxel_enc, dropout)
    is_train = True
    grasp_net.grasp_net_train_test(train_mode=is_train)

    init = tf.global_variables_initializer()
    # Saver to restore the voxel ae model parameters from pretrained model
    ae_saver = tf.train.Saver(tf.get_collection(
                                tf.GraphKeys.GLOBAL_VARIABLES, 'voxel_ae'))
    saver = tf.train.Saver()
    tf.get_default_graph().finalize()

    batch_size = 128 #64
    logs_path_train = logs_path + '/grasp_net_train'
    # logs_path_train = logs_path + '/grasp_net_train_focal'
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
    binary_threshold = 0.5

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        ae_saver.restore(sess, voxel_ae_model_path)
        # Training cycle
        while grasp_loader.epochs_completed < training_epochs:
            if grasp_loader.epochs_completed in learning_rates:
                learning_rate = learning_rates[grasp_loader.epochs_completed]

            grasp_configs, grasp_voxel_grids, grasp_obj_sizes, grasp_labels = \
                    grasp_loader.next_batch(batch_size)
            # Don't need to feed the learning rate of voxel ae, 
            # since only the optimizer in grasp_network is used.
            feed_dict = {grasp_net.voxel_ae.is_train: (is_train and update_voxel_enc),
                        grasp_net.voxel_ae.partial_voxel_grids: grasp_voxel_grids,
                        grasp_net.is_train: is_train,
                        grasp_net.holder_config: grasp_configs,
                        grasp_net.holder_obj_size: grasp_obj_sizes,
                        grasp_net.holder_labels: grasp_labels,
                        grasp_net.learning_rate: learning_rate,
                        }
            if dropout:
                feed_dict[grasp_net.keep_prob] = 0.9

            [batch_suc_prob, loss, _, train_summary] = \
                                            sess.run([grasp_net.grasp_net_res['suc_prob'], 
                                                    grasp_net.grasp_net_res['loss'],
                                                    grasp_net.grasp_net_res['opt_loss'],
                                                    grasp_net.grasp_net_res['train_summary'], ],
                                                    feed_dict=feed_dict)
            summary_writer.add_summary(train_summary, iter_num)

            if iter_num % 10 == 0:
                print 'epochs_completed:', grasp_loader.epochs_completed
                print 'iter_num:', iter_num
                print 'learning_rate:', learning_rate
                print 'loss:', loss
                # print 'batch_suc_prob:', batch_suc_prob
                print 'suc_num:', np.sum(batch_suc_prob > 0.5)
                print 'true suc_num:', np.sum(grasp_labels)
                batch_suc_prob[batch_suc_prob > binary_threshold] = 1.
                batch_suc_prob[batch_suc_prob <= binary_threshold] = 0.
                #avg_pred_error = np.mean(np.abs(batch_suc_prob - grasp_labels))
                #print 'avg_pred_error:', avg_pred_error
                prfc = precision_recall_fscore_support(grasp_labels, batch_suc_prob)
                print 'precision, recall, fscore, support:'
                print prfc


            iter_num += 1
 
        saver.save(sess, grasp_model_path)
         
    elapsed_time = time.time() - start_time
    print 'Total training elapsed_time: ', elapsed_time


def test_grasp_net(test_data_path, grasp_model_path, 
                    update_voxel_enc, dropout):
    grasp_net = GraspSuccessNetwork(update_voxel_enc, dropout)
    is_train = False
    grasp_net.grasp_net_train_test(train_mode=is_train)

    saver = tf.train.Saver()
    tf.get_default_graph().finalize()

    batch_size = 1 #64
    testing_epochs = 1
    grasp_loader = GraspDataLoader(test_data_path)
    start_time = time.time()
    iter_num = 0
    pred_suc_probs = np.array([]).reshape(0, 1)
    true_suc_labels = np.array([]).reshape(0, 1)
    top_labels = np.array([]).reshape(0, 1)

    # Launch the graph
    with tf.Session() as sess:
        saver.restore(sess, grasp_model_path)
        # Training cycle
        while grasp_loader.epochs_completed < testing_epochs:
            (grasp_configs, grasp_voxel_grids, grasp_obj_sizes, grasp_labels), \
                                grasp_top_labels = grasp_loader.next_batch(
                                                    batch_size, top_label=True)
            # Don't need to feed the learning rate of voxel ae, 
            # since only the optimizer in grasp_network is used.
            feed_dict = {grasp_net.voxel_ae.is_train: is_train,
                        grasp_net.voxel_ae.partial_voxel_grids: grasp_voxel_grids,
                        grasp_net.is_train: is_train,
                        grasp_net.holder_config: grasp_configs,
                        grasp_net.holder_obj_size: grasp_obj_sizes,
                        }
            if dropout:
                feed_dict[grasp_net.keep_prob] = 1.

            [batch_suc_prob] = sess.run([grasp_net.grasp_net_res['suc_prob'],], 
                                        feed_dict=feed_dict)
            pred_suc_probs = np.concatenate((pred_suc_probs, batch_suc_prob))
            true_suc_labels = np.concatenate((true_suc_labels, grasp_labels))
            top_labels = np.concatenate((top_labels, grasp_top_labels))

            iter_num += 1
            print 'iter num:', iter_num

    binary_threshold = 0.5

    print pred_suc_probs[:-10]
    print true_suc_labels[:-10]
    pred_suc_labels = np.copy(pred_suc_probs)
    pred_suc_labels[pred_suc_labels > binary_threshold] = 1.
    pred_suc_labels[pred_suc_labels <= binary_threshold] = 0.
    print 'pred_suc_labels.shape: ', pred_suc_labels.shape
    print 'pred success #:', np.sum(pred_suc_labels)
    print 'true_suc_labels.shape: ', true_suc_labels.shape
    print 'true success #:', np.sum(true_suc_labels)
    pred_errors_all = np.abs(pred_suc_labels - true_suc_labels) 
    test_acc = 1. - np.mean(pred_errors_all)
    print('test accuracy:', test_acc)

    prfc = precision_recall_fscore_support(true_suc_labels, pred_suc_labels)
    print 'precision, recall, fscore, support:'
    print prfc

    top_labels = top_labels.astype(bool)
    pred_suc_labels_top = np.copy(pred_suc_probs)[top_labels]
    true_suc_labels_top = true_suc_labels[top_labels]
    pred_suc_labels_top[pred_suc_labels_top > binary_threshold] = 1.
    pred_suc_labels_top[pred_suc_labels_top <= binary_threshold] = 0.
    print 'pred_suc_labels_top.shape: ', pred_suc_labels_top.shape
    print 'top pred success #:', np.sum(pred_suc_labels_top)
    print 'true_suc_labels_top.shape: ', true_suc_labels_top.shape
    print 'top true success #:', np.sum(true_suc_labels_top)
    pred_errors_top = np.abs(pred_suc_labels_top - true_suc_labels_top) 
    test_acc_top = 1. - np.mean(pred_errors_top)
    print('test accuracy of top grasps:', test_acc_top)

    prfc_top = precision_recall_fscore_support(true_suc_labels_top, 
                                            pred_suc_labels_top)
    print 'precision, recall, fscore, support of top grasps:'
    print prfc_top

    side_labels = np.invert(top_labels)
    pred_suc_labels_side = np.copy(pred_suc_probs)[side_labels]
    true_suc_labels_side = true_suc_labels[side_labels]
    pred_suc_labels_side[pred_suc_labels_side > binary_threshold] = 1.
    pred_suc_labels_side[pred_suc_labels_side <= binary_threshold] = 0.
    print 'pred_suc_labels_side.shape: ', pred_suc_labels_side.shape
    print 'side pred success #:', np.sum(pred_suc_labels_side)
    print 'true_suc_labels_side.shape: ', true_suc_labels_side.shape
    print 'side true success #:', np.sum(true_suc_labels_side)
    pred_errors_side = np.abs(pred_suc_labels_side - true_suc_labels_side) 
    test_acc_side = 1. - np.mean(pred_errors_side)
    print('test accuracy of side grasps:', test_acc_side)

    prfc_side = precision_recall_fscore_support(true_suc_labels_side, 
                                            pred_suc_labels_side)
    print 'precision, recall, fscore, support of side grasps:'
    print prfc_side

    elapsed_time = time.time() - start_time
    print 'Total testing elapsed_time: ', elapsed_time


if __name__ == '__main__':
    is_train = False #True
    dropout = False #True
    if is_train:
        # train_data_path = '/mnt/tars_data/gazebo_al_grasps/train/' + \
        #                 'multi_finger_sim_data_6_14/proc_grasp_data.h5'
        # train_data_path = '/mnt/tars_data/gazebo_al_grasps/train/' + \
        #                 'merged_grasp_data_6_6_and_6_8.h5'
        train_data_path = '/mnt/tars_data/gazebo_al_grasps/train/' + \
            'merged_grasp_data_6_6_and_6_8_and_6_10_and_6_11_and_6_13.h5'
        # train_data_path = '/mnt/tars_data/gazebo_al_grasps/train/' + \
        #     'merged_grasp_data_10_sets.h5'
        logs_path = '/home/qingkai/tf_logs/grasp_net_logs_5_sets/'
        update_voxel_enc = False #True
        if update_voxel_enc:
            grasp_model_path = pkg_path + '/models/grasp_al_net/grasp_net_update_enc.ckpt'
        else:
            # grasp_model_path = pkg_path + '/models/grasp_al_net/grasp_net_freeze_enc.ckpt'
            # grasp_model_path = pkg_path + '/models/grasp_al_net/grasp_net_freeze_enc_10_sets.ckpt'
            grasp_model_path = pkg_path + '/models/grasp_al_net/grasp_net_5_sets.ckpt'

        voxel_ae_model_path = pkg_path + '/models/voxel_ae/'

        train_grasp_net(train_data_path, logs_path, grasp_model_path, 
                        voxel_ae_model_path, update_voxel_enc, dropout)
    else:
        test_data_path = '/mnt/tars_data/gazebo_al_grasps/test/' + \
                        'merged_grasp_data_6_16_and_6_18.h5'

        # test_data_path = '/mnt/tars_data/gazebo_al_grasps/train/' + \
        #     'merged_grasp_data_10_sets.h5'

        #test_data_path = '/mnt/tars_data/gazebo_al_grasps/train/' + \
        #    'merged_grasp_data_6_6_and_6_8_and_6_10_and_6_11_and_6_13.h5'


        update_voxel_enc = False #True
        if update_voxel_enc:
            grasp_model_path = pkg_path + '/models/grasp_al_net/grasp_net_update_enc.ckpt'
        else:
            # grasp_model_path = pkg_path + '/models/grasp_al_net/grasp_net_freeze_enc_10_sets.ckpt'
            # grasp_model_path = pkg_path + '/models/grasp_al_net/'+ \
            #                             'grasp_net_freeze_enc_2_sets_dropout_90.ckpt'
            # grasp_model_path = pkg_path + '/models/grasp_al_net/' + \
            #                'grasp_net_freeze_enc_2_sets.ckpt'
            grasp_model_path = pkg_path + '/models/grasp_al_net/grasp_net_5_sets.ckpt'

        test_grasp_net(test_data_path, grasp_model_path, 
                        update_voxel_enc, dropout)

