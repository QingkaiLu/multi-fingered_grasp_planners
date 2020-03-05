import time
import tensorflow as tf
import numpy as np
from grasp_rgbd_config_net import GraspRgbdConfigNet
from grasp_rgbd_loader import GraspRgbdLoader 
import roslib.packages as rp
pkg_path = rp.get_pkg_dir('prob_grasp_planner')
from sklearn.metrics import precision_recall_fscore_support
from gen_rgbd_gazebo_kinect import GenRgbdGazeboKinect


def train_config_net(train_data_path, logs_path, grasp_model_path): 
    grasp_net = GraspRgbdConfigNet()

    #Create the variables.
    grasp_net.create_net_var()
    #Call the network building function.
    grasp_net.cost_function()

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    tf.get_default_graph().finalize()

    batch_size = 8 #16 #64
    logs_path_train = logs_path + '/config_net_train'
    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path_train, graph=tf.get_default_graph())
     
    learning_rates = {0:0.001, 30:0.0001, 60:0.00001}
    learning_rate = None
    training_epochs = 90
    grasp_loader = GraspRgbdLoader(train_data_path)
    start_time = time.time()
    iter_num = 0
    binary_threshold = 0.4 #0.5
    oversample_suc_num = 2

    # gen_rgbd = GenRgbdGazeboKinect() 

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        # Training cycle
        while grasp_loader.epochs_completed < training_epochs:
            if grasp_loader.epochs_completed in learning_rates:
                learning_rate = learning_rates[grasp_loader.epochs_completed]

            grasp_configs, grasp_rgbd_patches, grasp_labels = \
                    grasp_loader.next_batch(batch_size, 
                            oversample_suc_num=oversample_suc_num)
            feed_dict = {
                        grasp_net.holder_config: grasp_configs,
                        grasp_net.holder_grasp_patch: grasp_rgbd_patches,
                        grasp_net.holder_labels: grasp_labels,
                        grasp_net.learning_rate: learning_rate,
                        grasp_net.keep_prob: 0.75 
                        }

            # gen_rgbd.save_rgbd_image(grasp_rgbd_patches[0], 
            #        '/home/qingkai/Workspace/grasp_rgbd')
            # raw_input('Hold')

            [batch_suc_prob, loss, _, train_summary] = \
                                            sess.run([grasp_net.pred, 
                                                    grasp_net.cost,
                                                    grasp_net.optimizer,
                                                    grasp_net.train_summary, ],
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


def test_config_net(test_data_path, grasp_model_path): 
    grasp_net = GraspRgbdConfigNet()

    #Create the variables.
    grasp_net.create_net_var()
    #Call the network building function.
    grasp_net.cost_function()

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    tf.get_default_graph().finalize()

    batch_size = 1
    testing_epochs = 1
    grasp_loader = GraspRgbdLoader(test_data_path)
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
            (grasp_configs, grasp_rgbd_patches, grasp_labels), \
                                grasp_top_labels = grasp_loader.next_batch(
                                                    batch_size, top_label=True)
            feed_dict = {
                        grasp_net.holder_config: grasp_configs,
                        grasp_net.holder_grasp_patch: grasp_rgbd_patches,
                        grasp_net.holder_labels: grasp_labels,
                        grasp_net.keep_prob: 1. 
                        }

            [batch_suc_prob] = sess.run([grasp_net.pred,], 
                                        feed_dict=feed_dict)
            pred_suc_probs = np.concatenate((pred_suc_probs, batch_suc_prob))
            print pred_suc_probs
            true_suc_labels = np.concatenate((true_suc_labels, grasp_labels))
            top_labels = np.concatenate((top_labels, grasp_top_labels))

            iter_num += 1
            print 'iter num:', iter_num

    binary_threshold = 0.5 #0.4

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
    if is_train:
        train_data_path = '/mnt/tars_data/gazebo_al_grasps/train_isrr/' + \
             'merged_grasp_rgbd_10_sets.h5'
        # logs_path = '/home/qingkai/tf_logs/config_net_logs/'
        # grasp_model_path = pkg_path + '/models/grasp_al_net/rgbd_config_net.ckpt'
        logs_path = '/home/qingkai/tf_logs/config_net_logs_os/'
        grasp_model_path = pkg_path + '/models/grasp_al_net/rgbd_config_net_os.ckpt'

        # grasp_loader = GraspRgbdLoader(train_data_path)
        # print grasp_loader.next_batch(2)

        train_config_net(train_data_path, logs_path, grasp_model_path) 

    else:
        test_data_path = '/mnt/tars_data/gazebo_al_grasps/test_isrr/' + \
                        'merged_grasp_rgbd_test_sets.h5'       
        grasp_model_path = pkg_path + '/models/grasp_al_net/rgbd_config_net.ckpt'
        # grasp_model_path = pkg_path + '/models/grasp_al_net/rgbd_config_net_os.ckpt'

        test_config_net(test_data_path, grasp_model_path) 

