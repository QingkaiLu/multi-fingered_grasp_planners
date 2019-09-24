import time
from voxel_ae import VoxelAE 
from voxel_data_loader import VoxelLoader
import tensorflow as tf
import numpy as np
import copy
import roslib.packages as rp
#pkg_path = rp.get_pkg_dir('prob_grasp_planner')
import sys
import os
#sys.path.append(pkg_path + '/src/grasp_type_planner')

from show_voxel import plot_voxel, convert_to_sparse_voxel_grid

from voxel_dataset import get_voxel_dataset

def train_ae(train_data_path, logs_path, cnn_model_path):
    batch_size = 64

    # Get voxel dataset (tf Dataset).
    train_files = [os.path.join(train_data_path, filename) for filename in os.listdir(train_data_path) if ".tfrecord" in filename] 
    train_dataset = get_voxel_dataset(train_files, batch_size=batch_size)
    train_iterator = train_dataset.make_initializable_iterator()
    train_next_partial, train_next_full = train_iterator.get_next()

    # Build model
    voxel_ae = VoxelAE()
    voxel_ae.train_voxel_ae_model()
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()
    
    tf.get_default_graph().finalize()   
    logs_path_train = logs_path + '/voxel_train_ae_aug'
    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path_train, graph=tf.get_default_graph())
     
    #learning_rates = {0:0.0001, 1:0.001, 50:0.00001}
    #learning_rates = {0:0.0001, 10:0.001, 100:0.0001, 150:0.00001}
    learning_rates = {0:0.001, 20:0.0001, 80:0.00001}
    learning_rate = None
    training_epochs = 100

    #voxel_loader = VoxelLoader(train_data_path)

    start_time = time.time()
    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        # Training cycle
        #while voxel_loader.epochs_completed < training_epochs:

        iter_num = 0
        
        for epoch in range(training_epochs):
            print "Epoch: ", str(epoch)

            # Initialize the dataset for this round.
            sess.run(train_iterator.initializer)
            
            if epoch in learning_rates:
                learning_rate = learning_rates[epoch]

            # voxel_batch, voxel_names = voxel_loader.next_batch(batch_size,
            #                                                    preprocess=False)

            while True:
                try:
                    partial_voxel_, true_voxel_ = sess.run([train_next_partial, train_next_full])
            
                    feed_dict = {voxel_ae.is_train: True,
                                 voxel_ae.learning_rate: learning_rate,
                                 voxel_ae.partial_voxel_grids: partial_voxel_,
                                 voxel_ae.true_voxel_grids: true_voxel_,
                    }
                    [_, train_summary_output, loss_output, 
                     voxel_recons_output] = sess.run([voxel_ae.optimizer, 
                                                      voxel_ae.train_summary,
                                                      voxel_ae.loss,
                                                      voxel_ae.voxel_reconstructed],
                                                     feed_dict=feed_dict)
                    batch_TP, batch_FN, batch_FP, batch_TN = eval_voxel_batch(true_voxel_, 
                                                                      voxel_recons_output)

                    # Write logs at every iteration
                    summary_writer.add_summary(train_summary_output, iter_num)

                    batch_voxel_num = np.prod(partial_voxel_.shape)
                    if batch_voxel_num != batch_TP + batch_FN + batch_FP + batch_TN: 
                        print 'Eval voxel number is diffrent from true voxel number!'
                    iter_num += 1

                except tf.errors.OutOfRangeError:
                    break

            print 'epochs_completed:', epoch
            print 'iter_num:', iter_num
            print 'learning_rate:', learning_rate
            print 'loss:', loss_output
            print 'batch_TP:', batch_TP
            print 'batch_FN:', batch_FN
            print 'batch_FP:', batch_FP
            print 'batch_TN:', batch_TN
            if batch_TP + batch_FN != 0:
                print 'batch_FN / (batch_TP + batch_FN):', float(batch_FN) / (batch_TP + batch_FN)
            if batch_FP + batch_TN != 0:
                print 'batch_FP / (batch_FP + batch_TN):', float(batch_FP) / (batch_FP + batch_TN)
            if batch_FP + batch_TP != 0:
                print 'batch_TP / (batch_FP + batch_TP):', float(batch_TP) / (batch_FP + batch_TP)
            print '(batch_FN + batch_FP) / batch_voxel_num :',\
                    float(batch_FN + batch_FP) / batch_voxel_num


            # Save every epoch.
            saver.save(sess, cnn_model_path)
         
    elapsed_time = time.time() - start_time
    print 'Total training elapsed_time: ', elapsed_time


def eval_voxel_batch(voxel_batch, recons_voxel_batch):
    # Cancel the effect of the voxel preprocessing
    # bi_gt_voxel_batch = copy.deepcopy(gt_voxel_batch)
    # bi_gt_voxel_batch[gt_voxel_batch > 0] = 1
    # bi_gt_voxel_batch[gt_voxel_batch < 0] = 0

    binary_voxel_batch = copy.deepcopy(recons_voxel_batch)
    binary_thresh = 0.5
    binary_voxel_batch[recons_voxel_batch >= binary_thresh] = 1
    binary_voxel_batch[recons_voxel_batch < binary_thresh] = 0
    TP = np.sum(binary_voxel_batch[voxel_batch == 1] == 1)
    FN = np.sum(binary_voxel_batch[voxel_batch == 1] == 0)
    FP = np.sum(binary_voxel_batch[voxel_batch == 0] == 1)
    TN = np.sum(binary_voxel_batch[voxel_batch == 0] == 0)
    return TP, FN, FP, TN


def vis_batch_voxel(batch_voxel, batch_recons_voxel):
    for i, voxel in enumerate(batch_voxel):
        recons_voxel = batch_recons_voxel[i]
        binary_recons_voxel = copy.deepcopy(recons_voxel)
        binary_recons_voxel[recons_voxel >= 0.5] = 1.
        binary_recons_voxel[recons_voxel < 0.5] = 0.
        sparse_voxel = show_voxel.convert_to_sparse_voxel_grid(voxel)
        print 'Gt voxel grid non-empty voxels:', sparse_voxel.shape[0]
        show_voxel.plot_voxel(sparse_voxel) 
        sparse_recons_voxel = show_voxel.convert_to_sparse_voxel_grid(binary_recons_voxel)
        show_voxel.plot_voxel(sparse_recons_voxel) 


def test_ae(test_data_path, cnn_model_path):
    batch_size = 1

    # Get voxel dataset (tf Dataset).
    test_files = [os.path.join(test_data_path, filename) for filename in os.listdir(test_data_path) if ".tfrecord" in filename] 
    test_dataset = get_voxel_dataset(test_files, batch_size=batch_size)
    test_iterator = test_dataset.make_initializable_iterator()
    test_next_partial, test_next_full = test_iterator.get_next()

    # Build model
    voxel_ae = VoxelAE()
    voxel_ae.test_voxel_ae_model()
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()
    
    tf.get_default_graph().finalize() 
     
    start_time = time.time()
    # Launch the graph
    with tf.Session() as sess:

        # Load the model.
        saver.restore(sess, cnn_model_path)

        # Get results.
        iter_num = 0
        loss_sum = 0.
        total_TP = 0
        total_FN = 0
        total_FP = 0
        total_TN = 0
        non_empty_voxel_num = 0

        # Initialize the dataset for this round.
        sess.run(test_iterator.initializer)

        while True:
            try:
                partial_voxel_, true_voxel_ = sess.run([test_next_partial, test_next_full])

                feed_dict = {voxel_ae.is_train: False,
                             voxel_ae.partial_voxel_grids: partial_voxel_,
                             voxel_ae.true_voxel_grids: true_voxel_,
                }

                [voxel_recons_output, embed_output, 
                    loss_output] = sess.run([voxel_ae.voxel_reconstructed, 
                                            voxel_ae.embedding,
                                            voxel_ae.loss],
                                            feed_dict=feed_dict)

                batch_TP, batch_FN, batch_FP, batch_TN = eval_voxel_batch(true_voxel_, 
                                                                  voxel_recons_output)

                plot_voxel(convert_to_sparse_voxel_grid(np.reshape(partial_voxel_, (32,32,32))), voxel_res=(32,32,32))
                plot_voxel(convert_to_sparse_voxel_grid(np.reshape(true_voxel_, (32,32,32))), voxel_res=(32,32,32))                
                plot_voxel(convert_to_sparse_voxel_grid(np.reshape(voxel_recons_output, (32,32,32))), voxel_res=(32,32,32))

                # Write logs at every iteration
                #summary_writer.add_summary(train_summary_output, iter_num)

                batch_voxel_num = np.prod(partial_voxel_.shape)
                if batch_voxel_num != batch_TP + batch_FN + batch_FP + batch_TN: 
                    print 'Eval voxel number is diffrent from true voxel number!'
                iter_num += 1

                loss_sum += loss_output
                
                total_TP += batch_TP
                total_FN += batch_FN
                total_FP += batch_FP
                total_TN += batch_TN
                # print 'iter_num:', iter_num
                # print 'loss:', loss_output
                # print 'batch_TP:', batch_TP
                # print 'batch_FN:', batch_FN
                # print 'batch_FP:', batch_FP
                # print 'batch_TN:', batch_TN

                # batch_voxel_num = np.prod(partial_voxel_.shape)
                # if batch_voxel_num != batch_TP + batch_FN + batch_FP + batch_TN: 
                #     print 'Eval voxel number is diffrent from true voxel number!'
                # if batch_TP + batch_FN != 0:
                #     print 'batch_FN / (batch_TP + batch_FN):', float(batch_FN) / (batch_TP + batch_FN)
                # if batch_FP + batch_TN != 0:
                #     print 'batch_FP / (batch_FP + batch_TN):', float(batch_FP) / (batch_FP + batch_TN)
                # if batch_FP + batch_TP != 0:
                #     print 'batch_TP / (batch_FP + batch_TP):', float(batch_TP) / (batch_FP + batch_TP)
                # print '(batch_FN + batch_FP) / batch_voxel_num :',\
                #         float(batch_FN + batch_FP) / batch_voxel_num

                #vis_batch_voxel(voxel_batch, voxel_recons_output)
                batch_non_empty_voxel_num = np.sum(partial_voxel_ == 1)
                #print 'batch_non_empty_voxel_num:', batch_non_empty_voxel_num
                non_empty_voxel_num += batch_non_empty_voxel_num 
                iter_num += 1

            except tf.errors.OutOfRangeError:
                break

    elapsed_time = time.time() - start_time
    print 'Total testing elapsed_time: ', elapsed_time
    avg_loss = loss_sum / iter_num
    avg_TP = total_TP / iter_num
    avg_FN = total_FN / iter_num
    avg_FP = total_FP / iter_num
    avg_TN = total_TN / iter_num
    print 'avg_loss:', avg_loss
    print 'avg_TP:', avg_TP
    print 'avg_FN:', avg_FN
    print 'avg_FP:', avg_FP
    print 'avg_TN:', avg_TN
    print 'avg_FN / (avg_TP + avg_FN):', float(avg_FN) / (avg_TP + avg_FN)
    print 'avg_FP / (avg_FP + avg_TN):', float(avg_FP) / (avg_FP + avg_TN)
    print 'avg_TP / (avg_FP + avg_TP):', float(avg_TP) / (avg_FP + avg_TP)
    print '(batch_FN + batch_FP) / batch_voxel_num :', \
            float(batch_FN + batch_FP) / (batch_voxel_num)
    print 'Average non empty voxel num:', non_empty_voxel_num / (iter_num * batch_size)

if __name__ == '__main__':
    is_train = False
    #cnn_model_path = pkg_path + '/models/voxel_ae/voxel_ae_aug.ckpt'
    #cnn_model_path = pkg_path + '/models/voxel_ae/voxel_vae_ae_aug.ckpt'
    cnn_model_path = 'models/voxel_ae/'
    if is_train:
        #train_data_path = '/home/qingkai/Workspace/Generative-and-Discriminative-' + \
        #                    'Voxel-Modeling/datasets/shapenet10_train_nr.tar' 
        #train_data_path = '/home/qingkai/Workspace/Generative-and-Discriminative-' + \
        #                    'Voxel-Modeling/datasets/shapenet10_train.tar'
        train_data_path = '/dataspace/ReconstructionData/VoxelObjectFrameCenter/Train'
        logs_path = 'logs/voxel_ae/'
        train_ae(train_data_path, logs_path, cnn_model_path)
    else:
        #test_data_path = '/home/qingkai/Workspace/Generative-and-Discriminative-' + \
        #                    'Voxel-Modeling/datasets/shapenet10_test_nr.tar' 
        #test_data_path = '/home/qingkai/Workspace/Generative-and-Discriminative-' + \
        #                    'Voxel-Modeling/datasets/shapenet10_test.tar'
        test_data_path = '/dataspace/ReconstructionData/VoxelObjectFrameCenter/Validation'
        test_ae(test_data_path, cnn_model_path)

