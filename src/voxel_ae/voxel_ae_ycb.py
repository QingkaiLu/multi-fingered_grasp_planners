import roslib.packages as rp
import time
from voxel_ae import VoxelVAE 
#from voxel_data_loader import VoxelLoader
import tensorflow as tf
import numpy as np
import copy
import show_voxel
import ycb_utils
from ycb_objects import _OBJECTS, _OBJECTS_TEST
import train_voxel_ae


_CLOUD_BERKELEY_SFX = '/clouds/merged_cloud.ply'
_CLOUD_GOOGLE_SFX = '/google_64k/nontextured.ply'


def test_ae_ycb(base_obj_path, cnn_model_path):
    voxel_ae = VoxelVAE()
    voxel_ae.batch_size = 1
    voxel_ae.test_voxel_ae_model()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    tf.get_default_graph().finalize()   
     
    testing_epochs = 1
    start_time = time.time()
    iter_num = 0
    loss_sum = 0.
    total_TP = 0
    total_FN = 0
    total_FP = 0
    total_TN = 0
    non_empty_voxel_num = 0
    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, cnn_model_path)
        # Training cycle
        for object_name in _OBJECTS:
            # Read in data from ply files in ycb data set
            #object_path = base_obj_path + object_name + _CLOUD_GOOGLE_SFX
            object_path = base_obj_path + object_name + _CLOUD_BERKELEY_SFX
            object_voxel = ycb_utils.ply_to_voxel(object_path)
            #sparse_voxel = show_voxel.convert_to_sparse_voxel_grid(voxel)
            #print 'Gt voxel grid non-empty voxels:', sparse_voxel.shape[0]
            #show_voxel.plot_voxel(sparse_voxel) 

            #voxel_batch, voxel_names = voxel_loader.next_batch(voxel_ae.batch_size,
            #                                                   preprocess=False)
            voxel_batch = np.expand_dims(np.expand_dims(object_voxel, -1), 0)
            feed_dict = {voxel_ae.is_train: False,
                         voxel_ae.holder_voxel_grids: voxel_batch,
                         }
            [voxel_recons_output, loss_output] = sess.run([voxel_ae.voxel_reconstructed, 
                                                           voxel_ae.loss],
                                                          feed_dict=feed_dict)
            loss_sum += loss_output
            batch_TP, batch_FN, batch_FP, batch_TN = train_voxel_ae.eval_voxel_batch(voxel_batch, 
                                                                      voxel_recons_output)
            total_TP += batch_TP
            total_FN += batch_FN
            total_FP += batch_FP
            total_TN += batch_TN
            print 'iter_num:', iter_num
            print 'loss:', loss_output
            print 'batch_TP:', batch_TP
            print 'batch_FN:', batch_FN
            print 'batch_FP:', batch_FP
            print 'batch_TN:', batch_TN
            batch_voxel_num = np.prod(voxel_batch.shape)
            if batch_voxel_num != batch_TP + batch_FN + batch_FP + batch_TN: 
                print 'Eval voxel number is diffrent from true voxel number!'
            if batch_TP + batch_FN != 0:
                print 'batch_FN / (batch_TP + batch_FN):', float(batch_FN) / (batch_TP + batch_FN)
            if batch_FP + batch_TN != 0:
                print 'batch_FP / (batch_FP + batch_TN):', float(batch_FP) / (batch_FP + batch_TN)
            if batch_FP + batch_TP != 0:
                print 'batch_TP / (batch_FP + batch_TP):', float(batch_TP) / (batch_FP + batch_TP)
            print '(batch_FN + batch_FP) / batch_voxel_num :',\
                    float(batch_FN + batch_FP) / batch_voxel_num

            train_voxel_ae.vis_batch_voxel(voxel_batch, voxel_recons_output)
            batch_non_empty_voxel_num = np.sum(voxel_batch == 1)
            print 'batch_non_empty_voxel_num:', batch_non_empty_voxel_num
            non_empty_voxel_num += batch_non_empty_voxel_num 
            iter_num += 1
 
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
    print 'Average non empty voxel num:', non_empty_voxel_num / (iter_num * voxel_ae.batch_size)


if __name__ == '__main__':
    #is_train = False #True
    pkg_path = rp.get_pkg_dir('prob_grasp_planner') 
    cnn_model_path = pkg_path + '/models/voxel_ae/voxel_vae_ae_aug.ckpt'
    base_obj_path = '/mnt/tars_data/ycb/'
    test_ae_ycb(base_obj_path, cnn_model_path)

