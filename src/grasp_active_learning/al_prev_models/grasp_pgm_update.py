#!/usr/bin/env python

import numpy as np
import time
import h5py
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut
from sklearn.decomposition import PCA
from sklearn import neighbors, linear_model
from sklearn.svm import SVC
from sklearn import mixture
from sklearn.cluster import KMeans
#import cross_validation
#import plot_roc_pr_curve as plot_curve
import matplotlib.pyplot as plt
#import map_classifier as mc

#import os
#os.sys.path.append('./ppca/src/pca')
#import pca
#import ppca as prob_pca
import pickle
from scipy.stats import multivariate_normal
import rospy
from prob_grasp_planner.srv import *
#import proc_grasp_data as pgd
import roslib.packages as rp
pkg_path = rp.get_pkg_dir('prob_grasp_planner')
sys.path.append(pkg_path + '/src/grasp_type_planner')
from grasp_pgm_inference import GraspPgmInfer


class GraspPgmUpdate:
    '''
    Update grasp pgms using active learning grasp queries.
    '''
    def __init__(self):
        self.query_batch_num = 3
        pkg_path = rp.get_pkg_dir('prob_grasp_planner') 
        self.load_pre_train_pca = True
        self.pre_train_pca_path = pkg_path + '/models/grasp_type_planner/' + \
                              'train_models/pca/pca.model'
        self.load_pre_train_clf = False
        # TODO: if self.load_pre_train_clf is true, the logistic regression training 
        # algorithm needs to be changed from the default liblinear to other methods,
        # such as sag or lbfgs, because liblinear doesn't support warm_start being true.
        self.pre_train_clf_path = pkg_path + '/models/grasp_type_planner' + \
                                '/train_models/classifiers/'
        self.load_pre_train_prior = False
        self.pre_train_prior_path = pkg_path + '/models/grasp_type_planner' + \
                          '/train_models/priors_all/'
        self.grasp_data_path = '/mnt/tars_data/grasp_queries/' 
        self.save_pca_model = False
        self.gmm_components_num = 2
        self.al_clf_path = pkg_path + '/models/grasp_type_planner' + \
                                '/al_models/classifiers/'
        self.al_prior_path = pkg_path + '/models/grasp_type_planner' + \
                          '/al_models/priors/'
        self.grasp_model_inf = GraspPgmInfer(pgm_grasp_type=True)


    def load_pre_train_models(self):
        if self.load_pre_train_pca:
            self.pca_model = pickle.load(open(self.pre_train_pca_path, 'rb'))
        if self.load_pre_train_clf:
            self.clf_model = pickle.load(open(self.pre_train_clf_path + 'prec_clf.model', 'rb'))
        else:
            # Solvers: 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'
            self.clf_model = linear_model.LogisticRegression(solver='lbfgs', warm_start=True)
            self.clf_model.warm_start = True
        if self.load_pre_train_prior:
            self.prior_model = pickle.load(open(self.pre_train_prior_path + 'prec_gmm.model', 'rb'))
            self.prior_model.warm_start = True
        else:
            self.prior_model = mixture.GaussianMixture(n_components=self.gmm_components_num, covariance_type='full', 
                    random_state=0, init_params='kmeans', n_init=5, warm_start=True)


    def load_prev_models(self):
        '''
        Load models trained from previous grasp queries.
        '''
        if self.load_pre_train_pca:
            self.pca_model = pickle.load(open(self.pre_train_pca_path, 'rb'))
        prev_al_clf_path = self.al_clf_path + 'al_clf_batches_' + \
                            str(self.query_batch_num - 1) + '.model'
        self.clf_model = pickle.load(open(prev_al_clf_path, 'rb'))
        prev_al_prior_path = self.al_prior_path + 'al_prior_batches_'+ \
                                str(self.query_batch_num - 1) + '.model'
        self.prior_model = pickle.load(open(prev_al_prior_path, 'rb'))


    def load_grasp_queries(self):
        grasp_voxel_grids = None
        grasp_configs = None
        grasp_labels = None
        for batch_id in xrange(self.query_batch_num):
            query_batch_path = self.grasp_data_path + 'query_batch_' + str(batch_id) + \
                                '/grasp_voxel_data.h5'
            query_batch_file = h5py.File(query_batch_path, 'r')
            batch_voxel_grids = query_batch_file['grasp_voxel_grids'][()] 
            batch_configs_obj = query_batch_file['grasp_configs_obj'][()]
            batch_labels = query_batch_file['grasp_labels'][()]
            # Convert negative labels to zero
            batch_labels[batch_labels < 0] = 0
            if batch_id == 0:
                grasp_voxel_grids = batch_voxel_grids
                grasp_configs = batch_configs_obj
                grasp_labels = batch_labels
            else:
                grasp_voxel_grids = np.concatenate((grasp_voxel_grids, batch_voxel_grids))
                grasp_configs = np.concatenate((grasp_configs, batch_configs_obj))
                grasp_labels = np.concatenate((grasp_labels, batch_labels))
            query_batch_file.close()
            
        return grasp_voxel_grids, grasp_configs, grasp_labels
 

    def train_al_clf(self, latent_dim=15, whiten=False):
        '''
        Train and save logistic regression model using all grasp queries. 

        Args:
                latent_dim: the dimension of the latent space. Don't do dimension reduction when 
                it is -1.

        '''
        # Read and preprocess the data
        grasp_voxel_grids = np.copy(self.grasp_voxel_grids)
        grasp_configs = np.copy(self.grasp_configs)
        grasp_labels = np.copy(self.grasp_labels)
        #grasp_voxel_grids = np.array(grasp_voxel_grids)
        grasps_num = grasp_voxel_grids.shape[0]
        voxel_grid_dim = grasp_voxel_grids[0].shape
        voxel_num = voxel_grid_dim[0] * voxel_grid_dim[1] * voxel_grid_dim[2]  
        #Convert voxel to 1d.
        grasp_1d_voxel_grids = np.reshape(grasp_voxel_grids, (grasps_num, voxel_num))
        #grasp_voxel_config = np.concatenate((grasp_1d_voxel_grids, grasp_configs), axis=1)
        #grasp_voxel_config = grasp_configs
        grasp_voxel_config = grasp_1d_voxel_grids
        
        # Shuffule the data for cross validation.
        print 'Total number of grasps:', grasps_num
        print 'Number of success grasps', np.sum(grasp_labels)
        
        X_train = grasp_voxel_config
        Y_train = grasp_labels

        if latent_dim != -1:
            # PCA
            #if self.save_pca_model:
            #    pca = PCA(n_components=latent_dim, whiten=whiten, svd_solver='full')
            #    pca.fit(X_train)
            #    pickle.dump(pca, open(self.pca_model_path, 'wb'))
            print X_train.shape
            X_train = self.pca_model.transform(X_train)
            print X_train.shape

        #Concatenate the latent voxel and the grasp configuration.
        X_train = np.concatenate((X_train, grasp_configs), axis=1)

        self.clf_model.fit(X_train, Y_train)
        print self.clf_model.intercept_
        print self.clf_model.coef_
        print self.clf_model.n_iter_
        #np.save(self.train_model_path + '.npy', 
        #        np.concatenate((self.clf_model.coef_[0], self.clf_model.intercept_)))
        cur_al_clf_path = self.al_clf_path + 'al_clf_batches_'+ str(self.query_batch_num) + '.model'
        pickle.dump(self.clf_model, open(cur_al_clf_path, 'wb'))
        pred_train = self.clf_model.predict(X_train)
        pred_prob_train = self.clf_model.predict_proba(X_train)
        print 'Training pred success grasps number:', np.sum(pred_train)
        train_acc = 1. - np.mean(np.abs(pred_train - Y_train))
        print('Training logisticRegression accuracy: %f' % train_acc)


    def update_prior_poses_client(self, prior_means):
        '''
        Client to update the GMM prior mean poses.
        '''
        #Add average grasp config
        prior_poses = []
        for config in prior_means:
            hand_config = self.grasp_model_inf.convert_preshape_to_full_config(config) 
            prior_poses.append(hand_config.palm_pose)

        rospy.loginfo('Waiting for service update_grasp_prior_poses.')
        rospy.wait_for_service('update_grasp_prior_poses')
        rospy.loginfo('Calling service update_grasp_prior_poses.')
        try:
            update_prior_poses_proxy = rospy.ServiceProxy('update_grasp_prior_poses', UpdatePriorPoses)
            update_prior_poses_request = UpdatePriorPosesRequest()
            update_prior_poses_request.prior_poses = prior_poses
            update_prior_poses_response = update_prior_poses_proxy(update_prior_poses_request) 
        except rospy.ServiceException, e:
            rospy.loginfo('Service update_grasp_prior_poses call failed: %s'%e)
        rospy.loginfo('Service update_grasp_prior_poses is executed.')


    def fit_al_prior(self):
        grasp_configs = np.copy(self.grasp_configs)
        #grasp_configs = np.copy(self.suc_configs)
        self.prior_model.fit(grasp_configs)
        #pred_prob = self.prior_model.predict_proba(grasp_configs)
        #print pred_prob
        #print self.prior_model.score_samples(grasp_configs)
        #np.save(self.prior_path + 'covariances.npy', self.prior_model.covariances_)
        #np.save(self.prior_path + 'weights.npy', self.prior_model.weights_)
        #np.save(self.prior_path + 'means.npy', self.prior_model.means_)
        print 'weights:', self.prior_model.weights_
        print 'means:', self.prior_model.means_
        cur_al_prior_path = self.al_prior_path + 'al_prior_batches_'+ \
                                str(self.query_batch_num) + '.model'
        pickle.dump(self.prior_model, open(cur_al_prior_path, 'wb'))

        self.update_prior_poses_client(self.prior_model.means_)


    def update_models(self):
        if self.query_batch_num <= 0:
            print 'Can not update the model without grasp queries!'
            return
        self.grasp_voxel_grids, self.grasp_configs, self.grasp_labels = \
                self.load_grasp_queries()
        print self.grasp_voxel_grids
        print self.grasp_configs
        print self.grasp_labels

        if self.query_batch_num == 1:
            self.load_pre_train_models()
        else:
            self.load_prev_models()
        self.train_al_clf()
        self.fit_al_prior()


if __name__ == '__main__':
    grasp_pgm_update = GraspPgmUpdate()    
    grasp_pgm_update.update_models()

