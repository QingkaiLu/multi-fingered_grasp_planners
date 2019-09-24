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
import plot_roc_pr_curve as plot_curve
import matplotlib.pyplot as plt
import map_classifier as mc

import os
os.sys.path.append('./ppca/src/pca')
#import pca
import ppca as prob_pca
import pickle
from scipy.stats import multivariate_normal
import rospy
from prob_grasp_planner.srv import *
import proc_grasp_data as pgd
import roslib.packages as rp


class GraspPgmLearner:
    '''
    Learning of grasp pgms with and without grasp types.
    '''
    def __init__(self):
        self.power_data_path = '/mnt/tars_data/multi_finger_sim_data_complete_v4/'
        #Successful power and precision grasp data.
        self.align_object_tf = True
        if self.align_object_tf:
            self.power_grasp_voxel_paths = [self.power_data_path + 'power_grasps/power_align_suc_grasps.h5']
            self.prec_data_path = '/mnt/tars_data/multi_finger_sim_data_precision'
            self.prec_grasp_voxel_paths = [self.prec_data_path + '_1/' + 'prec_grasps/prec_align_suc_grasps.h5',
                                            self.prec_data_path + '_2/' + 'prec_grasps/prec_align_suc_grasps.h5',
                                            self.prec_data_path + '_3/' + 'prec_grasps/prec_align_suc_grasps.h5',
                                            self.prec_data_path + '_4/' + 'prec_grasps/prec_align_suc_grasps.h5',
                                            self.prec_data_path + '_5/' + 'prec_grasps/prec_align_suc_grasps.h5',]
        else:
            self.power_grasp_voxel_paths = [self.power_data_path + 'power_grasps/grasp_voxel_obj_tf_data.h5']
            self.prec_data_path = '/mnt/tars_data/multi_finger_sim_data_precision'
            self.prec_grasp_voxel_paths = [self.prec_data_path + '_1/' + 'prec_grasps/grasp_voxel_obj_tf_data.h5',
                                            self.prec_data_path + '_2/' + 'prec_grasps/grasp_voxel_obj_tf_data.h5',
                                            self.prec_data_path + '_3/' + 'prec_grasps/grasp_voxel_obj_tf_data.h5',
                                            self.prec_data_path + '_4/' + 'prec_grasps/grasp_voxel_obj_tf_data.h5',
                                            self.prec_data_path + '_5/' + 'prec_grasps/grasp_voxel_obj_tf_data.h5',]
        #Failure power and precision grasp data.
        if self.align_object_tf:
            self.fail_power_grasp_voxel_paths = [self.power_data_path + 'power_grasps/power_align_failure_grasps.h5']
            self.fail_prec_grasp_voxel_paths = [self.prec_data_path + '_1/' + 'prec_grasps/prec_align_failure_grasps.h5',
                                                self.prec_data_path + '_2/' + 'prec_grasps/prec_align_failure_grasps.h5',
                                                self.prec_data_path + '_3/' + 'prec_grasps/prec_align_failure_grasps.h5',
                                                self.prec_data_path + '_4/' + 'prec_grasps/prec_align_failure_grasps.h5',
                                                self.prec_data_path + '_5/' + 'prec_grasps/prec_align_failure_grasps.h5']
        else:
            self.fail_power_grasp_voxel_paths = [self.power_data_path + 'power_grasps/power_failure_grasp_voxel_data.h5']
            self.fail_prec_grasp_voxel_paths = [self.prec_data_path + '_1/' + 'prec_grasps/prec_failure_grasp_voxel_data.h5',
                                                self.prec_data_path + '_2/' + 'prec_grasps/prec_failure_grasp_voxel_data.h5',
                                                self.prec_data_path + '_3/' + 'prec_grasps/prec_failure_grasp_voxel_data.h5',
                                                self.prec_data_path + '_4/' + 'prec_grasps/prec_failure_grasp_voxel_data.h5',
                                                self.prec_data_path + '_5/' + 'prec_grasps/prec_failure_grasp_voxel_data.h5']
        #Grasp types classification or grasp success classification 
        self.grasp_type_clf = False
        self.all_type_grasp_clf = True 
        self.power_grasp_clf = True
        self.gen_failure_grasp = False #True

        pkg_path = rp.get_pkg_dir('prob_grasp_planner') 
        self.save_pca_model = True
        self.pca_model_path = pkg_path + '/models/grasp_type_planner/train_models/pca/pca.model'
        #Grasp success classification for power or precision grasps.
        self.fail_indices_path = pkg_path + '/models/grasp_type_planner/train_models/rand_grasp_indices/'

        self.proc_grasp = pgd.ProcGraspData('')
        self.load_grasp_data()

        self.train_model_path = pkg_path + '/models/grasp_type_planner/train_models/classifiers/'
        self.prior_path = pkg_path + '/models/grasp_type_planner/train_models/priors/'
        #self.prior_path = '../train_models/gmm_priors/'
        self.roc_fig_path = '../cross_val/plots/'
        self.cross_val_path = '../cross_val/'
        self.leave_one_out_path = '../leave_one_out/'
        if self.all_type_grasp_clf:
            self.train_model_path += 'all_type_clf' 
            self.prior_path += 'all_type_'
            self.roc_fig_path += 'roc_all.png'
            self.cross_val_path += 'all/'
            self.leave_one_out_path += 'all/'
        elif self.power_grasp_clf:
            self.train_model_path += 'power_clf' 
            self.prior_path += 'power_'
            self.roc_fig_path += 'roc_power.png'
            self.cross_val_path += 'power/'
            self.leave_one_out_path += 'power/'
        else:
            self.train_model_path += 'prec_clf' 
            self.prior_path += 'prec_'
            self.roc_fig_path += 'roc_prec.png'
            self.cross_val_path += 'prec/'
            self.leave_one_out_path += 'prec/'

    def load_suc_or_failure_grasp(self, load_power, load_success):
        '''
        Load successful or failure grasp of one type (precision or power).
        '''
        grasp_voxel_paths = None
        if load_success:
            #Load successful grasps.
            if load_power:
                grasp_voxel_paths = self.power_grasp_voxel_paths
            else:
                grasp_voxel_paths = self.prec_grasp_voxel_paths
        else:
            #Load failure grasps.
            if load_power:
                grasp_voxel_paths = self.fail_power_grasp_voxel_paths
            else:
                grasp_voxel_paths = self.fail_prec_grasp_voxel_paths

        return self.read_grasp_voxel_data(grasp_voxel_paths)

    def load_one_type_grasp(self, load_power):
        '''
        Load successful and failure grasp of one type (precision or power).
        '''
        suc_grasp_voxel_grids, suc_grasp_configs = \
                            self.load_suc_or_failure_grasp(load_power, load_success=True)
        fail_grasp_voxel_grids, fail_grasp_configs = \
                            self.load_suc_or_failure_grasp(load_power, load_success=False)
        suc_grasps_num = len(suc_grasp_configs)
        suc_grasp_labels = np.ones(suc_grasps_num)
        fail_grasps_num = 2 * suc_grasps_num
        grasp_type = 'power'
        if not load_power:
            grasp_type = 'prec'
        rand_indices_path = self.fail_indices_path + grasp_type + '_rand_indices.npy'
        #Generate random failure grasps or not. If not, use the previous saved failure grasps.
        if self.gen_failure_grasp:
            #Randomly draw and save failure grasp indices for all types.
            random_indices = np.random.choice(len(fail_grasp_configs), fail_grasps_num, replace=False)
            np.save(rand_indices_path, random_indices)
        else:
            #Read failure grasp indices generated for all types.
            random_indices = np.load(rand_indices_path)
        print 'random_indices:', random_indices
        fail_grasp_labels = np.zeros(fail_grasps_num)
        grasp_voxel_grids = np.concatenate((suc_grasp_voxel_grids, fail_grasp_voxel_grids[random_indices]))
        grasp_configs = np.concatenate((suc_grasp_configs, fail_grasp_configs[random_indices]))
        grasp_labels = np.concatenate((suc_grasp_labels, fail_grasp_labels))

        #self.update_palm_poses_client(suc_grasp_configs, tf_prefix='s_')
        #self.update_palm_poses_client(fail_grasp_configs[random_indices], tf_prefix='f_')
        #self.update_palm_poses_client([np.mean(grasp_configs, axis=0)], tf_prefix='a_')

        return grasp_voxel_grids, grasp_configs, grasp_labels, suc_grasp_configs

    def load_grasp_data(self):
        if self.grasp_type_clf:
            #Only load success grasps
            power_grasp_voxel_grids, power_grasp_configs = \
                                    self.load_suc_or_failure_grasp(load_power=True, load_success=True)
            prec_grasp_voxel_grids, prec_grasp_configs = \
                                    self.load_suc_or_failure_grasp(load_power=False, load_success=True)
            self.grasp_voxel_grids = np.concatenate((power_grasp_voxel_grids, prec_grasp_voxel_grids))
            self.grasp_configs = np.concatenate((power_grasp_configs, prec_grasp_configs))
            self.power_grasp_labels = np.zeros(len(power_grasp_configs))
            self.prec_grasp_labels = np.ones(len(prec_grasp_configs))
            self.grasp_labels = np.concatenate((power_grasp_labels, prec_grasp_labels))
        elif self.all_type_grasp_clf:
            power_grasp_voxel_grids, power_grasp_configs, power_grasp_labels, \
                                        power_suc_configs = self.load_one_type_grasp(load_power=True)
            prec_grasp_voxel_grids, prec_grasp_configs, prec_grasp_labels, \
                                        prec_suc_configs = self.load_one_type_grasp(load_power=False)
            self.grasp_voxel_grids = np.concatenate((power_grasp_voxel_grids, prec_grasp_voxel_grids))
            self.grasp_configs = np.concatenate((power_grasp_configs, prec_grasp_configs))
            self.grasp_labels = np.concatenate((power_grasp_labels, prec_grasp_labels))
            self.suc_configs = np.concatenate((power_suc_configs, prec_suc_configs))
        else:
            self.grasp_voxel_grids, self.grasp_configs, self.grasp_labels, \
                                self.suc_configs = self.load_one_type_grasp(self.power_grasp_clf)

    def read_grasp_voxel_data(self, voxel_paths):
        grasp_voxel_grids = None
        grasp_configs = None
        for i, voxel_path in enumerate(voxel_paths): 
            grasp_voxel_file = h5py.File(voxel_path, 'r')
            if i==0:
                grasp_voxel_grids = grasp_voxel_file['grasp_voxel_grids'][()] 
                grasp_configs = grasp_voxel_file['grasp_configs_obj'][()]
            else:
                grasp_voxel_grids = np.concatenate((grasp_voxel_grids, grasp_voxel_file['grasp_voxel_grids'][()])) 
                grasp_configs = np.concatenate((grasp_configs, grasp_voxel_file['grasp_configs_obj'][()]))
            grasp_voxel_file.close()
        return grasp_voxel_grids, grasp_configs

    def update_palm_poses_client(self, grasp_configs, tf_prefix):
        '''
        Client to update the palm pose tf.
        '''
        #Add average grasp config
        avg_config = np.mean(grasp_configs, axis=0)
        grasp_configs = np.append(grasp_configs, [avg_config], axis=0)

        palm_poses_in_obj = []
        for config in grasp_configs:
            hand_config = self.proc_grasp.convert_preshape_to_full_config(config) 
            palm_poses_in_obj.append(hand_config.palm_pose)

        rospy.loginfo('Waiting for service update_grasp_palm_poses.')
        rospy.wait_for_service('update_grasp_palm_poses')
        rospy.loginfo('Calling service update_grasp_palm_poses.')
        try:
            update_palm_poses_proxy = rospy.ServiceProxy('update_grasp_palm_poses', UpdatePalmPosesInObj)
            update_palm_poses_request = UpdatePalmPosesInObjRequest()
            update_palm_poses_request.palm_poses_in_obj = palm_poses_in_obj
            update_palm_poses_request.tf_prefix = tf_prefix
            update_palm_poses_response = update_palm_poses_proxy(update_palm_poses_request) 
        except rospy.ServiceException, e:
            rospy.loginfo('Service update_grasp_palm_poses call failed: %s'%e)
        rospy.loginfo('Service update_grasp_palm_poses is executed.')

    def gmm_cluster_grasp_types(self, latent_dim, seen_or_unseen='seen', whiten=False):
        '''
        Use GMM to cluster precision and power grasps.
        
        Args:
                latent_dim: the dimension of the latent space. Don't do dimension reduction when 
                it is -1.

        '''
        # Read and preprocess the data
        grasp_voxel_grids = np.copy(self.grasp_voxel_grids)
        grasp_configs = np.copy(self.grasp_configs)
        grasp_labels = np.copy(self.grasp_labels)
        grasp_voxel_grids = np.array(grasp_voxel_grids)
        grasps_num = grasp_voxel_grids.shape[0]
        voxel_grid_dim = grasp_voxel_grids[0].shape
        voxel_num = voxel_grid_dim[0] * voxel_grid_dim[1] * voxel_grid_dim[2]  
        #Convert voxel to 1d.
        grasp_1d_voxel_grids = np.reshape(grasp_voxel_grids, (grasps_num, voxel_num))
        grasp_voxel_config = np.concatenate((grasp_1d_voxel_grids, grasp_configs), axis=1)
        #grasp_voxel_config = grasp_configs
        #grasp_voxel_config = grasp_1d_voxel_grids

        if latent_dim != -1:
            run_pca = True
            if run_pca:
                # PCA
                pca = PCA(n_components=latent_dim, whiten=whiten, svd_solver='full')
                pca.fit(grasp_voxel_config)
                print grasp_voxel_config.shape
                grasp_voxel_config = pca.transform(grasp_voxel_config)
                print grasp_voxel_config.shape
            else:
                # PPCA
                ppca = prob_pca.PPCA(q=latent_dim)
                grasp_voxel_config_trans = np.transpose(grasp_voxel_config)
                ppca.fit(grasp_voxel_config_trans)
                grasp_voxel_config_trans = ppca.transform(grasp_voxel_config_trans)
                print grasp_voxel_config_trans.shape
                grasp_voxel_config = np.transpose(grasp_voxel_config_trans)

        deprecated_gmm = False #True
        num_components = 2
        if not deprecated_gmm:
            g = mixture.GaussianMixture(n_components=num_components, covariance_type='full', 
                    random_state=0, init_params='random', n_init=5) #'kmeans')
        else:
            import warnings
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            g = mixture.GMM(n_components=num_components, covariance_type='full', 
                    random_state=100000, n_init=5)

        g.fit(grasp_voxel_config)
        pred_prob = g.predict_proba(grasp_voxel_config)
        print pred_prob
        print np.sum(pred_prob[:, 0]), np.sum(pred_prob[:, 1])

    def kmeans_cluster_grasp_types(self, latent_dim, seen_or_unseen='seen', whiten=False):
        '''
        Use kmeans to cluster precision and power grasps.
        
        Args:
                latent_dim: the dimension of the latent space. Don't do dimension reduction when 
                it is -1.

        '''
        # Read and preprocess the data
        grasp_voxel_grids = np.copy(self.grasp_voxel_grids)
        grasp_configs = np.copy(self.grasp_configs)
        grasp_labels = np.copy(self.grasp_labels)
        grasp_voxel_grids = np.array(grasp_voxel_grids)
        grasps_num = grasp_voxel_grids.shape[0]
        voxel_grid_dim = grasp_voxel_grids[0].shape
        voxel_num = voxel_grid_dim[0] * voxel_grid_dim[1] * voxel_grid_dim[2]  
        #Convert voxel to 1d.
        grasp_1d_voxel_grids = np.reshape(grasp_voxel_grids, (grasps_num, voxel_num))
        grasp_voxel_config = np.concatenate((grasp_1d_voxel_grids, grasp_configs), axis=1)
        #grasp_voxel_config = grasp_configs
        #grasp_voxel_config = grasp_1d_voxel_grids

        if latent_dim != -1:
            run_pca = True
            if run_pca:
                # PCA
                pca = PCA(n_components=latent_dim, whiten=whiten, svd_solver='full')
                pca.fit(grasp_voxel_config)
                print grasp_voxel_config.shape
                grasp_voxel_config = pca.transform(grasp_voxel_config)
                print grasp_voxel_config.shape
            else:
                # PPCA
                ppca = prob_pca.PPCA(q=latent_dim)
                grasp_voxel_config_trans = np.transpose(grasp_voxel_config)
                ppca.fit(grasp_voxel_config_trans)
                grasp_voxel_config_trans = ppca.transform(grasp_voxel_config_trans)
                print grasp_voxel_config_trans.shape
                grasp_voxel_config = np.transpose(grasp_voxel_config_trans)

        init_method = 'random'
        #init_method = 'k-means++'
        kmeans = KMeans(n_clusters=3, random_state=0, init='random').fit(grasp_voxel_config)
        print kmeans.labels_

    def train_all(self, latent_dim, model='logistic', whiten=False):
        '''
        Train and save logistic regression model using all training grasps. 

        Args:
                latent_dim: the dimension of the latent space. Don't do dimension reduction when 
                it is -1.

        '''
        # Read and preprocess the data
        grasp_voxel_grids = np.copy(self.grasp_voxel_grids)
        grasp_configs = np.copy(self.grasp_configs)
        grasp_labels = np.copy(self.grasp_labels)
        grasp_voxel_grids = np.array(grasp_voxel_grids)
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
        
        #TO DO: test training data.
        X_train = grasp_voxel_config
        Y_train = grasp_labels

        if latent_dim != -1:
            run_pca = True
            if run_pca:
                # PCA
                #Important issue TO fix: pgm with and without grasp types
                #should use the same failing grasps!!!
                if self.all_type_grasp_clf and self.save_pca_model:
                    pca = PCA(n_components=latent_dim, whiten=whiten, svd_solver='full')
                    pca.fit(X_train)
                    pickle.dump(pca, open(self.pca_model_path, 'wb'))
                else:
                    pca = pickle.load(open(self.pca_model_path, 'rb'))
                print X_train.shape
                X_train = pca.transform(X_train)
                print X_train.shape
            else:
                # PPCA
                ppca = prob_pca.PPCA(q=latent_dim)
                X_train_trans = np.transpose(X_train)
                ppca.fit(X_train_trans)
                X_train_trans = ppca.transform(X_train_trans)
                X_train = np.transpose(X_train_trans)

        #Concatenate the latent voxel and the grasp configuration.
        X_train = np.concatenate((X_train, grasp_configs), axis=1)
        #X_train = grasp_configs

        # logistic regression
        if model == 'logistic':
            #TO DO: tune regularization to get better performance of logistic regression.
            logistic = linear_model.LogisticRegression() #C=0.5)
            logistic.fit(X_train, Y_train)
            print logistic.intercept_
            print logistic.coef_
            print logistic.n_iter_
            np.save(self.train_model_path + '.npy', 
                    np.concatenate((logistic.coef_[0], logistic.intercept_)))
            pickle.dump(logistic, open(self.train_model_path + '.model', 'wb'))
            pred_train = logistic.predict(X_train)
            pred_prob_train = logistic.predict_proba(X_train)
            print 'Training pred success grasps number:', np.sum(pred_train)
            train_acc = 1. - np.mean(np.abs(pred_train - Y_train))
            print('Training logisticRegression accuracy: %f' % train_acc)

        # MAP with Gaussian or GMM
        if model in ['map_gaus', 'map_gmm']:
            map_gaus = mc.MapClassifier(prob=model.split('_')[1])
            map_gaus.fit(X_train, Y_train)
            pred_prob = map_gaus.predict_prob(X_train)
            print pred_prob
            pred = (pred_prob[:, 1] > 0.5).astype(int)
            print 'pred success grasps number:', np.sum(pred)
            score = 1. - np.mean(np.abs(pred - Y_train))
            #score = svm.score(X_train, Y_train)
            print('MAP Gaussian score: %f' % score)

    def fit_prior(self):
        grasp_configs = np.copy(self.grasp_configs)
        #grasp_configs = np.copy(self.suc_configs)
        deprecated_gmm = False #True
        num_components = 4
        if not deprecated_gmm:
            #g = mixture.GaussianMixture(n_components=num_components, covariance_type='full', 
            #        random_state=0, init_params='random', n_init=5)
            g = mixture.GaussianMixture(n_components=num_components, covariance_type='full', 
                    random_state=0, init_params='kmeans', n_init=5)
        else:
            import warnings
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            g = mixture.GMM(n_components=num_components, covariance_type='full', 
                    random_state=100000, n_init=5)

        g.fit(grasp_configs)
        #pred_prob = g.predict_proba(grasp_configs)
        #print pred_prob
        #print g.score_samples(grasp_configs)
        if not deprecated_gmm:
            np.save(self.prior_path + 'covariances.npy', g.covariances_)
        else:
            np.save(self.prior_path + 'covariances.npy', g.covars_)
        np.save(self.prior_path + 'weights.npy', g.weights_)
        np.save(self.prior_path + 'means.npy', g.means_)
        print 'weights:', g.weights_
        print 'means:', g.means_
        pickle.dump(g, open(self.prior_path + 'gmm.model', 'wb'))
        self.update_prior_poses_client(g.means_)

    def update_prior_poses_client(self, prior_means):
        '''
        Client to update the GMM prior mean poses.
        '''
        #Add average grasp config
        prior_poses = []
        for config in prior_means:
            hand_config = self.proc_grasp.convert_preshape_to_full_config(config) 
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

    def leave_one_out(self, latent_dim, model='logistic', 
                    data_fraction=1., whiten=False):
        '''
        Cross validation for precision and power grasps classification.
        
        Args:
                latent_dim: the dimension of the latent space. Don't do dimension reduction when 
                it is -1.

        '''
        # Read and preprocess the data
        grasp_voxel_grids = np.copy(self.grasp_voxel_grids)
        grasp_configs = np.copy(self.grasp_configs)
        grasp_labels = np.copy(self.grasp_labels)
        grasp_voxel_grids = np.array(grasp_voxel_grids)
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
        loo = LeaveOneOut()
        train_test_indices = loo.split(grasp_voxel_config)
        
        test_accuracy = [] 
        train_accuracy = []
        test_pred_probs = []
        test_true_labels = []
        for train_indices, test_indices in train_test_indices:
            # Sample a fraction of grasps and verify smaller data-sets
            if data_fraction < 1.:
                train_indices = np.random.choice(train_indices, 
                        int(data_fraction * len(train_indices)), replace=False)
                test_indices = np.random.choice(test_indices, 
                        int(data_fraction * len(test_indices)), replace=False)

            print train_indices
            print test_indices

            X_train = grasp_voxel_config[train_indices]
            Y_train = grasp_labels[train_indices]
            X_test = grasp_voxel_config[test_indices] 
            Y_test = grasp_labels[test_indices]

            if latent_dim != -1:
                run_pca = True
                if run_pca:
                    # PCA
                    pca = PCA(n_components=latent_dim, whiten=whiten, svd_solver='full')
                    pca.fit(X_train)
                    print X_train.shape
                    X_train = pca.transform(X_train)
                    print X_train.shape
                    print X_test.shape
                    X_test = pca.transform(X_test)
                    print X_test.shape
                else:
                    # PPCA
                    ppca = prob_pca.PPCA(q=latent_dim)
                    X_train_trans = np.transpose(X_train)
                    X_test_trans = np.transpose(X_test)
                    ppca.fit(X_train_trans)
                    X_train_trans = ppca.transform(X_train_trans)
                    print X_test_trans.shape
                    X_test_trans = ppca.transform(X_test_trans)
                    print X_test_trans.shape
                    X_train = np.transpose(X_train_trans)
                    X_test = np.transpose(X_test_trans)


            X_train = np.concatenate((X_train, grasp_configs[train_indices]), axis=1)
            X_test = np.concatenate((X_test, grasp_configs[test_indices]), axis=1)

            #X_train = grasp_configs[train_indices]
            #X_test = grasp_configs[test_indices]

            # logistic regression
            if model == 'logistic':
                logistic = linear_model.LogisticRegression() #C=0.5)
                logistic.fit(X_train, Y_train)
                #print np.mean(logistic.coef_[0, :14])
                #print np.mean(logistic.coef_[0, 14:])
                #print np.mean(X_train[:, :14])
                #print np.mean(X_train[:, 14:])
                #print logistic.intercept_
                #print logistic.n_iter_
                pred_train = logistic.predict(X_train)
                pred_prob_train = logistic.predict_proba(X_train)
                print 'Training pred success grasps number:', np.sum(pred_train)
                train_acc = 1. - np.mean(np.abs(pred_train - Y_train))
                print('Training logistic regression accuracy: %f' % train_acc)
                train_accuracy.append(train_acc) 
                pred_test = logistic.predict(X_test)
                pred_prob_test = logistic.predict_proba(X_test)
                print 'Testing pred success grasps number:', np.sum(pred_test)
                test_acc = 1. - np.mean(np.abs(pred_test - Y_test))
                print('Testing logistic regression accuracy: %f' % test_acc)
                test_accuracy.append(test_acc) 
                test_pred_probs.append(pred_prob_test[0][1])
                test_true_labels.append(Y_test[0])

            # KNN
            if model == 'knn':
                knn = neighbors.KNeighborsClassifier()
                knn.fit(X_train, Y_train)
                pred = knn.predict(X_test)
                print 'pred success grasps number:', np.sum(pred)
                score = 1. - np.mean(np.abs(pred - Y_test))
                #score = knn.score(X_test, Y_test)
                print('KNN score: %f' % score)
                test_accuracy.append(score) 

            # SVM 
            if model == 'svm':
                svm = SVC() 
                svm.fit(X_train, Y_train)
                pred = svm.predict(X_test)
                print 'pred success grasps number:', np.sum(pred)
                score = 1. - np.mean(np.abs(pred - Y_test))
                #score = svm.score(X_test, Y_test)
                print('SVM score: %f' % score)
                test_accuracy.append(score) 

            # MAP with Gaussian or GMM
            if model in ['map_gaus', 'map_gmm']:
                map_gaus = mc.MapClassifier(prob=model.split('_')[1])
                map_gaus.fit(X_train, Y_train)
                pred_prob = map_gaus.predict_prob(X_test)
                print pred_prob
                pred = (pred_prob[:, 1] > 0.5).astype(int)
                print 'pred success grasps number:', np.sum(pred)
                score = 1. - np.mean(np.abs(pred - Y_test))
                #score = svm.score(X_test, Y_test)
                print('MAP Gaussian score: %f' % score)
                test_accuracy.append(score) 

        print train_accuracy
        print 'train_accurcy:', np.mean(train_accuracy)
        print test_accuracy
        print 'test_accuracy:', np.mean(test_accuracy)

        if self.all_type_grasp_clf:
            grasps_num = len(train_accuracy)
            power_num = grasps_num / 2
            print 'power train_accurcy:', np.mean(train_accuracy[:power_num])
            print 'power test_accuracy:', np.mean(test_accuracy[:power_num])
            print 'prec train_accurcy:', np.mean(train_accuracy[power_num:])
            print 'prec test_accuracy:', np.mean(test_accuracy[power_num:])

            gt_labels_file_name = self.leave_one_out_path + '/gt_labels_loo_power.txt'
            np.savetxt(gt_labels_file_name, test_true_labels[:power_num])
            pred_score_file_name = self.leave_one_out_path + '/pred_score_loo_power.txt'
            np.savetxt(pred_score_file_name, test_pred_probs[:power_num])

            gt_labels_file_name = self.leave_one_out_path + '/gt_labels_loo_prec.txt'
            np.savetxt(gt_labels_file_name, test_true_labels[power_num:])
            pred_score_file_name = self.leave_one_out_path + '/pred_score_loo_prec.txt'
            np.savetxt(pred_score_file_name, test_pred_probs[power_num:])
        else:
            gt_labels_file_name = self.leave_one_out_path + '/gt_labels_loo.txt'
            np.savetxt(gt_labels_file_name, test_true_labels)
            pred_score_file_name = self.leave_one_out_path + '/pred_score_loo.txt'
            np.savetxt(pred_score_file_name, test_pred_probs)

        #print 'test_pred_probs:', test_pred_probs
        #plot_curve.plot_roc_curve(grasp_labels, test_pred_probs, self.roc_fig_path)


    def cross_validation(self, folds_num=5, latent_dim=15, cv_idx=-1):
        '''
        Cross validation for precision and power grasps classification.
        
        Args:
                latent_dim: the dimension of the latent space. Don't do dimension reduction when 
                it is -1.

        '''
        # Read and preprocess the data
        grasp_voxel_grids = np.copy(self.grasp_voxel_grids)
        grasp_configs = np.copy(self.grasp_configs)
        grasp_labels = np.copy(self.grasp_labels)
        grasp_voxel_grids = np.array(grasp_voxel_grids)
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
        
        skf = StratifiedKFold(n_splits=folds_num)
        train_test_indices = skf.split(grasp_voxel_config, grasp_labels)

        test_accuracy = [] 
        train_accuracy = []
        test_pred_probs = []
        test_true_labels = []
        whiten = False
        for k, (train_indices, test_indices) in enumerate(train_test_indices):
            X_train = grasp_voxel_config[train_indices]
            Y_train = grasp_labels[train_indices]
            X_test = grasp_voxel_config[test_indices] 
            Y_test = grasp_labels[test_indices]

            if latent_dim != -1:
                run_pca = True
                if run_pca:
                    # PCA
                    pca = PCA(n_components=latent_dim, whiten=whiten, svd_solver='full')
                    pca.fit(X_train)
                    print X_train.shape
                    X_train = pca.transform(X_train)
                    print X_train.shape
                    print X_test.shape
                    X_test = pca.transform(X_test)
                    print X_test.shape
                else:
                    # PPCA
                    ppca = prob_pca.PPCA(q=latent_dim)
                    X_train_trans = np.transpose(X_train)
                    X_test_trans = np.transpose(X_test)
                    ppca.fit(X_train_trans)
                    X_train_trans = ppca.transform(X_train_trans)
                    print X_test_trans.shape
                    X_test_trans = ppca.transform(X_test_trans)
                    print X_test_trans.shape
                    X_train = np.transpose(X_train_trans)
                    X_test = np.transpose(X_test_trans)


            X_train = np.concatenate((X_train, grasp_configs[train_indices]), axis=1)
            X_test = np.concatenate((X_test, grasp_configs[test_indices]), axis=1)

            # logistic regression
            logistic = linear_model.LogisticRegression() #C=0.5)
            logistic.fit(X_train, Y_train)
            #print np.mean(logistic.coef_[0, :14])
            #print np.mean(logistic.coef_[0, 14:])
            #print np.mean(X_train[:, :14])
            #print np.mean(X_train[:, 14:])
            #print logistic.intercept_
            #print logistic.n_iter_
            pred_train = logistic.predict(X_train)
            pred_prob_train = logistic.predict_proba(X_train)
            print 'Training pred success grasps number:', np.sum(pred_train)
            train_acc = 1. - np.mean(np.abs(pred_train - Y_train))
            print('Training logistic regression accuracy: %f' % train_acc)
            train_accuracy.append(train_acc) 
            pred_test = logistic.predict(X_test)
            pred_prob_test = logistic.predict_proba(X_test)
            print 'Testing pred success grasps number:', np.sum(pred_test)
            test_acc = 1. - np.mean(np.abs(pred_test - Y_test))
            print('Testing logistic regression accuracy: %f' % test_acc)
            test_accuracy.append(test_acc) 
            test_pred_probs = np.concatenate((test_pred_probs, pred_prob_test[:, 1]))
            print test_pred_probs
            test_true_labels = np.concatenate((test_true_labels, Y_test))

        print train_accuracy
        print np.mean(train_accuracy)
        print test_accuracy
        print np.mean(test_accuracy)

        gt_labels_file_name = self.cross_val_path + '/gt_labels_cv_' + str(cv_idx) + '.txt'
        np.savetxt(gt_labels_file_name, test_true_labels)
        pred_score_file_name = self.cross_val_path + '/pred_score_cv_' + str(cv_idx) + '.txt'
        np.savetxt(pred_score_file_name, test_pred_probs)


if __name__ == '__main__':
    grasp_learner = GraspPgmLearner()
    #latent_dim = -1
    latent_dim = 15
    #seen_or_unseen = 'unseen'
    seen_or_unseen = 'seen'
    model = 'logistic'
    #model = 'map_gaus'
    #model = 'map_gmm'
    data_fraction = 1.
    #whiten = True
    whiten = False
    #cv_res = grasp_learner.leave_one_out(latent_dim, model, data_fraction, whiten)
    grasp_learner.train_all(latent_dim, model, whiten)
    grasp_learner.fit_prior()

    #for i in xrange(10):
    #    grasp_learner.cross_validation(folds_num=5, cv_idx=i)
    #    print '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$'

    #grasp_learner.gmm_cluster_grasp_types(latent_dim, seen_or_unseen, whiten)
    #grasp_learner.kmeans_cluster_grasp_types(latent_dim, seen_or_unseen, whiten)
