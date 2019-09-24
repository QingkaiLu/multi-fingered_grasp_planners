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
#import plot_roc_pr_curve as plot_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import map_classifier as mc
import os
os.sys.path.append('./ppca/src/pca')
import ppca as prob_pca
import pickle
from scipy.stats import multivariate_normal

class GraspPgmLearnerSim:
    '''
    Test the grasp learner with generated simulation data.
    '''
    def __init__(self):
        self.power_data_path = '/media/kai/grasp_gaus_data/'
        #Successful power and precision grasp data.
        self.power_grasp_voxel_path = self.power_data_path + 'power_grasps/power_suc_data.h5'
        self.prec_data_path = '/media/kai/grasp_gaus_data/'
        self.prec_grasp_voxel_path = self.prec_data_path + 'prec_grasps/prec_suc_data.h5'
        #Failure power and precision grasp data.
        self.fail_power_grasp_voxel_path = self.power_data_path + 'power_grasps/power_failure_data.h5'
        self.fail_prec_grasp_voxel_path = self.prec_data_path + 'prec_grasps/prec_failure_data.h5'
        #Grasp types classification or grasp success classification 
        self.grasp_type_clf = False
        self.all_type_grasp_clf = False #True 
        self.power_grasp_clf = True #False
        self.save_pca_model = False #True

        self.pca_model_path = '../train_models_sim/pca/pca.model'
        #Grasp success classification for power or precision grasps.
        self.fail_indices_path = '../train_models_sim/rand_grasp_indices/'

        #self.load_grasp_data()

        self.train_model_path = '../train_models_sim/classifiers/'
        self.prior_path = '../train_models_sim/priors/'
        #self.prior_path = '../train_models_sim/gmm_priors/'
        self.roc_fig_path = '../cross_val_sim/plots/'
        self.cross_val_path = '../cross_val_sim/'
        if self.all_type_grasp_clf:
            self.train_model_path += 'all_type_clf' 
            self.prior_path += 'all_type_'
            self.roc_fig_path += 'roc_all.png'
            self.cross_val_path += 'all/'
        elif self.power_grasp_clf:
            self.train_model_path += 'power_clf' 
            self.prior_path += 'power_'
            self.roc_fig_path += 'roc_power.png'
            self.cross_val_path += 'power/'
        else:
            self.train_model_path += 'prec_clf' 
            self.prior_path += 'prec_'
            self.roc_fig_path += 'roc_prec.png'
            self.cross_val_path += 'prec/'

    def gen_sim_data(self, prec=True):
        '''
        Generate Gaussian simulation data.
        Adjust the mean and covariance of successful and failure grasps to make 
        simulation data satistfy these properties:
        1. both latent object voxel grid and grasp configuration should matter for the grasp success, 
        but grasp configuration should be more important than latent voxel grid;
        2. for most latent object voxel grids, there should exist both successful and failure grasp configurations;
        3. for most grasp configurations, there should exist both successful and failure latent voxel grids.  
        '''
        voxel_dim = 800 #15
        grasp_config_dim = 14
        data_dim = voxel_dim + grasp_config_dim
        if prec:
            voxel_mean = 0.5 * np.ones(voxel_dim)
        else:
            voxel_mean = -0.5 * np.ones(voxel_dim)

        data_num = 500 #35

        #Generate grasp data.
        #suc_data_mean = [0.5, 1.]
        #suc_data_covar = 2. * np.array([[0.5, 0.2], [0.2, 0.5]])

        suc_config_mean = np.ones(grasp_config_dim) 
        suc_data_mean = np.concatenate((voxel_mean, suc_config_mean))
        #suc_data_covar = 0.2 * np.ones((data_dim, data_dim)) + 0.3 * np.identity(data_dim)
        suc_data_covar = 2 * np.ones((data_dim, data_dim)) + 2.5 * np.identity(data_dim)

        suc_data = np.random.multivariate_normal(suc_data_mean, suc_data_covar, data_num)
        suc_latent_voxels = suc_data[:, :voxel_dim] 
        suc_grasp_configs = suc_data[:, voxel_dim:] 

        #fail_data_mean = [0.5, -1.]
        #fail_data_covar = 2. * np.array([[0.5, 0.2], [0.2, 0.5]])

        fail_config_mean = -np.ones(grasp_config_dim) 
        fail_data_mean = np.concatenate((voxel_mean, fail_config_mean))
        #fail_data_covar = 0.2 * np.ones((data_dim, data_dim)) + 0.3 * np.identity(data_dim)
        fail_data_covar = 2 * np.ones((data_dim, data_dim)) + 2.5 * np.identity(data_dim)

        fail_data = np.random.multivariate_normal(fail_data_mean, fail_data_covar, data_num)
        fail_latent_voxels = fail_data[:, :voxel_dim] 
        fail_grasp_configs = fail_data[:, voxel_dim:] 

        #print 'suc_data:', suc_data
        #print 'fail_data:', fail_data

        #Write grasp data.
        if prec:
            suc_data_path = self.prec_grasp_voxel_path
            fail_data_path = self.fail_prec_grasp_voxel_path 
        else:
            suc_data_path = self.power_grasp_voxel_path
            fail_data_path = self.fail_power_grasp_voxel_path 
        
        grasp_suc_data_file = h5py.File(suc_data_path, 'w')
        grasp_suc_data_file.create_dataset('grasp_voxel_grids', data=suc_latent_voxels)
        grasp_suc_data_file.create_dataset('grasp_configs_obj', data=suc_grasp_configs)
        grasp_suc_data_file.close()

        grasp_fail_data_file = h5py.File(fail_data_path, 'w')
        grasp_fail_data_file.create_dataset('grasp_voxel_grids', data=fail_latent_voxels)
        grasp_fail_data_file.create_dataset('grasp_configs_obj', data=fail_grasp_configs)
        grasp_fail_data_file.close()

        #Plot grasp data.
        if prec:
            fig_name = '../train_models_sim/gaus_prec_data.png'
        else:
            fig_name = '../train_models_sim/gaus_power_data.png'

        #plt.figure()
        #plt.scatter(suc_latent_voxels, suc_grasp_configs, color='r', label='success')
        #plt.scatter(fail_latent_voxels, fail_grasp_configs, color='b', label='failure')
        #plt.legend(loc="lower right")
        #plt.xlabel('voxel')
        #plt.ylabel('config')
        ##plt.show()
        #plt.savefig(fig_name)
        #plt.close()

    def load_suc_or_failure_grasp(self, load_power, load_success):
        '''
        Load successful or failure grasp of one type (precision or power).
        '''
        grasp_voxel_path = None
        if load_success:
            #Load successful grasps.
            if load_power:
                grasp_voxel_path = self.power_grasp_voxel_path
            else:
                grasp_voxel_path = self.prec_grasp_voxel_path
        else:
            #Load failure grasps.
            if load_power:
                grasp_voxel_path = self.fail_power_grasp_voxel_path
            else:
                grasp_voxel_path = self.fail_prec_grasp_voxel_path

        return self.read_grasp_voxel_data(grasp_voxel_path)

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
        fail_grasp_labels = np.zeros(suc_grasps_num)
        grasp_voxel_grids = np.concatenate((suc_grasp_voxel_grids, fail_grasp_voxel_grids))
        grasp_configs = np.concatenate((suc_grasp_configs, fail_grasp_configs))
        grasp_labels = np.concatenate((suc_grasp_labels, fail_grasp_labels))

        return grasp_voxel_grids, grasp_configs, grasp_labels

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
            power_grasp_voxel_grids, power_grasp_configs, power_grasp_labels = \
                                        self.load_one_type_grasp(load_power=True)
            prec_grasp_voxel_grids, prec_grasp_configs, prec_grasp_labels = \
                                        self.load_one_type_grasp(load_power=False)
            self.grasp_voxel_grids = np.concatenate((power_grasp_voxel_grids, prec_grasp_voxel_grids))
            self.grasp_configs = np.concatenate((power_grasp_configs, prec_grasp_configs))
            self.grasp_labels = np.concatenate((power_grasp_labels, prec_grasp_labels))
        else:
            self.grasp_voxel_grids, self.grasp_configs, self.grasp_labels = \
                                        self.load_one_type_grasp(self.power_grasp_clf)

    def read_grasp_voxel_data(self, voxel_path):
        grasp_voxel_file = h5py.File(voxel_path, 'r')
        grasp_voxel_grids = grasp_voxel_file['grasp_voxel_grids'][()] 
        grasp_configs = grasp_voxel_file['grasp_configs_obj'][()]
        grasp_voxel_file.close()
        return grasp_voxel_grids, grasp_configs

    def leave_one_out(self, latent_dim=-1, model='logistic', 
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
        #voxel_num = voxel_grid_dim[0] * voxel_grid_dim[1] * voxel_grid_dim[2]  
        ##Convert voxel to 1d.
        #grasp_1d_voxel_grids = np.reshape(grasp_voxel_grids, (grasps_num, voxel_num))
        grasp_1d_voxel_grids = grasp_voxel_grids
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
        for train_indices, test_indices in train_test_indices:
            # Sample a fraction of grasps and verify smaller data-sets
            if data_fraction < 1.:
                train_indices = np.random.choice(train_indices, 
                        int(data_fraction * len(train_indices)), replace=False)
                test_indices = np.random.choice(test_indices, 
                        int(data_fraction * len(test_indices)), replace=False)

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

        #print train_accuracy
        print 'train_accuracy:', np.mean(train_accuracy)
        #print test_accuracy
        print 'test_accuracy:', np.mean(test_accuracy)

        #print 'test_pred_probs:', test_pred_probs
        #plot_curve.plot_roc_curve(grasp_labels, test_pred_probs, self.roc_fig_path)

    def train_all(self, latent_dim=-1, model='logistic', whiten=False):
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

        #voxel_num = voxel_grid_dim[0] * voxel_grid_dim[1] * voxel_grid_dim[2]  
        ##Convert voxel to 1d.
        #grasp_1d_voxel_grids = np.reshape(grasp_voxel_grids, (grasps_num, voxel_num))
        grasp_1d_voxel_grids = grasp_voxel_grids

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

        X_train = np.concatenate((X_train, grasp_configs), axis=1)

        # logistic regression
        if model == 'logistic':
            #TO DO: tune regularization to get better performance of logistic regression.
            logistic = linear_model.LogisticRegression() #C=0.5)
            logistic.fit(X_train, Y_train)
            print 'logistic.intercept_:', logistic.intercept_
            print 'logistic.coef_:', logistic.coef_
            print 'logistic.n_iter_:', logistic.n_iter_
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
        deprecated_gmm = False #True
        num_components = 2
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
        pickle.dump(g, open(self.prior_path + 'gmm.model', 'wb'))

        print 'weights_:', g.weights_
        print 'means_:', g.means_ 
        print 'covariances_:', g.covariances_

if __name__ == '__main__':
    sim_learner = GraspPgmLearnerSim()
    #sim_learner.gen_sim_data(prec=not sim_learner.power_grasp_clf)
    sim_learner.load_grasp_data()
    #sim_learner.leave_one_out()
    latent_dim = 15
    sim_learner.train_all(latent_dim)
    sim_learner.fit_prior()

