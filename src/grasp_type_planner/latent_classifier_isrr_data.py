#!/usr/bin/env python
import numpy as np
import time
import h5py
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn import neighbors, linear_model
from sklearn.svm import SVC
import cross_validation
import plot_roc_pr_curve as plot_curve
import matplotlib.pyplot as plt
import map_classifier as mc

import os
os.sys.path.append('./ppca/src/pca')
#import pca
import ppca as prob_pca

class LatentClassifier:
    '''
    Grasp success classifier in different latent space.
    '''
    def __init__(self):
        self.data_path = '/media/kai/multi_finger_sim_data_complete_v4/'
        #Uses voxel grids and grasp configs in object tf or not
        #self.object_tf = False
        self.object_tf = True
        if not self.object_tf:
            self.grasp_voxel_file_path = self.data_path + 'grasp_voxel_data.h5'
        else:
            self.grasp_voxel_file_path = self.data_path + 'grasp_voxel_obj_tf_data.h5'
        self.grasp_voxel_grids, self.grasp_configs, self.grasp_labels = \
                                                    self.read_grasp_voxel_data()

    def read_grasp_voxel_data(self):
        grasp_voxel_file = h5py.File(self.grasp_voxel_file_path, 'r')
        grasp_voxel_grids = grasp_voxel_file['grasp_voxel_grids'][()] 
        if not self.object_tf:
            grasp_configs = grasp_voxel_file['grasp_configs'][()]
        else:
            grasp_configs = grasp_voxel_file['grasp_configs_obj'][()]
        grasp_labels = grasp_voxel_file['grasp_labels'][()]
        grasp_voxel_file.close()
        return grasp_voxel_grids, grasp_configs, grasp_labels

    def read_grasp_indices(self, grasps_num):
        '''
        Read the grasp indices of these grasps having pcd files.
        '''
        grasp_voxel_file = h5py.File(self.grasp_voxel_file_path, 'r')
        grasp_id_map = {}
        for i in xrange(grasps_num):
            voxel_grasp_id_key = 'voxel_grasp_' + str(i)
            grasp_id = grasp_voxel_file[voxel_grasp_id_key][()].split('_')[1]
            grasp_id_map[int(grasp_id)] = i
        grasp_voxel_file.close()
        return grasp_id_map

    def classify_with_pca(self):
        # Read and preprocess the data
        grasp_voxel_grids, grasp_configs, grasp_labels = self.read_grasp_voxel_data()
        grasp_voxel_grids = np.array(grasp_voxel_grids)
        grasps_num = grasp_voxel_grids.shape[0]
        voxel_grid_dim = grasp_voxel_grids[0].shape
        voxel_num = voxel_grid_dim[0] * voxel_grid_dim[1] * voxel_grid_dim[2]  
        grasp_1d_voxel_grids = np.reshape(grasp_voxel_grids, (grasps_num, voxel_num))
        grasp_voxel_config = np.concatenate((grasp_1d_voxel_grids, grasp_configs), axis=1)
        #grasp_voxel_config = grasp_configs
        #grasp_voxel_config = grasp_1d_voxel_grids
        
        # Shuffule the data for cross validation.
        print 'Total number of grasps:', grasps_num
        print 'Number of success grasps', np.sum(grasp_labels)
        folds_num = 5
        skf = StratifiedKFold(n_splits=folds_num)
        train_test_indices = skf.split(grasp_voxel_config, grasp_labels)
        
        model = 'logistic'
        #model = 'knn'
        #model = 'svm'
        test_accuracy = [] 
        for train_indices, test_indices in train_test_indices:
            X_train = grasp_voxel_config[train_indices]
            Y_train = grasp_labels[train_indices]
            X_test = grasp_voxel_config[test_indices] 
            Y_test = grasp_labels[test_indices]

            ## PCA
            #pca = PCA(n_components=50)
            ##pca = PCA(n_components='mle', svd_solver='full')
            #pca.fit(X_train)
            #X_train = pca.transform(X_train)
            #print X_test.shape
            #X_test = pca.transform(X_test)
            #print X_test.shape

            # PPCA
            ppca = prob_pca.PPCA(q=100, sigma=0.001)
            #pca = PCA(n_components='mle', svd_solver='full')
            X_train_trans = np.transpose(X_train)
            X_test_trans = np.transpose(X_test)
            ppca.fit(X_train_trans, em=False)
            print ppca.sigma
            X_train_trans = ppca.transform(X_train_trans)
            print X_test_trans.shape
            X_test_trans = ppca.transform(X_test_trans)
            print X_test_trans.shape
            X_train = np.transpose(X_train_trans)
            X_test = np.transpose(X_test_trans)

            # logistic regression
            if model == 'logistic':
                logistic = linear_model.LogisticRegression()
                logistic.fit(X_train, Y_train)
                pred = logistic.predict(X_test)
                print 'pred success grasps number:', np.sum(pred)
                score = 1. - np.mean(np.abs(pred - Y_test))
                #score = logistic.score(X_test, Y_test)
                print('LogisticRegression score: %f' % score)
                test_accuracy.append(score) 

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
        print test_accuracy
        print np.mean(test_accuracy)

    def cross_val(self, latent_dim, seen_or_unseen='seen', model='logistic', 
                    data_fraction=1., whiten=False):
        '''
        Cross validation for seen and unseen objects.
        
        Args:
                latent_dim: the dimension of the latent space. Don't do dimension reduction when 
                it is -1.

        '''
        # Read and preprocess the data
        #grasp_voxel_grids, grasp_configs, grasp_labels = self.read_grasp_voxel_data()
        grasp_voxel_grids = np.copy(self.grasp_voxel_grids)
        grasp_configs = np.copy(self.grasp_configs)
        grasp_labels = np.copy(self.grasp_labels)
        grasp_voxel_grids = np.array(grasp_voxel_grids)
        grasps_num = grasp_voxel_grids.shape[0]
        grasp_id_map = self.read_grasp_indices(grasps_num) 
        voxel_grid_dim = grasp_voxel_grids[0].shape
        voxel_num = voxel_grid_dim[0] * voxel_grid_dim[1] * voxel_grid_dim[2]  
        grasp_1d_voxel_grids = np.reshape(grasp_voxel_grids, (grasps_num, voxel_num))
        grasp_voxel_config = np.concatenate((grasp_1d_voxel_grids, grasp_configs), axis=1)
        #grasp_voxel_config = grasp_configs
        #grasp_voxel_config = grasp_1d_voxel_grids
        
        # Shuffule the data for cross validation.
        print 'Total number of grasps:', grasps_num
        print 'Number of success grasps', np.sum(grasp_labels)
        folds_num = 5
        #skf = StratifiedKFold(n_splits=folds_num)
        #train_test_indices = skf.split(grasp_voxel_config, grasp_labels)
        
        #To do: random forest
        test_accuracy = [] 
        train_file_name = '../cross_val_isrr/cross_val_train_' + seen_or_unseen
        test_file_name = '../cross_val_isrr/cross_val_test_' + seen_or_unseen
        test_true_y_list = []
        test_pred_y_list = []
        for k in xrange(folds_num):
            train_indices_raw, test_indices_raw = \
            cross_validation.get_k_fold_train_test_indices(k, train_file_name, test_file_name)
            train_indices = []
            for t in train_indices_raw:
                if t in grasp_id_map:
                    train_indices.append(grasp_id_map[t])
            test_indices = []
            for t in test_indices_raw:
                if t in grasp_id_map:
                    test_indices.append(grasp_id_map[t])

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

            #X_train = grasp_configs[train_indices]
            #X_test = grasp_configs[test_indices]

            # logistic regression
            if model == 'logistic':
                logistic = linear_model.LogisticRegression()
                logistic.fit(X_train, Y_train)
                #print np.mean(logistic.coef_[rrayIrray0, :14])
                #print np.mean(logistic.coef_[0, 14`:])
                #print np.mean(X_train[:, :14])
                #print np.mean(X_train[:, 14:])
                #print logistic.intercept_
                #print logistic.n_iter_
                pred = logistic.predict(X_test)
                pred_prob = logistic.predict_proba(X_test)
                print 'pred success grasps number:', np.sum(pred)
                score = 1. - np.mean(np.abs(pred - Y_test))
                #score = logistic.score(X_test, Y_test)
                print('LogisticRegression score: %f' % score)
                test_accuracy.append(score) 

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
                pred = (pred_prob[:, 1] > 0.5).astype(int)
                print 'pred success grasps number:', np.sum(pred)
                score = 1. - np.mean(np.abs(pred - Y_test))
                #score = svm.score(X_test, Y_test)
                print('MAP Gaussian score: %f' % score)
                test_accuracy.append(score) 

            ## MAP with GMM
            #if model == 'map_gmm':
            #    map_gaus = mc.MapClassifier(prob='gmm')
            #    map_gaus.fit(X_train, Y_train)
            #    pred = map_gaus.predict(X_test)
            #    print 'pred success grasps number:', np.sum(pred)
            #    score = 1. - np.mean(np.abs(pred - Y_test))
            #    #score = svm.score(X_test, Y_test)
            #    print('MAP Gaussian score: %f' % score)
            #    test_accuracy.append(score) 


            gt_labels_file_name = '../cross_val_isrr/' + seen_or_unseen + '/gt_labels_fold_' + str(k) + '.txt'
            np.savetxt(gt_labels_file_name, Y_test)
            pred_score_file_name = '../cross_val_isrr/' + seen_or_unseen + '/pred_score_fold_' + str(k) + '.txt'
            test_true_y_list.append(Y_test)
            if model in ['logistic', 'map_gaus', 'map_gmm']:
                np.savetxt(pred_score_file_name, pred_prob[:, 1])
                test_pred_y_list.append(pred_prob[:, 1])
            else:
                np.savetxt(pred_score_file_name, pred)
            #print pred

        print test_accuracy
        print np.mean(test_accuracy)

        cv_roc_auc = plot_curve.get_cv_roc_auc(test_true_y_list, test_pred_y_list)
        cv_pr_auc = plot_curve.get_cv_pr_auc(test_true_y_list, test_pred_y_list)
        cv_acc_list = []
        cv_f_score_list = []
        #thresh_list = [0.2, 0.3, 0.4, 0.5, 0.6]
        #thresh_list = [0.5]
        thresh_list = [0.4, 0.5, 0.6]
        #thresh_list = np.linspace(0.1, 1., 10)
        for thresh in thresh_list: 
            mean_acc, mean_f_score, _, _, _, _ = \
                    plot_curve.compute_metrics_one_thresh_cv(test_true_y_list, 
                            test_pred_y_list, threshold=thresh)
            cv_acc_list.append(mean_acc)
            cv_f_score_list.append(mean_f_score)
        cv_mean_acc = np.mean(cv_acc_list)
        cv_mean_f_score = np.mean(cv_f_score_list)
        return cv_roc_auc, cv_pr_auc, cv_mean_acc, cv_mean_f_score 

    def evaluate_latent_dim(self, seen_or_unseen, model, data_fraction, whiten):
        #dims_num = 200
        #max_dim = 8014
        #latent_dims = np.linspace(10, max_dim, dims_num).astype(int)
        #latent_dims[-1] = -1
        dims_num = 200 
        max_dim = 399
        latent_dims = np.linspace(1, max_dim, dims_num).astype(int)
        roc_auc = np.zeros(dims_num)
        pr_auc = np.zeros(dims_num)
        mean_acc = np.zeros(dims_num)
        mean_f_score = np.zeros(dims_num)
        for i, dim in enumerate(latent_dims): 
            print '##########'
            print 'dim:', dim
            roc_auc[i], pr_auc[i], mean_acc[i], mean_f_score[i] = \
                    self.cross_val(dim, seen_or_unseen, model, data_fraction, whiten)
        latent_dims[-1] = max_dim

        if whiten:
            fig_name = '../cross_val_isrr/' + seen_or_unseen + '/latent_dim_frac_' + \
                        str(int(data_fraction * 100)) + '_whiten.png'
        else:
            fig_name = '../cross_val_isrr/' + seen_or_unseen + '/latent_dim_frac_' + \
                        str(int(data_fraction * 100)) + '.png'

        plt.figure()
        lw = 2
        print roc_auc, pr_auc, mean_acc, mean_f_score
        plt.plot(latent_dims, roc_auc, color='b', lw=lw, label='ROC auc.')
        plt.plot(latent_dims, pr_auc, color='g', lw=lw, label='PR auc.')
        plt.plot(latent_dims, mean_acc, color='k', lw=lw, label='Mean acc.')
        plt.plot(latent_dims, mean_f_score, color='r', lw=lw, label='Mean f1.')
        plt.xlim([0, max_dim + 10])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Latent dim')
        plt.ylabel('Metrics')
        plt.title('Latent dimension' + ' for ' + seen_or_unseen + ' Objects')
        plt.legend(loc="lower right")
        plt.savefig(fig_name)
        plt.close()

    def evaluate_data_size(self, seen_or_unseen, model, latent_dim, whiten):
        data_sizes_num = 500
        data_sizes = np.linspace(0.05, 1., data_sizes_num)
        roc_auc = np.zeros(data_sizes_num)
        pr_auc = np.zeros(data_sizes_num)
        mean_acc = np.zeros(data_sizes_num)
        mean_f_score = np.zeros(data_sizes_num)
        for i, d_size in enumerate(data_sizes): 
            roc_auc[i], pr_auc[i], mean_acc[i], mean_f_score[i] = \
                    self.cross_val(latent_dim, seen_or_unseen, model, d_size, whiten)

        if whiten:
            fig_name = '../cross_val_isrr/' + seen_or_unseen + '/data_size_dim_' + \
                        str(latent_dim) + '_whiten.png'
        else:
            fig_name = '../cross_val_isrr/' + seen_or_unseen + '/data_size_dim_' + \
                        str(latent_dim) + '.png'

        plt.figure()
        lw = 2
        print roc_auc, pr_auc, mean_acc, mean_f_score
        plt.plot(data_sizes, roc_auc, color='b', lw=lw, label='ROC auc.')
        plt.plot(data_sizes, pr_auc, color='g', lw=lw, label='PR auc.')
        plt.plot(data_sizes, mean_acc, color='k', lw=lw, label='Mean acc.')
        plt.plot(data_sizes, mean_f_score, color='r', lw=lw, label='Mean f1.')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Data size')
        plt.ylabel('Metrics')
        plt.title('Data percentage' + ' for ' + seen_or_unseen + ' objects')
        plt.legend(loc="lower right")
        plt.savefig(fig_name)
        plt.close()



if __name__ == '__main__':
    latent_clf = LatentClassifier()
    #print latent_clf.grasp_labels.shape
    #print np.sum(latent_clf.grasp_labels)
    #latent_clf.classify_with_pca()

    #latent_dim = -1
    latent_dim = 15
    seen_or_unseen = 'unseen'
    #seen_or_unseen = 'seen'
    model = 'logistic'
    #model = 'map_gaus'
    #model = 'map_gmm'
    data_fraction = 1.
    #whiten = True
    whiten = False
    #cv_res = latent_clf.cross_val(latent_dim, seen_or_unseen, model, data_fraction, whiten)
    #print cv_res
    #latent_clf.evaluate_latent_dim(seen_or_unseen, model, data_fraction, whiten)
    latent_clf.evaluate_data_size(seen_or_unseen, model, latent_dim, whiten)
