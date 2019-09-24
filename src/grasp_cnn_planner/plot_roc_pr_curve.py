import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.metrics import precision_recall_fscore_support

def plot_roc_curve(y_true, y_pred, fig_name):
    '''
    Plot ROC curve.
    '''
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)        
    roc_auc = auc(fpr, tpr) 
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', 
            label='Baseline: random guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    #plt.show()
    plt.savefig(fig_name)
    #plt.clf()
    plt.close()

def plot_pr_curve(y_true, y_pred, fig_name):
    '''
    Plot PR curve.
    '''
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    average_precision = average_precision_score(y_true, y_pred)
    plt.figure()
    lw = 2
    #plt.plot(recall, precision, lw=lw, color='navy',
    #             label='Precision-Recall curve')
    plt.plot(recall, precision, color='darkorange',
                 lw=lw, label='PR curve (area = %0.2f)' % average_precision)
    pos_rate = np.sum(y_true) / np.shape(y_true)[0]
    plt.plot([0, 1], [pos_rate, pos_rate], color='navy', lw=lw, linestyle='--',
            label='Baseline: random guess (area = %0.2f)' % pos_rate)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    #plt.show()
    plt.savefig(fig_name)
    plt.close()

def plot_average_roc_curve(y_true_list, y_pred_list, fig_name, seen_or_unseen):
    '''
    Plot average ROC curve for cross validation.
    '''
    k_folds = len(y_true_list)
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    roc_auc_list = []
    for i, y_true in enumerate(y_true_list):
        y_pred = y_pred_list[i]
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)        
        roc_auc = auc(fpr, tpr) 
        roc_auc_list.append(roc_auc)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0

    print 'roc_auc_list:', roc_auc_list
    roc_auc_list = np.array(roc_auc_list)
    roc_auc_mean = np.mean(roc_auc_list)
    roc_auc_std = np.std(roc_auc_list)
    print 'roc auc mean:', roc_auc_mean
    print 'roc auc std:', roc_auc_std

    plt.figure()
    lw = 2
    mean_tpr /= float(k_folds)
    #cv.get_n_splits(X, y)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    print 'mean_auc:', mean_auc
    plt.plot(mean_fpr, mean_tpr, color='b', 
            label='Cross validation mean ROC.\nAUC mean = %0.2f, AUC std = %0.2f.'
            % (mean_auc, roc_auc_std), lw=lw)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
                 label='Random guess ROC. AUC = %0.2f.' %0.5)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Cross Validation Mean ROC Curve' + ' for ' + seen_or_unseen + ' Objects')
    plt.legend(loc="lower right")
    #plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #                   ncol=2, mode="expand", borderaxespad=0.)
    #plt.show()
    plt.savefig(fig_name)
    #plt.clf()
    plt.close()

def compute_accuracy_f1(y_true, y_pred, threshold):
    '''
    Comptue the accuracy and f1 score of a given threshold. 
    '''
    pred_binary = np.copy(y_pred)
    pred_binary[pred_binary > threshold] = 1.
    pred_binary[pred_binary <= threshold] = 0.

    accuracy = 1. - np.mean(np.abs(pred_binary - y_true))
    precision, recall, fscore, support = \
            precision_recall_fscore_support(y_true, pred_binary, average='binary')

    pos_rate = np.sum(y_true) / np.shape(y_true)[0]
    rand_recall = 0.5
    rand_f_score = 2. * pos_rate * rand_recall / (pos_rate + rand_recall) 
    return accuracy, fscore, rand_f_score

def plot_accuracy_f1_cv(y_true_list, y_pred_list, fig_name, seen_or_unseen):
    '''
    Compute and plot the accuracy and f1 score for cross validation with
    different thresholds.
    '''
    k_folds = len(y_true_list)
    thresh_num = 101
    thresholds = np.linspace(0, 1, thresh_num)
    accuracy = np.zeros((k_folds, thresh_num))
    f_score = np.zeros((k_folds, thresh_num))
    rand_f_score = np.zeros((k_folds, thresh_num))
    for i, y_true in enumerate(y_true_list):
        y_pred = y_pred_list[i]
        for j, t in enumerate(thresholds): 
            accuracy[i, j], f_score[i, j], rand_f_score[i, j] = \
                    compute_accuracy_f1(y_true, y_pred, t)

    mean_acc = np.mean(accuracy, axis=0)
    #print 'mean_acc:', mean_acc
    mean_f_score = np.mean(f_score, axis=0) 
    mean_rand_f_score = np.mean(rand_f_score, axis=0)

    plt.figure()
    lw = 2
    plt.plot(thresholds, mean_acc, color='b', lw=lw, label='Mean accuracy.')
    plt.plot(thresholds, mean_f_score, color='g', lw=lw, label='Mean f1_score.')
    plt.plot(thresholds, mean_rand_f_score, color='k', lw=lw, label='Mean random f1 score.')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Thresholds')
    plt.ylabel('Metrics')
    plt.title('Cross Validation Mean Metrics' + ' for ' + seen_or_unseen + ' Objects')
    plt.legend(loc="lower right")
    plt.savefig(fig_name)
    plt.close()

def compute_metrics_one_thresh_cv(y_true_list, y_pred_list, threshold):
    '''
    Compute the mean and std of accuracy and f1 score at a certian threshold for 
    cross validation.
    '''
    k_folds = len(y_true_list)
    accuracy = np.zeros(k_folds)
    f_score = np.zeros(k_folds)
    rand_f_score = np.zeros(k_folds)
    for i, y_true in enumerate(y_true_list):
        y_pred = y_pred_list[i]
        accuracy[i], f_score[i], rand_f_score[i] = \
                compute_accuracy_f1(y_true, y_pred, threshold)

    mean_acc = np.mean(accuracy)
    mean_f_score = np.mean(f_score) 
    mean_rand_f_score = np.mean(rand_f_score)

    std_acc = np.std(accuracy)
    std_f_score = np.std(f_score) 
    std_rand_f_score = np.std(rand_f_score)
  
    print 'mean_acc, mean_f_score, mean_rand_f_score, \
            std_acc, std_f_score, std_rand_f_score'
    print mean_acc, mean_f_score, mean_rand_f_score, \
            std_acc, std_f_score, std_rand_f_score

    return mean_acc, mean_f_score, mean_rand_f_score, \
            std_acc, std_f_score, std_rand_f_score

def main():
    k_folds = 5
    #seen_or_unseen = 'seen'
    seen_or_unseen = 'unseen'
    #cross_val_path = '../cross_val/'
    #cross_val_path = '../cross_val_oversample1_v1/'
    cross_val_path = '/data_space/data_kai/grasp_cnn_analysis/data_v4/cross_val_oversample1/'
    y_true_list = []
    y_pred_list = []
    for k_fold in xrange(k_folds):
        gt_labels_file_name = cross_val_path + seen_or_unseen + '/gt_labels_fold_' + str(k_fold) + '.txt'
        y_true = np.loadtxt(gt_labels_file_name)
        y_true_list.append(y_true)
        pred_score_file_name = cross_val_path + seen_or_unseen + '/pred_score_fold_' + str(k_fold) + '.txt'
        y_pred = np.loadtxt(pred_score_file_name)
        y_pred_list.append(y_pred)

    #print y_true_list, y_pred_list
    fig_name = cross_val_path + seen_or_unseen + '/avg_roc.png'
    plot_average_roc_curve(y_true_list, y_pred_list, fig_name, seen_or_unseen.title())

    fig_name = cross_val_path + seen_or_unseen + '/avg_acc_f1.png'
    plot_accuracy_f1_cv(y_true_list, y_pred_list, fig_name, seen_or_unseen.title())

    compute_metrics_one_thresh_cv(y_true_list, y_pred_list, threshold=0.4)

if __name__ == '__main__':
    main()

