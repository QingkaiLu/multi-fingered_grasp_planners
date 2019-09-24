import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.metrics import precision_recall_fscore_support

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

    pos_rate = float(np.sum(y_true)) / np.shape(y_true)[0]
    rand_recall = 0.5
    rand_f_score = 2. * pos_rate * rand_recall / (pos_rate + rand_recall) 
    return accuracy, fscore, rand_f_score

def compute_metrics_one_thresh_cv(y_true_list, y_pred_list, threshold):
    '''
    Compute the mean and std of accuracy and f1 score at a certian threshold for 
    cross validation.
    '''
    cv_times = len(y_true_list)
    accuracy = np.zeros(cv_times)
    f_score = np.zeros(cv_times)
    rand_f_score = np.zeros(cv_times)
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

def plot_average_roc_curve_gtypes(y_true_list_prec, y_pred_list_prec, 
                                y_true_list_power, y_pred_list_power, 
                                y_true_list_all, y_pred_list_all, fig_name): 
    '''
    Plot average ROC curve for the cross validation of all grasp types.
    '''
    cv_times = len(y_true_list_prec)

    mean_tpr_prec = 0.0
    mean_fpr_prec = np.linspace(0, 1, 100)
    roc_auc_list_prec = []
    for i, y_true in enumerate(y_true_list_prec):
        y_pred = y_pred_list_prec[i]
        fpr_prec, tpr_prec, thresholds = roc_curve(y_true, y_pred)        
        roc_auc = auc(fpr_prec, tpr_prec) 
        roc_auc_list_prec.append(roc_auc)
        mean_tpr_prec += interp(mean_fpr_prec, fpr_prec, tpr_prec)
        mean_tpr_prec[0] = 0.0

    mean_tpr_prec /= float(cv_times)
    mean_tpr_prec[-1] = 1.0
    mean_auc_prec = auc(mean_fpr_prec, mean_tpr_prec)

    mean_tpr_power = 0.0
    mean_fpr_power = np.linspace(0, 1, 100)
    roc_auc_list_power = []
    for i, y_true in enumerate(y_true_list_power):
        y_pred = y_pred_list_power[i]
        fpr_power, tpr_power, thresholds = roc_curve(y_true, y_pred)        
        roc_auc = auc(fpr_power, tpr_power) 
        roc_auc_list_power.append(roc_auc)
        mean_tpr_power += interp(mean_fpr_power, fpr_power, tpr_power)
        mean_tpr_power[0] = 0.0

    mean_tpr_power /= float(cv_times)
    mean_tpr_power[-1] = 1.0
    mean_auc_power = auc(mean_fpr_power, mean_tpr_power)

    mean_tpr_all = 0.0
    mean_fpr_all = np.linspace(0, 1, 100)
    roc_auc_list_all = []
    for i, y_true in enumerate(y_true_list_all):
        y_pred = y_pred_list_all[i]
        fpr_all, tpr_all, thresholds = roc_curve(y_true, y_pred)        
        roc_auc = auc(fpr_all, tpr_all) 
        roc_auc_list_all.append(roc_auc)
        mean_tpr_all += interp(mean_fpr_all, fpr_all, tpr_all)
        mean_tpr_all[0] = 0.0

    mean_tpr_all /= float(cv_times)
    mean_tpr_all[-1] = 1.0
    mean_auc_all = auc(mean_fpr_all, mean_tpr_all)

    print 'roc_auc_list_prec:', roc_auc_list_prec
    roc_auc_list_prec = np.array(roc_auc_list_prec)
    roc_auc_mean_prec = np.mean(roc_auc_list_prec)
    roc_auc_std_prec = np.std(roc_auc_list_prec)
    print 'prec roc auc mean:', roc_auc_mean_prec
    print 'prec roc auc std:', roc_auc_std_prec

    print 'roc_auc_list_power:', roc_auc_list_power
    roc_auc_list_power = np.array(roc_auc_list_power)
    roc_auc_mean_power = np.mean(roc_auc_list_power)
    roc_auc_std_power = np.std(roc_auc_list_power)
    print 'power roc auc mean:', roc_auc_mean_power
    print 'power roc auc std:', roc_auc_std_power

    print 'roc_auc_list_all:', roc_auc_list_all
    roc_auc_list_all = np.array(roc_auc_list_all)
    roc_auc_mean_all = np.mean(roc_auc_list_all)
    roc_auc_std_all = np.std(roc_auc_list_all)
    print 'all roc auc mean:', roc_auc_mean_all
    print 'all roc auc std:', roc_auc_std_all

    plt.figure()
    matplotlib.rcParams.update({'font.size': 16})
    lw = 2
    plt.plot(mean_fpr_prec, mean_tpr_prec, color='b', 
            label='Precision', lw=lw)
    plt.plot(mean_fpr_power, mean_tpr_power, linestyle='--', color='g', 
            label='Power', lw=lw)
    plt.plot(mean_fpr_all, mean_tpr_all, linestyle='-.', color='r', 
            label='No type', lw=lw)
    plt.plot([0, 1], [0, 1], linestyle=':', lw=lw, color='k',
            label='Random')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Mean ROC Curve of Cross Validations')
    plt.legend(loc="lower right")
    #plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #                   ncol=2, mode="expand", borderaxespad=0.)
    #plt.show()
    plt.savefig(fig_name)
    #plt.clf()
    plt.close()

def plot_roc_non_type_seperate(y_true_list_prec, y_pred_list_prec, 
                               y_true_list_power, y_pred_list_power, 
                               y_true_list_all_prec, y_pred_list_all_prec, 
                               y_true_list_all_power, y_pred_list_all_power, 
                               fig_name): 
    '''
    Plot average ROC curve for the cross validation of all grasp types.
    The precision and power grasps roc of the non-type (all) classifier 
    are plotted seperately.
    '''
    cv_times = len(y_true_list_prec)

    mean_tpr_prec = 0.0
    mean_fpr_prec = np.linspace(0, 1, 100)
    roc_auc_list_prec = []
    for i, y_true in enumerate(y_true_list_prec):
        y_pred = y_pred_list_prec[i]
        fpr_prec, tpr_prec, thresholds = roc_curve(y_true, y_pred)        
        roc_auc = auc(fpr_prec, tpr_prec) 
        roc_auc_list_prec.append(roc_auc)
        mean_tpr_prec += interp(mean_fpr_prec, fpr_prec, tpr_prec)
        mean_tpr_prec[0] = 0.0

    mean_tpr_prec /= float(cv_times)
    mean_tpr_prec[-1] = 1.0

    mean_tpr_power = 0.0
    mean_fpr_power = np.linspace(0, 1, 100)
    roc_auc_list_power = []
    for i, y_true in enumerate(y_true_list_power):
        y_pred = y_pred_list_power[i]
        fpr_power, tpr_power, thresholds = roc_curve(y_true, y_pred)        
        roc_auc = auc(fpr_power, tpr_power) 
        roc_auc_list_power.append(roc_auc)
        mean_tpr_power += interp(mean_fpr_power, fpr_power, tpr_power)
        mean_tpr_power[0] = 0.0

    mean_tpr_power /= float(cv_times)
    mean_tpr_power[-1] = 1.0

    mean_tpr_all_prec = 0.0
    mean_fpr_all_prec = np.linspace(0, 1, 100)
    roc_auc_list_all_prec = []
    for i, y_true in enumerate(y_true_list_all_prec):
        y_pred = y_pred_list_all_prec[i]
        fpr_all_prec, tpr_all_prec, thresholds = roc_curve(y_true, y_pred)        
        roc_auc = auc(fpr_all_prec, tpr_all_prec) 
        roc_auc_list_all_prec.append(roc_auc)
        mean_tpr_all_prec += interp(mean_fpr_all_prec, fpr_all_prec, tpr_all_prec)
        mean_tpr_all_prec[0] = 0.0

    mean_tpr_all_prec /= float(cv_times)
    mean_tpr_all_prec[-1] = 1.0

    mean_tpr_all_power = 0.0
    mean_fpr_all_power = np.linspace(0, 1, 100)
    roc_auc_list_all_power = []
    for i, y_true in enumerate(y_true_list_all_power):
        y_pred = y_pred_list_all_power[i]
        fpr_all_power, tpr_all_power, thresholds = roc_curve(y_true, y_pred)        
        roc_auc = auc(fpr_all_power, tpr_all_power) 
        roc_auc_list_all_power.append(roc_auc)
        mean_tpr_all_power += interp(mean_fpr_all_power, fpr_all_power, tpr_all_power)
        mean_tpr_all_power[0] = 0.0

    mean_tpr_all_power /= float(cv_times)
    mean_tpr_all_power[-1] = 1.0

    print 'roc_auc_list_prec:', roc_auc_list_prec

    print 'roc_auc_list_power:', roc_auc_list_power

    print 'roc_auc_list_all_prec:', roc_auc_list_all_prec

    print 'roc_auc_list_all_power:', roc_auc_list_all_power

    sub_fig = False #True
    if sub_fig:
        plt.figure()
        matplotlib.rcParams.update({'font.size': 16})
        lw = 2
        ##plt.subplot(121)
        #plt.plot(mean_fpr_prec, mean_tpr_prec, linestyle='-', color='b', 
        #        label='Precision', lw=lw)
        #plt.plot(mean_fpr_all_prec, mean_tpr_all_prec, linestyle='-.', color='r', 
        #        label='Non-type precision', lw=lw)
        #plt.plot([0, 1], [0, 1], linestyle=':', lw=lw, color='m',
        #        label='Random')   
        #plt.xlim([-0.05, 1.05])   
        #plt.ylim([-0.05, 1.05])
        #plt.xlabel('False Positive Rate')
        #plt.ylabel('True Positive Rate')
        #plt.title('ROC Curve of Precision Grasps')
        #plt.legend(loc="lower right")

        ##plt.subplot(122)
        plt.plot(mean_fpr_power, mean_tpr_power, dashes=[10, 5, 20, 5], color='g', 
                label='Power', lw=lw)
        plt.plot(mean_fpr_all_power, mean_tpr_all_power, linestyle='--', color='k', 
                label='Non-type power', lw=lw)
        plt.plot([0, 1], [0, 1], linestyle=':', lw=lw, color='m',
                label='Random')   
        plt.xlim([-0.05, 1.05])   
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve of Power Grasps')
        plt.legend(loc="lower right")

        plt.savefig(fig_name)
        plt.close()
    else:
        plt.figure()
        matplotlib.rcParams.update({'font.size': 16})
        lw = 2
        plt.plot(mean_fpr_prec, mean_tpr_prec, linestyle='-', color='b', 
                label='Precision', lw=lw)
        plt.plot(mean_fpr_power, mean_tpr_power, dashes=[10, 5, 20, 5], color='g', 
                label='Power', lw=lw)
        plt.plot(mean_fpr_all_prec, mean_tpr_all_prec, linestyle='-.', color='r', 
                label='NT precision', lw=lw)
        plt.plot(mean_fpr_all_power, mean_tpr_all_power, linestyle='--', color='k', 
                label='NT power', lw=lw)
        plt.plot([0, 1], [0, 1], linestyle=':', lw=lw, color='m',
                label='Random')   
                                  
        plt.xlim([-0.05, 1.05])   
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve of Leave-one-out')
        plt.legend(loc="lower right")
        #plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
        #                   ncol=2, mode="expand", borderaxespad=0.)
        #plt.show()
        plt.savefig(fig_name)
        #plt.clf()
        plt.close()

def cv_roc_plot():
    cv_idx = 10 #5
    cross_val_path = '../cross_val/'

    grasp_type = 'prec'
    y_true_list_prec = []
    y_pred_list_prec = []
    for cv_i in xrange(cv_idx):
        gt_labels_file_name = cross_val_path + grasp_type + '/gt_labels_cv_' + str(cv_i) + '.txt'
        y_true = np.loadtxt(gt_labels_file_name)
        y_true_list_prec.append(y_true)
        pred_score_file_name = cross_val_path + grasp_type + '/pred_score_cv_' + str(cv_i) + '.txt'
        y_pred = np.loadtxt(pred_score_file_name)
        y_pred_list_prec.append(y_pred)

    grasp_type = 'power'
    y_true_list_power = []
    y_pred_list_power = []
    for cv_i in xrange(cv_idx):
        gt_labels_file_name = cross_val_path + grasp_type + '/gt_labels_cv_' + str(cv_i) + '.txt'
        y_true = np.loadtxt(gt_labels_file_name)
        y_true_list_power.append(y_true)
        pred_score_file_name = cross_val_path + grasp_type + '/pred_score_cv_' + str(cv_i) + '.txt'
        y_pred = np.loadtxt(pred_score_file_name)
        y_pred_list_power.append(y_pred)

    grasp_type = 'all'
    y_true_list_all = []
    y_pred_list_all = []
    for cv_i in xrange(cv_idx):
        gt_labels_file_name = cross_val_path + grasp_type + '/gt_labels_cv_' + str(cv_i) + '.txt'
        y_true = np.loadtxt(gt_labels_file_name)
        y_true_list_all.append(y_true)
        pred_score_file_name = cross_val_path + grasp_type + '/pred_score_cv_' + str(cv_i) + '.txt'
        y_pred = np.loadtxt(pred_score_file_name)
        y_pred_list_all.append(y_pred)

    fig_name = cross_val_path + '/avg_roc_types.png'
    plot_average_roc_curve_gtypes(y_true_list_prec, y_pred_list_prec, 
                                y_true_list_power, y_pred_list_power, 
                                y_true_list_all, y_pred_list_all, fig_name) 
    threshold = 0.5
    print '######################################################'
    print 'Precision:'
    compute_metrics_one_thresh_cv(y_true_list_prec, y_pred_list_prec, threshold)
    print '######################################################'
    print 'Power:'
    compute_metrics_one_thresh_cv(y_true_list_power, y_pred_list_power, threshold)
    print '######################################################'
    print 'All:'
    compute_metrics_one_thresh_cv(y_true_list_all, y_pred_list_all, threshold)

def loo_roc_plot():
    leave_one_out_path = '../leave_one_out/'

    grasp_type = 'prec'
    y_true_list_prec = []
    y_pred_list_prec = []
    gt_labels_file_name = leave_one_out_path + grasp_type + '/gt_labels_loo.txt'
    y_true = np.loadtxt(gt_labels_file_name)
    y_true_list_prec.append(y_true)
    pred_score_file_name = leave_one_out_path + grasp_type + '/pred_score_loo.txt'
    y_pred = np.loadtxt(pred_score_file_name)
    y_pred_list_prec.append(y_pred)

    grasp_type = 'power'
    y_true_list_power = []
    y_pred_list_power = []
    gt_labels_file_name = leave_one_out_path + grasp_type + '/gt_labels_loo.txt'
    y_true = np.loadtxt(gt_labels_file_name)
    y_true_list_power.append(y_true)
    pred_score_file_name = leave_one_out_path + grasp_type + '/pred_score_loo.txt'
    y_pred = np.loadtxt(pred_score_file_name)
    y_pred_list_power.append(y_pred)

    grasp_type = 'all'

    y_true_list_all_prec = []
    y_pred_list_all_prec = []
    gt_labels_file_name = leave_one_out_path + grasp_type + '/gt_labels_loo_prec.txt'
    y_true = np.loadtxt(gt_labels_file_name)
    y_true_list_all_prec.append(y_true)
    pred_score_file_name = leave_one_out_path + grasp_type + '/pred_score_loo_prec.txt'
    y_pred = np.loadtxt(pred_score_file_name)
    y_pred_list_all_prec.append(y_pred)

    y_true_list_all_power = []
    y_pred_list_all_power = []
    gt_labels_file_name = leave_one_out_path + grasp_type + '/gt_labels_loo_power.txt'
    y_true = np.loadtxt(gt_labels_file_name)
    y_true_list_all_power.append(y_true)
    pred_score_file_name = leave_one_out_path + grasp_type + '/pred_score_loo_power.txt'
    y_pred = np.loadtxt(pred_score_file_name)
    y_pred_list_all_power.append(y_pred)

    fig_name = leave_one_out_path + '/roc_all_sep.png'
    plot_roc_non_type_seperate(y_true_list_prec, y_pred_list_prec, 
                               y_true_list_power, y_pred_list_power, 
                               y_true_list_all_prec, y_pred_list_all_prec,
                               y_true_list_all_power, y_pred_list_all_power,
                               fig_name) 
    threshold = 0.5
    print '######################################################'
    print 'Precision:'
    #compute_metrics_one_thresh_cv(y_true_list_prec, y_pred_list_prec, threshold)
    print 'accuracy, fscore, rand_f_score:'
    print compute_accuracy_f1(y_true_list_prec[0], y_pred_list_prec[0], threshold)
    print '######################################################'
    print 'Power:'
    #compute_metrics_one_thresh_cv(y_true_list_power, y_pred_list_power, threshold)
    print 'accuracy, fscore, rand_f_score:'
    print compute_accuracy_f1(y_true_list_power[0], y_pred_list_power[0], threshold)
    print '######################################################'
    print 'All precision:'
    #compute_metrics_one_thresh_cv(y_true_list_all, y_pred_list_all, threshold)
    print 'accuracy, fscore, rand_f_score:'
    print compute_accuracy_f1(y_true_list_all_prec[0], y_pred_list_all_prec[0], threshold)
    print 'All power:'
    #compute_metrics_one_thresh_cv(y_true_list_all, y_pred_list_all, threshold)
    print 'accuracy, fscore, rand_f_score:'
    print compute_accuracy_f1(y_true_list_all_power[0], y_pred_list_all_power[0], threshold)

if __name__ == '__main__':
    #cv_roc_plot()
    loo_roc_plot()

