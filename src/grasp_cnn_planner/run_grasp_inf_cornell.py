import h5py
import cv2
import numpy as np
import time
import os
import gen_rgbd_patches as gp
from grasp_rgbd_inf import GraspRgbdInf 

def get_initial_configs(i, j):
    fg_path = '/media/kai/cornell_grasp_data/foregrounds/' + str(i+1).zfill(2) 
    ellipse_file_path = fg_path + '/ellipse_' + str(i+1).zfill(2) + str(j).zfill(2) + '.txt' 
    ellipse_file = open(ellipse_file_path, 'r')
    ellipse_lines = ellipse_file.readlines()
    ellipse_lines_split = map(str.split, ellipse_lines) 
    #ellipse_lines_split = np.array(map(np.array, ellipse_lines_split)) 
    #ellipse_lines_split = np.array(ellipse_lines_split)
    #print ellipse_lines_split
    ellipse = np.array(ellipse_lines_split[0]).astype(float) 
    rect = np.array(ellipse_lines_split[1:]).astype(float) 
    #print ellipse
    #print rect
    center_x = ellipse[0]
    center_y = ellipse[1]
    width = ellipse[2]
    height = ellipse[3]
    min_width = 10
    if width < min_width:
        width = min_width
    max_width = 60
    if width > max_width:
        width = max_width
    two_pi = 2 * np.pi
    angle = np.radians(ellipse[4])
    angle_rev = (angle + np.pi) % two_pi
    angle_perp = (angle + np.pi*0.5) % two_pi
    angle_perp_rev = (angle_perp + np.pi) % two_pi

    img_rows = 480
    img_cols = 640
    configs = []
    configs.append([center_x, center_y, angle_perp, height])
    configs.append([center_x, center_y, angle_perp_rev, height])
    configs.append([center_x, center_y, angle, width])
    configs.append([center_x, center_y, angle_rev, width])
    #for i in xrange(4):
    #    [mid_pnt_x, mid_pnt_y]= 0.5*(rect[i] + rect[(i+1)%4]) 
    #    if (mid_pnt_x < 0 or mid_pnt_x > img_cols
    #            or mid_pnt_y < 0 or mid_pnt_y > img_rows):
    #        continue
    #    configs.append([mid_pnt_x, mid_pnt_y, angle, width])
    #    configs.append([mid_pnt_x, mid_pnt_y, angle_rev, width])
    #    configs.append([mid_pnt_x, mid_pnt_y, angle_perp, width])
    #    configs.append([mid_pnt_x, mid_pnt_y, angle_perp_rev, width])
    
    #rand_angle = np.random.uniform(.0, 2*np.pi)
    #configs.append([center_x, center_y, rand_angle, width])

    ellipse_file.close()
    return np.array(configs)

# Check if a point is inside a rectangle represented by a list of four vertices.
# The order of vertices need to be counter clockwise.
# The z axis points down for the opencv image x, y coordinate. 
# For example, np.cross(np.array([1, 0, 0]), np.array([1, -1, 0])) = [0, 0, -1]
def is_pnt_inside_rect(point, rect):
    vec_ab_2d = rect[1] - rect[0] 
    vec_bc_2d = rect[2] - rect[1]
    vec_cd_2d = rect[3] - rect[2]
    vec_da_2d = rect[0] - rect[3]

    vec_ab_3d = np.append(vec_ab_2d, .0) 
    vec_bc_3d = np.append(vec_bc_2d, .0)  
    vec_cd_3d = np.append(vec_cd_2d, .0) 
    vec_da_3d = np.append(vec_da_2d, .0) 

    vec_ap_3d = np.append(point - rect[0], .0)
    vec_bp_3d = np.append(point - rect[1], .0)
    vec_cp_3d = np.append(point - rect[2], .0)
    vec_dp_3d = np.append(point - rect[3], .0)

    crs_ab_ap = np.cross(vec_ab_3d, vec_ap_3d) 
    crs_bc_bp = np.cross(vec_bc_3d, vec_bp_3d) 
    crs_cd_cp = np.cross(vec_cd_3d, vec_cp_3d) 
    crs_da_dp = np.cross(vec_da_3d, vec_dp_3d) 
    
    return np.max([crs_ab_ap, crs_bc_bp, crs_cd_cp, crs_da_dp]) <= .0

def test_pnt_inside_rect():
    rect = np.array([[.0, .0], [.0, 1.], [1., 1.], [1., .0]])
    point1 = np.array([.5, .5])
    point2 = np.array([1., 1.])
    point3 = np.array([2., 2.])
    point4 = np.array([1.1, .0])
    print is_pnt_inside_rect(point1, rect)
    print is_pnt_inside_rect(point2, rect)
    print is_pnt_inside_rect(point3, rect)
    print is_pnt_inside_rect(point4, rect)


# Get the detection accuracy: intersection over union
# between the detection and all ground truth positive grasps.
def get_inter_over_union(rect_gt, rect_detect):
    min_x = np.min(rect_gt[:, 0]).astype(int) - 1
    max_x = np.max(rect_gt[:, 0]).astype(int) + 1
    min_y = np.min(rect_gt[:, 1]).astype(int) - 1
    max_y = np.max(rect_gt[:, 1]).astype(int) + 1

    inter_pix_num = 0
    for x in range(min_x, max_x):
        for y in range(min_y, max_y):
            point = np.array([x, y])
            if is_pnt_inside_rect(point, rect_gt) and is_pnt_inside_rect(point, rect_detect):
                inter_pix_num += 1
    rect_gt_area = np.linalg.norm(rect_gt[1] - rect_gt[0]) * np.linalg.norm(rect_gt[2] - rect_gt[1])
    rect_detect_area = np.linalg.norm(rect_detect[1] - rect_detect[0]) * np.linalg.norm(rect_detect[2] - rect_detect[1])
    #print inter_pix_num
    #print rect_gt_area
    #print rect_detect_area
    inter_over_union = inter_pix_num / (rect_gt_area + rect_detect_area - inter_pix_num)
    return inter_over_union

def test_inter_over_union():
    rect1 = 10*np.array([[.0, .0], [.0, 1.], [1., 1.], [1., .0]])
    rect2 = 10*np.array([[1., .0], [.0, 1.], [.5, 1.5], [1.5, .5]])
    rect3 = 10*np.ones(2) + rect1
    print get_inter_over_union(rect1, rect1*0.5)
    print get_inter_over_union(rect1, rect2)
    print get_inter_over_union(rect1, rect3)

def is_detection_correct_one_rect(rect_gt, rect_detect, config_detect):
    angle_detect = config_detect[2]
    #print rect_gt
    config_gt = gp.get_config_from_rect(rect_gt)
    angle_gt = config_gt[2]
    #To be a correct detetction, 1) the grasp angle is within 30 degrees of
    #the ground truth; 2) the intersection over union is greater than 25%.
    #The angle needs to be moded by pi since two fingers are symmetric. 
    #angle_diff = np.abs(angle_detect % np.pi - angle_gt % np.pi)
    two_pi = 2 * np.pi
    #As long as one finger satisfies the angle criteria.
    angle_diff_1 = angle_detect % two_pi - angle_gt % two_pi
    angle_diff_1 = np.degrees(angle_diff_1) % 360
    angle_diff_2 = (angle_detect + np.pi) % two_pi - angle_gt % two_pi
    angle_diff_2 = np.degrees(angle_diff_2) % 360
    #angle_crt = angle_diff_1 < 30. or angle_diff_1 > 330. or \
    #        angle_diff_2 < 30. or angle_diff_2 > 330. 
    #print 'angle_diff:', angle_diff_1, angle_diff_2
    angle_crt = False
    if angle_diff_1 <= 30. or angle_diff_1 >= 330.:
        angle_crt = True
    elif angle_diff_2 <= 30. or angle_diff_2 >= 330.:
        angle_crt = True
    
    inter_over_union = get_inter_over_union(rect_gt, rect_detect)
    iou_crt = inter_over_union >= 0.25
    correct = angle_crt and iou_crt
    return correct, angle_diff_1, angle_diff_2, inter_over_union
            #angle_crt, iou_crt
    
def is_detection_correct_all_rects(rects_gt, config_detect, 
        rgbd=None, rgbd_id=None, cfg_id=None, path=None, init=True, 
        suc_prob=None):
    correct_all_gt = False
    rect_detect = gp.get_rect_from_config(config_detect, height=80)
    gt_rects_num = rects_gt.shape[0]/4
    for r in xrange(gt_rects_num):
        rect = rects_gt[r*4:(r+1)*4]
        #rect_area = np.linalg.norm(rect[1] - rect[0]) * np.linalg.norm(rect[2] - rect[1])
        ##Make the dection rectangle have the same area with the ground truth rectangle.
        #rect_detect = gp.get_rect_from_config(config_detect, height=rect_area/config_detect[3])
        #print rect
        correct, angle_diff_1, angle_diff_2, inter_over_union = \
        is_detection_correct_one_rect(rect, rect_detect, config_detect)
        plot_info = str(correct) + ' ' + str(int(angle_diff_1)) + \
                ' ' + str(int(angle_diff_2)) + ' ' + \
                str(int(100*inter_over_union)) + '% '
        if suc_prob != None:
            plot_info += str(int(100*suc_prob)) + '%'
        if rgbd != None:
            rgbd_copy = rgbd.copy()
            gp.plot_rect_rgbd(rgbd_copy, rect, plot_info)
            gp.plot_rect_rgbd(rgbd_copy, rect_detect)
            plot_path = path + str(rgbd_id) + '_cfg_' + str(cfg_id) + '_gt_' + str(r)
            if init:
                plot_path += '_init.png'
            else:
                plot_path += '_opt.png'
            cv2.imwrite(plot_path, rgbd_copy[:, :, :3])

            if r == 0:
                plot_path_one_gt = path + 'one_gt/' + str(rgbd_id) + '_cfg_' + \
                        str(cfg_id) + '_gt_' + str(r)
                if init:
                    plot_path_one_gt += '_init.png'
                else:
                    plot_path_one_gt += '_opt.png'
                cv2.imwrite(plot_path_one_gt, rgbd_copy[:, :, :3])

        if correct and not correct_all_gt:
            correct_all_gt = True

        if suc_prob == 1. and r == 0:
            #print 'suc_prob:', suc_prob
            plot_path_suc = path + 'suc_prob_is_1/' + str(rgbd_id) + '_cfg_' + \
                    str(cfg_id) + '_gt_' + str(r)
            if init:
                plot_path_suc += '_init.png'
            else:
                plot_path_suc += '_opt.png'
            #print plot_path_suc
            cv2.imwrite(plot_path_suc, rgbd_copy[:, :, :3])

    return correct_all_gt

def plot_gt_rects(rects_gt, rgbd, rgbd_id, path):
    gt_rects_num = rects_gt.shape[0]/4
    for r in xrange(gt_rects_num):
        rect = rects_gt[r*4:(r+1)*4]
        gp.plot_rect_rgbd(rgbd, rect)
    plot_path = path + str(rgbd_id) + '_all_gt.png'
    cv2.imwrite(plot_path, rgbd[:, :, :3])


def main():
    grasp_inf = GraspRgbdInf()
    pcd_nums = [100] * 10
    pcd_nums[8] = 50
    pcd_nums[9] = 35
    data_path = '/media/kai/cornell_grasp_data/'
    total_pcd_num = np.sum(pcd_nums)
    test_pcd_num = int(0.9*total_pcd_num)
    rgbd_file = h5py.File(data_path + 'h5_data/rgbd.h5', 'r')
    rgbd_data = rgbd_file['RGBD'] 

    start_time = time.time()
    prev_num = 0
    cur_grasp_num = 0
    init_eval_list = []
    init_inf_eval_list = []
    init_suc_prob_list = []
    init_inf_suc_prob_list = []
    det_opt_eval_list = []
    det_init_eval_list = []
    det_opt_suc_prob_list = []
    det_init_suc_prob_list = []

    #correct_pred_num = 0
    #grasp_labels = []
    #pred_labels = []
    for i, n in enumerate(pcd_nums):
        if i >= 1:
            prev_num += pcd_nums[i-1]
        #if i != 7:
        #    continue
        #if i > 0 and i < 9:
        #    continue
        for j in xrange(n):
            rgbd_id = prev_num + j
            if rgbd_id < test_pcd_num:
                continue
            #if rgbd_id >= test_pcd_num:
            #    break
            #if rgbd_id > int(total_pcd_num*0.1):
            #    break
            #if j != 55:
            #    continue
            print 'rgbd_id:', rgbd_id
            rgbd = rgbd_data[rgbd_id]
            print i, j

            cpos_file = data_path + str(i+1).zfill(2) + '/pcd' + \
                     str(i+1).zfill(2) + str(j).zfill(2) + 'cpos.txt' 
            #rects_gt = read_gt_rects(cpos_file)
            rects_gt = gp.read_rect(cpos_file)
            
            #plot_gt_rects(rects_gt, rgbd, rgbd_id, '/media/kai/tmp/gt/')
            #return

            init_configs = get_initial_configs(i, j)
            #print init_configs
            max_suc_prob = -1
            suc_probs = []
            init_suc_probs = []
            config_inits = []
            config_init_rects = []
            init_evals = []
            config_opts = []
            config_opt_rects = []
            opt_evals = []
            for r, config in enumerate(init_configs):
                print 'config:', r
                config_opt, suc_prob, suc_prob_init = \
                        grasp_inf.grad_descent_inf(rgbd.copy(), config.copy(), rgbd_id, r) 
                config_rect = gp.get_rect_from_config(config)
                config_opt_rect = gp.get_rect_from_config(config_opt)
                print config, config_opt
                #print config_rect, config_opt_rect
                #config_test = gp.get_config_from_rect(config_rect)
                #config_rect_test = gp.get_rect_from_config(config_test)
                #print "TEST:", config, config_rect, config_test, config_rect_test

                gp.extract_rgbd_patches(rgbd.copy(), config_rect.copy(), rgbd_id=rgbd_id, 
                        grasp_id=r, path='/media/kai/tmp/images_init/', save=True) 
                gp.extract_rgbd_patches(rgbd.copy(), config_opt_rect.copy(), rgbd_id=rgbd_id, 
                        grasp_id=r, path='/media/kai/tmp/images_init_opt/', save=True)
                gp.extract_rgbd_patches_cfg(rgbd.copy(), config.copy(), rgbd_id=rgbd_id, 
                        grasp_id=r, path='/media/kai/tmp/images_init_cfg/', save=True)


                init_eval = is_detection_correct_all_rects(rects_gt.copy(), config.copy(), 
                        rgbd.copy(), rgbd_id, r, '/media/kai/tmp/init_inf_eval/', init=True, 
                        suc_prob=suc_prob_init)
                init_eval_list.append(init_eval) 
                init_inf_eval = is_detection_correct_all_rects(rects_gt.copy(), config_opt.copy(), 
                        rgbd.copy(), rgbd_id, r, '/media/kai/tmp/init_inf_eval/', init=False, 
                        suc_prob=suc_prob)
                init_inf_eval_list.append(init_inf_eval) 
                init_suc_prob_list.append(suc_prob_init)
                init_inf_suc_prob_list.append(suc_prob)

                config_inits.append(config)
                config_init_rects.append(config_rect)
                init_evals.append(init_eval)
                suc_probs.append(suc_prob)
                init_suc_probs.append(suc_prob_init)
                config_opts.append(config_opt)
                config_opt_rects.append(config_opt_rect)
                opt_evals.append(init_inf_eval)
                if suc_prob > max_suc_prob:
                    max_suc_prob = suc_prob
            
            best_suc_probs = []
            best_config_inits = []
            best_config_init_rects = []
            best_init_evals = []
            best_config_opts = []
            best_config_opt_rects = []
            best_opt_evals = []
            best_init_ids = []
            for k, suc_prob in enumerate(suc_probs):
                if suc_prob == max_suc_prob:
                    best_init_ids.append(k)
                    best_suc_probs.append(suc_probs[k])
                    best_config_inits.append(config_inits[k])
                    best_config_init_rects.append(config_init_rects[k])
                    best_init_evals.append(init_evals[k])
                    best_config_opts.append(config_opts[k])
                    best_config_opt_rects.append(config_opt_rects[k])
                    best_opt_evals.append(opt_evals[k])

                    det_init_eval_list.append(init_evals[k])
                    det_opt_eval_list.append(opt_evals[k])
                    det_init_suc_prob_list.append(init_suc_probs[k])
                    det_opt_suc_prob_list.append(suc_probs[k])
            
            detection_log_path = '/media/kai/tmp/logs_opt/' + str(rgbd_id) + '_det.txt' 
            detection_log_file = open(detection_log_path, 'w')
            detection_log_file.writelines('image_id: \n' + str(i) + ' ' + str(j) + '\n')
            detection_log_file.writelines('best_init_ids: \n' + str(best_init_ids) + '\n') 
            detection_log_file.writelines('best_suc_probs: \n' + str(best_suc_probs) + '\n')
            detection_log_file.writelines('suc_probs: \n' + str(suc_probs) + '\n')
            detection_log_file.writelines('init_suc_probs: \n' + str(init_suc_probs) + '\n')
            detection_log_file.writelines('best_config_inits: \n' + str(best_config_inits) + '\n')
            detection_log_file.writelines('best_config_init_rects: \n' + str(best_config_init_rects) + '\n')
            detection_log_file.writelines('best_init_evals: \n' + str(best_init_evals) + '\n')
            detection_log_file.writelines('best_config_opts: \n' + str(best_config_opts) + '\n')
            detection_log_file.writelines('best_config_opt_rects: \n' + str(best_config_opt_rects) + '\n')
            detection_log_file.writelines('best_opt_evals: \n' + str(best_opt_evals) + '\n')

            detection_log_file.writelines('config_inits: \n' + str(config_inits) + '\n')
            detection_log_file.writelines('config_init_rects: \n' + str(config_init_rects) + '\n')
            detection_log_file.writelines('config_opts: \n' + str(config_opts) + '\n')
            detection_log_file.writelines('config_opt_rects: \n' + str(config_opt_rects) + '\n')

            detection_log_file.close()
            elapsed_time = time.time() - start_time
            print 'elapsed_time: ', elapsed_time
            print '####################################'
    
    #print det_correct_list
    #print init_correct_list
    #print 'Total images:', len(det_correct_list)
    #print 'Detection accuracy:', np.mean(det_correct_list)
    #print 'Total inits:', len(init_correct_list)
    #print 'Init accuary:', np.mean(init_correct_list)
    det_eval_file = open('/media/kai/tmp/det_eval.txt', 'w')
    det_eval_file.writelines('init_eval accuracy: ' + str(np.mean(init_eval_list)) + '\n')
    det_eval_file.writelines('init_inf_eval accuracy: ' + str(np.mean(init_inf_eval_list)) + '\n')
    det_eval_file.writelines('det_init_eval accuracy: ' + str(np.mean(det_init_eval_list)) + '\n')
    det_eval_file.writelines('det_opt_eval accuracy: ' + str(np.mean(det_opt_eval_list)) + '\n')
    det_eval_file.writelines('init_eval_list length: ' + str(len(init_eval_list)) + '\n')
    det_eval_file.writelines('init_inf_eval_list length: ' + str(len(init_inf_eval_list)) + '\n')
    det_eval_file.writelines('det_init_eval_list length: ' + str(len(det_init_eval_list)) + '\n')
    det_eval_file.writelines('det_opt_eval_list length: ' + str(len(det_opt_eval_list)) + '\n')
    det_eval_file.writelines('init_eval_list: ' + str(init_eval_list) + '\n')
    det_eval_file.writelines('init_inf_eval_list: ' + str(init_inf_eval_list) + '\n')
    det_eval_file.writelines('det_init_eval_list: ' + str(det_init_eval_list) + '\n')
    det_eval_file.writelines('det_opt_eval_list: ' + str(det_opt_eval_list) + '\n')
    det_eval_file.writelines('init_suc_prob_list: ' + str(init_suc_prob_list) + '\n')
    det_eval_file.writelines('init_inf_suc_prob_list: ' + str(init_inf_suc_prob_list) + '\n')
    det_eval_file.writelines('det_init_suc_prob_list: ' + str(det_init_suc_prob_list) + '\n')
    det_eval_file.writelines('det_opt_suc_prob_list: ' + str(det_opt_suc_prob_list) + '\n')
    det_eval_file.writelines('init_suc_prob_list mean: ' + str(np.mean(init_suc_prob_list)) + '\n')
    det_eval_file.writelines('init_inf_suc_prob_list mean: ' + str(np.mean(init_inf_suc_prob_list)) + '\n')
    det_eval_file.writelines('det_init_suc_prob_list mean: ' + str(np.mean(det_init_suc_prob_list)) + '\n')
    det_eval_file.writelines('det_opt_suc_prob_list mean: ' + str(np.mean(det_opt_suc_prob_list)) + '\n')

    det_eval_file.close()
    #print 'grasps number:', cur_grasp_num
    elapsed_time = time.time() - start_time
    print 'total elapsed_time: ', elapsed_time

    rgbd_file.close()

if __name__ == '__main__':
    main()
    #test_pnt_inside_rect()
    #test_inter_over_union()
