import h5py
import cv2
import numpy as np
import time
import os

rgbd_channels = 8

def rotateRgbd(rgbd, angle, rot_center):
    #cv2.getRotationMatrix2D(center, angle, scale)
    #center=(x,y)=(col, row)
    #(x=0, y=0) is the top left corner of image in opencv.
    #angle Rotation angle in degrees. Positive values mean counter-clockwise 
    #rotation (the coordinate origin is assumed to be the top-left corner)
    rot_mat = cv2.getRotationMatrix2D(rot_center, angle, 1.0)
    rgbd_rot = np.copy(rgbd)
    cols = rgbd.shape[1]
    rows = rgbd.shape[0]
    rgbd_rot[:, :, :3] = cv2.warpAffine(rgbd[:, :, :3], rot_mat, (cols, rows), flags=cv2.INTER_LINEAR)
    rgbd_rot[:, :, 3:6] = cv2.warpAffine(rgbd[:, :, 3:6], rot_mat, (cols, rows), flags=cv2.INTER_LINEAR)
    rgbd_rot[:, :, 6:8] = cv2.warpAffine(rgbd[:, :, 6:8], rot_mat, (cols, rows), flags=cv2.INTER_LINEAR)

    return rgbd_rot

def extract_rgbd_patch(rgbd, center, patch_size):
    patch = np.zeros((patch_size, patch_size, rgbd_channels))
    patch_sizes = (patch_size, patch_size)
    #patch[:, :, :3] = cv2.getRectSubPix(rgbd[:, :, :3].astype('uint8'), patch_sizes, center)
    #patch[:, :, 3:6] = cv2.getRectSubPix(rgbd[:, :, 3:6].astype('float32'), patch_sizes, center)
    #patch[:, :, 6:8] = cv2.getRectSubPix(rgbd[:, :, 6:8].astype('float32'), patch_sizes, center)
    for i in xrange(rgbd_channels):
        #cv2.getRectSubPix(image, patchSize, center[, patch[, patchType]])
        #center=(x,y)=(col, row)
        patch[:, :, i] = cv2.getRectSubPix(rgbd[:, :, i].astype('float32'), patch_sizes, center)

    return patch

#Plot grasps without saving
def plot_rect_rgbd(rgbd, rect, plot_info=None):
    rgb = np.copy(rgbd[:, :, :3]).astype('uint8')
    #(B, G, R)
    cv2.line(rgb, tuple(rect[0].astype(int)), tuple(rect[1].astype(int)), (255,0,0), 2)
    cv2.line(rgb, tuple(rect[1].astype(int)), tuple(rect[2].astype(int)), (0,255,0), 2)
    cv2.line(rgb, tuple(rect[2].astype(int)), tuple(rect[3].astype(int)), (0,0,255), 2)
    cv2.line(rgb, tuple(rect[3].astype(int)), tuple(rect[0].astype(int)), (0,0,0), 2)
    
    #cv2.line(rgb, tuple(rect[0].astype(int)), tuple(rect[1].astype(int)), (255,0,0), 2)
    #cv2.line(rgb, tuple(rect[1].astype(int)), tuple(rect[2].astype(int)), (0,0,255), 2)
    #cv2.line(rgb, tuple(rect[2].astype(int)), tuple(rect[3].astype(int)), (255,0,0), 2)
    #cv2.line(rgb, tuple(rect[3].astype(int)), tuple(rect[0].astype(int)), (0,0,255), 2)
    
    palm_loc = tuple(np.mean(rect, 0).astype(int))
    finger_1_loc = tuple(np.mean(rect[0:2, :], 0).astype(int))
    finger_2_loc = tuple(np.mean(rect[2:4, :], 0).astype(int))
    cv2.circle(rgb, palm_loc, 4, (255, 255, 255), -1)
    cv2.circle(rgb, finger_1_loc, 4, (0, 0, 255), -1)
    cv2.circle(rgb, finger_2_loc, 4, (255, 0, 0), -1)
    if plot_info != None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(rgb, plot_info, palm_loc, font, 0.5, (0,0,0), 1)
    else:
        cv2.line(rgb, finger_1_loc, finger_2_loc, (255,0,255), 5)
    
    rgbd[:, :, :3] = rgb


#Plot grasps with saving
def plot_rgbd(rgbd, path, rect=None, line_orig_rgb=False, palm_finger_locs=None):
    rgb = np.copy(rgbd[:, :, :3]).astype('uint8')
    if rect != None:
        #(B, G, R)
        cv2.line(rgb, tuple(rect[0].astype(int)), tuple(rect[1].astype(int)), (255,0,0), 5)
        cv2.line(rgb, tuple(rect[1].astype(int)), tuple(rect[2].astype(int)), (0,255,0), 5)
        cv2.line(rgb, tuple(rect[2].astype(int)), tuple(rect[3].astype(int)), (0,0,255), 5)
        cv2.line(rgb, tuple(rect[3].astype(int)), tuple(rect[0].astype(int)), (0,0,0), 5)
        palm_loc = tuple(np.mean(rect, 0).astype(int))
        finger_1_loc = tuple(np.mean(rect[0:2, :], 0).astype(int))
        finger_2_loc = tuple(np.mean(rect[2:4, :], 0).astype(int))
        cv2.line(rgb, finger_1_loc, finger_2_loc, (255,0,255), 5)
        cv2.circle(rgb, palm_loc, 4, (255, 255, 255), -1)
        cv2.circle(rgb, finger_1_loc, 4, (0, 0, 255), -1)
        cv2.circle(rgb, finger_2_loc, 4, (255, 0, 0), -1)

    if palm_finger_locs != None:
        #(B, G, R)
        palm_loc = tuple(palm_finger_locs[0, :].astype(int))
        finger_1_loc = tuple(palm_finger_locs[1, :].astype(int))
        finger_2_loc = tuple(palm_finger_locs[2, :].astype(int))
        cv2.line(rgb, finger_1_loc, finger_2_loc, (100,0,100), 5)
        cv2.circle(rgb, palm_loc, 6, (255, 255, 255), -1)
        cv2.circle(rgb, finger_1_loc, 6, (255, 0, 0), -1)
        cv2.circle(rgb, finger_2_loc, 6, (0, 0, 255), -1)

    cv2.imwrite(path + '/rgb.png', rgb)

    if line_orig_rgb:
        rgbd[:, :, :3] = rgb

    depth = np.copy(rgbd[:, :, 3])
    depth = 255. * (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    cv2.imwrite(path + '/depth.png', depth)
    
    normal = np.copy(rgbd[:, :, 4:7])
    for i in xrange(3):
        normal[:, :, i] = 255. * (normal[:, :, i] - np.min(normal[:, :, i])) / \
                (np.max(normal[:, :, i]) - np.min(normal[:, :, i]))
    normal = normal.astype('uint8')
    cv2.imwrite(path + '/normal.png', normal)
    
    curv = np.copy(rgbd[:, :, 7])
    curv = 255. * (curv - np.min(curv)) / (np.max(curv) - np.min(curv))
    cv2.imwrite(path + '/curv.png', curv)

def extract_rgbd_patches(rgbd, rect, patch_size=200, finer_level=2,
        rgbd_id=None, grasp_id=None, path='../images/', save=False): 

    dxy = rect[1] - rect[0]
    #Y axis of image points direction. Otherwise, need to rotate -angle.
    #Positive values mean counter-clockwise rotation for cv2.getRotationMatrix2D.
    #np.math.atan2 is in range [-pi, pi]
    angle = np.math.atan2(dxy[1], dxy[0])
    angle = np.degrees(angle)
    #angle %= 360
        
    #path = '../images/' + str(rgbd_id)
    path += str(rgbd_id) + '_' + str(grasp_id)
    if mirror:
        path += '_mirror'
    if save:
        if not os.path.exists(path):
            os.makedirs(path)
        
        rgbd_path = path + '/rgbd/'
        if not os.path.exists(rgbd_path):
            os.makedirs(rgbd_path)
        plot_rgbd(rgbd, rgbd_path, rect, line_orig_rgb=True)

    palm_loc = tuple(np.mean(rect, 0))
    rgbd_rot_palm = rotateRgbd(rgbd, angle, palm_loc)
    palm_patch = extract_rgbd_patch(rgbd_rot_palm, palm_loc, patch_size)  
    finger_1_loc = tuple(np.mean(rect[0:2, :], 0))
    rgbd_rot_f_1 = rotateRgbd(rgbd, angle, finger_1_loc)
    finger_1_patch = extract_rgbd_patch(rgbd_rot_f_1, finger_1_loc, patch_size/finer_level)
    #finger_1_patch = cv2.resize(finger_1_patch, (patch_size, patch_size))
    finger_2_loc = tuple(np.mean(rect[2:4, :], 0))
    rgbd_rot_f_2 = rotateRgbd(rgbd, angle, finger_2_loc)
    finger_2_patch = extract_rgbd_patch(rgbd_rot_f_2, finger_2_loc, patch_size/finer_level)
    #finger_2_patch = cv2.resize(finger_2_patch, (patch_size, patch_size))

    if save:
        palm_rgbd_rot_path = path + '/rgbd_rot_palm/'
        if not os.path.exists(palm_rgbd_rot_path):
            os.makedirs(palm_rgbd_rot_path)
        plot_rgbd(rgbd_rot_palm, palm_rgbd_rot_path)

        f_1_rgbd_rot_path = path + '/rgbd_rot_f_1/'
        if not os.path.exists(f_1_rgbd_rot_path):
            os.makedirs(f_1_rgbd_rot_path)
        plot_rgbd(rgbd_rot_f_1, f_1_rgbd_rot_path)
 
        f_2_rgbd_rot_path = path + '/rgbd_rot_f_2/'
        if not os.path.exists(f_2_rgbd_rot_path):
            os.makedirs(f_2_rgbd_rot_path)
        plot_rgbd(rgbd_rot_f_2, f_2_rgbd_rot_path)
 
        palm_path = path + '/palm/'
        if not os.path.exists(palm_path):
            os.makedirs(palm_path)
        plot_rgbd(palm_patch, palm_path)

        finger_1_path = path + '/finger_1/'
        if not os.path.exists(finger_1_path):
            os.makedirs(finger_1_path)
        plot_rgbd(finger_1_patch, finger_1_path)

        finger_2_path = path + '/finger_2/'
        if not os.path.exists(finger_2_path):
            os.makedirs(finger_2_path)
        plot_rgbd(finger_2_patch, finger_2_path)

    return palm_patch, finger_1_patch, finger_2_patch

def get_config_from_rect(rect):
    #Compute the configuration parameters [x, y,, theta, width] from the rectangle
    #Since this function will extract features around 1st and 3rd edges,
    #need to make the 2nd edge the 1st. 
    #rect_copy = rect.copy()
    #rect[1] = rect_copy[0]
    #rect[2] = rect_copy[1]
    #rect[3] = rect_copy[2]
    #rect[0] = rect_copy[3]
   
    #This works for both couter colockwise and clockwise rectangle. 
    vec_f1_to_f2 = rect[2] - rect[1]
    angle_f1_to_f2 = np.math.atan2(vec_f1_to_f2[1], vec_f1_to_f2[0])
    angle_f1_to_f2 %= 2 * np.pi
    width = np.linalg.norm(vec_f1_to_f2) 
    palm_loc = np.mean(rect, 0)
   
    return np.array([palm_loc[0], palm_loc[1], angle_f1_to_f2, width])

def get_rect_from_config(config, height=80):
    palm_loc = np.array(config[:2])
    #angle_palm_to_f1 = -config[2]
    angle_palm_to_f1 = (config[2] + np.pi) % (2 * np.pi)
    width = config[3]
    palm_to_f1_vec = (0.5 * width) * np.array([np.cos(angle_palm_to_f1), np.sin(angle_palm_to_f1)])
    #Rotate palm_to_f1_bec 90 degrees to get p_f1_rot_90_vec
    p_f1_rot_90_vec = (0.5 * height) * np.array([-np.sin(angle_palm_to_f1), np.cos(angle_palm_to_f1)])
    rect = palm_loc + np.array([palm_to_f1_vec+p_f1_rot_90_vec, palm_to_f1_vec-p_f1_rot_90_vec, 
        -palm_to_f1_vec-p_f1_rot_90_vec, -palm_to_f1_vec+p_f1_rot_90_vec])
    return rect


#Extract rgbd patches from the configuration parameter.
def extract_rgbd_patches_cfg(rgbd, config, patch_size=200, finer_level=2,
        rgbd_id=None, grasp_id=None, path='../images/', save=False):
    #print 'extract patches using config parameter from rgbd_id:', rgbd_id
    palm_loc = tuple(config[:2])
    angle_f1_to_f2 = config[2]
    width = config[3]
    palm_to_f2_vec = (0.5 * width) * np.array([np.cos(angle_f1_to_f2), np.sin(angle_f1_to_f2)])
    finger_1_loc = tuple(palm_loc - palm_to_f2_vec)
    finger_2_loc = tuple(palm_loc + palm_to_f2_vec)

    #angle between the 1st edge (ab) of the rectangle abcd and the x axis. 
    angle_ab = np.degrees(angle_f1_to_f2 + 0.5 * np.pi)

    #path = '../images/' + str(rgbd_id)
    path += str(rgbd_id) + '_' + str(grasp_id)
    if save:
        if not os.path.exists(path):
            os.makedirs(path)
        
        rgbd_path = path + '/rgbd/'
        if not os.path.exists(rgbd_path):
            os.makedirs(rgbd_path)
        plot_rgbd(rgbd, rgbd_path, rect=None, line_orig_rgb=True, 
                palm_finger_locs=np.array([palm_loc, finger_1_loc, finger_2_loc]))

    rgbd_rot_palm = rotateRgbd(rgbd, angle_ab, palm_loc)
    palm_patch = extract_rgbd_patch(rgbd_rot_palm, palm_loc, patch_size)  
    rgbd_rot_f_1 = rotateRgbd(rgbd, angle_ab, finger_1_loc)
    finger_1_patch = extract_rgbd_patch(rgbd_rot_f_1, finger_1_loc, patch_size/finer_level)
    rgbd_rot_f_2 = rotateRgbd(rgbd, angle_ab, finger_2_loc)
    finger_2_patch = extract_rgbd_patch(rgbd_rot_f_2, finger_2_loc, patch_size/finer_level)

    if save:
        palm_rgbd_rot_path = path + '/rgbd_rot_palm/'
        if not os.path.exists(palm_rgbd_rot_path):
            os.makedirs(palm_rgbd_rot_path)
        plot_rgbd(rgbd_rot_palm, palm_rgbd_rot_path)

        f_1_rgbd_rot_path = path + '/rgbd_rot_f_1/'
        if not os.path.exists(f_1_rgbd_rot_path):
            os.makedirs(f_1_rgbd_rot_path)
        plot_rgbd(rgbd_rot_f_1, f_1_rgbd_rot_path)
 
        f_2_rgbd_rot_path = path + '/rgbd_rot_f_2/'
        if not os.path.exists(f_2_rgbd_rot_path):
            os.makedirs(f_2_rgbd_rot_path)
        plot_rgbd(rgbd_rot_f_2, f_2_rgbd_rot_path)
 
        palm_path = path + '/palm/'
        if not os.path.exists(palm_path):
            os.makedirs(palm_path)
        plot_rgbd(palm_patch, palm_path)

        finger_1_path = path + '/finger_1/'
        if not os.path.exists(finger_1_path):
            os.makedirs(finger_1_path)
        plot_rgbd(finger_1_patch, finger_1_path)

        finger_2_path = path + '/finger_2/'
        if not os.path.exists(finger_2_path):
            os.makedirs(finger_2_path)
        plot_rgbd(finger_2_patch, finger_2_path)

    return palm_patch, finger_1_patch, finger_2_patch

def read_rect(path):
    rects_file = open(path, 'r')
    rects_lines = rects_file.readlines()
    rects_lines_split = np.array(map(str.split, rects_lines))
    rects = np.array(map(lambda x: map(float, x), rects_lines_split))
    #print 'rects:', rects
    rects_file.close()
    rects_num = rects.shape[0]/4
    rects_new = []
    for r in xrange(rects_num):
        rect = rects[r*4:(r+1)*4]
        #check if there is nan value in the rectangle 
        if np.isnan(rect).any():
            continue

        #Acoording to the Cornell data readme, the 1st edge of the rectangle
        #represents the gripper plate orientation.
        #Since this function will extract features around 1st and 3rd edges,
        #need to make the 2nd edge the 1st. 
        rect_copy = rect.copy()
        rect[1] = rect_copy[0]
        rect[2] = rect_copy[1]
        rect[3] = rect_copy[2]
        rect[0] = rect_copy[3]

        dxy = rect[1] - rect[0]
        #Y axis of image points direction. Otherwise, need to rotate -angle.
        #Positive values mean counter-clockwise rotation for cv2.getRotationMatrix2D.
        #np.math.atan2 is in range [-pi, pi]
        angle = np.math.atan2(dxy[1], dxy[0])
        angle = np.degrees(angle)
        #angle %= 360
 
        #check and organize clockwise rectangle to counter clockwise
        #Rectagle is abcd. 
        vec_bc = rect[2] - rect[1]
        angle_bc = np.degrees(np.math.atan2(vec_bc[1], vec_bc[0]))
        #angle_bc %= 360
        angle_diff = -angle_bc + angle
        #Convert to [0, 360]
        if angle_diff < 0.:
            angle_diff += 360.
        #Clockwise: 270, counter: 90.
        if angle_diff > 180.:
            rect_copy = rect.copy()
            #print angle_diff, 'clockwise!'
            #print 'before changing:', rect
            rect[1] = rect_copy[0]
            rect[0] = rect_copy[1]
            rect[3] = rect_copy[2]
            rect[2] = rect_copy[3]
        rects_new = rects_new + rect.tolist()
 
    return np.array(rects_new)

def main():    
    pcd_nums = [100] * 10
    pcd_nums[8] = 50
    pcd_nums[9] = 35
    data_path = '/media/kai/cornell_grasp_data/'
    total_pcd_num = np.sum(pcd_nums)
    rgbd_file = h5py.File(data_path + 'h5_data/rgbd.h5', 'r')
    rgbd_data = rgbd_file['RGBD'] 
    
    patch_file = h5py.File(data_path + 'h5_data/patch.h5', 'w')

    start_time = time.time()
    prev_num = 0
    cur_grasp_num = 0
    grasp_labels = []
    for i, n in enumerate(pcd_nums):
        #if i >= 1:
        #    break
        if i >= 1:
            prev_num += pcd_nums[i-1]
        for j in xrange(n):
            #print i, j
            #if j >= 10:
            #    break
            rgbd_id = prev_num + j
            rgbd = rgbd_data[rgbd_id]
            cpos_file = data_path + str(i+1).zfill(2) + '/pcd' + \
                     str(i+1).zfill(2) + str(j).zfill(2) + 'cpos.txt' 
            cneg_file = data_path + str(i+1).zfill(2) + '/pcd' + \
                     str(i+1).zfill(2) + str(j).zfill(2) + 'cneg.txt' 
            print cpos_file
            pos_rects = read_rect(cpos_file)
            pos_rects_num = pos_rects.shape[0]/4
            for r in xrange(pos_rects_num):
                #check if there is nan value in the rectangle 
                if np.isnan(pos_rects[r*4:(r+1)*4]).any():
                    continue
                palm_patch, finger_1_patch, finger_2_patch = \
                        extract_rgbd_patches(rgbd.copy(), pos_rects[r*4:(r+1)*4], rgbd_id=rgbd_id, 
                                grasp_id=r, save=False)
                #For drawing rectangle and save rgbd images. 
                #palm_patch, finger_1_patch, finger_2_patch = \
                #        extract_rgbd_patches(rgbd.copy(), pos_rects[r*4:(r+1)*4], (rgbd_id+1)*100+r, save=True)
                patch_file.create_dataset('palm_'+str(cur_grasp_num), data=palm_patch)
                patch_file.create_dataset('f_1_'+str(cur_grasp_num), data=finger_1_patch)
                patch_file.create_dataset('f_2_'+str(cur_grasp_num), data=finger_2_patch)
                cur_grasp_num += 1
                grasp_labels.append(1)
                palm_patch, finger_1_patch, finger_2_patch = \
                        extract_rgbd_patches(rgbd.copy(), pos_rects[r*4:(r+1)*4], rgbd_id=rgbd_id, 
                                grasp_id=r, save=False, mirror=True)
                #For drawing rectangle and save rgbd images. 
                #palm_patch, finger_1_patch, finger_2_patch = \
                #        extract_rgbd_patches(rgbd.copy(), pos_rects[r*4:(r+1)*4], (rgbd_id+1)*100+r, save=True)
                patch_file.create_dataset('palm_'+str(cur_grasp_num), data=palm_patch)
                patch_file.create_dataset('f_1_'+str(cur_grasp_num), data=finger_1_patch)
                patch_file.create_dataset('f_2_'+str(cur_grasp_num), data=finger_2_patch)
                cur_grasp_num += 1
                grasp_labels.append(1)

            neg_rects = read_rect(cneg_file)
            neg_rects_num = neg_rects.shape[0]/4
            for r in xrange(neg_rects_num):
                if np.isnan(neg_rects[r*4:(r+1)*4]).any():
                    continue
                palm_patch, finger_1_patch, finger_2_patch = \
                        extract_rgbd_patches(rgbd.copy(), neg_rects[r*4:(r+1)*4], rgbd_id=rgbd_id, 
                                grasp_id=-r, save=False)
                #For drawing rectangle and save rgbd images. 
                #palm_patch, finger_1_patch, finger_2_patch = \
                #        extract_rgbd_patches(rgbd.copy(), neg_rects[r*4:(r+1)*4], -((rgbd_id+1)*100+r), save=True)
                patch_file.create_dataset('palm_'+str(cur_grasp_num), data=palm_patch)
                patch_file.create_dataset('f_1_'+str(cur_grasp_num), data=finger_1_patch)
                patch_file.create_dataset('f_2_'+str(cur_grasp_num), data=finger_2_patch)
                cur_grasp_num += 1
                #grasp_labels.append(-1)
                grasp_labels.append(0)
                palm_patch, finger_1_patch, finger_2_patch = \
                        extract_rgbd_patches(rgbd.copy(), neg_rects[r*4:(r+1)*4], rgbd_id=rgbd_id, 
                                grasp_id=-r, save=False, mirror=True)
                #For drawing rectangle and save rgbd images. 
                #palm_patch, finger_1_patch, finger_2_patch = \
                #        extract_rgbd_patches(rgbd.copy(), neg_rects[r*4:(r+1)*4], -((rgbd_id+1)*100+r), save=True)
                patch_file.create_dataset('palm_'+str(cur_grasp_num), data=palm_patch)
                patch_file.create_dataset('f_1_'+str(cur_grasp_num), data=finger_1_patch)
                patch_file.create_dataset('f_2_'+str(cur_grasp_num), data=finger_2_patch)
                cur_grasp_num += 1
                #grasp_labels.append(-1)
                grasp_labels.append(0)

            elapsed_time = time.time() - start_time
            print 'elapsed_time: ', elapsed_time
            print '####################################'
    
    patch_file.create_dataset('labels', data=np.array(grasp_labels))
    elapsed_time = time.time() - start_time
    print 'total elapsed_time: ', elapsed_time

    rgbd_file.close()
    patch_file.close()

if __name__ == '__main__':
    main()

