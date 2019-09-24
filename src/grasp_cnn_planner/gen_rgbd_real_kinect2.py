#!/usr/bin/env python

import rospy
import roslib
roslib.load_manifest('prob_grasp_planner')
import tf
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped, Pose, Quaternion
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import os
import h5py

class GenRgbdRealKinect2:
    def __init__(self, rgbd_patches_save_path=None, use_hd=True):
        self.rgbd_channels = 8
        self.plot_palm_fingers = False
        self.save_rgbd = True
        # if rgbd_patches_save_path is not None:
        #     self.rgbd_file_name = rgbd_patches_save_path + 'grasp_rgbd.h5'
        #     self.intialize_grasp_data_file(self.rgbd_file_name)
        #     self.grasp_patches_file_name = rgbd_patches_save_path + 'grasp_patches.h5'
        #     self.intialize_grasp_data_file(self.grasp_patches_file_name)
        #     self.finger_patches_file_name = rgbd_patches_save_path + 'grasp_finger_patches.h5'
        #     self.intialize_grasp_data_file(self.finger_patches_file_name)
        #     self.finger_no_rot_patches_file_name = rgbd_patches_save_path + 'grasp_finger_no_rot_patches.h5'
        #     self.intialize_grasp_data_file(self.finger_no_rot_patches_file_name)
        #     self.close_hand_patches_file_name = rgbd_patches_save_path + 'close_grasp_finger_patches.h5'
        #     self.intialize_grasp_data_file(self.close_hand_patches_file_name)
        #     self.close_hand_no_rot_patches_file_name = rgbd_patches_save_path + 'close_grasp_finger_no_rot_patches.h5'
        #     self.intialize_grasp_data_file(self.close_hand_no_rot_patches_file_name)

        self.save_rgbd_image_path = rgbd_patches_save_path
        
        # self.rgbd_n_channels = 8
        self.use_hd = use_hd
        self.rgbd_n_channels = 8

        self.hd_image_n_rows = -1
        self.hd_image_n_cols = -1
        self.sd_image_n_rows = -1
        self.sd_image_n_cols = -1
        self.hd_to_sd_ratio = -1.
       
        if self.use_hd:
            self.hd_to_sd_ratio = 0.5 #1. / 3.
            self.hd_crop_left_width = 50
            self.hd_crop_right_width = 100

    def intialize_grasp_data_file(self, file_name):
        '''
            Initialize the rgbd patches h5 file.
        '''
        rgbd_patches_file = h5py.File(file_name, 'a')
        grasps_number_key = 'grasps_number'
        if grasps_number_key not in rgbd_patches_file:
            rgbd_patches_file.create_dataset(grasps_number_key, data=0)
        rgbd_patches_file.close()

    def get_rgbd_image(self, bgr_img_path, depth_img_path, normal_msg):
        '''
            Get rgb + depth + normal + curvature map.
        '''
        #cv_bridge = CvBridge()
        #rospy.loginfo(rgb_msg.encoding)
        #bgr_image = cv_bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
        #depth_image = cv_bridge.imgmsg_to_cv2(depth_msg, '32FC1')
        bgr_image = cv2.imread(bgr_img_path, cv2.IMREAD_COLOR)
        
        depth_image = cv2.imread(depth_img_path, cv2.IMREAD_COLOR)#cv2.IMREAD_GRAYSCALE)
        depth_image = cv2.cvtColor(depth_image, cv2.COLOR_RGB2GRAY)
        # depth_image = cv2.imread(depth_img_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        self.hd_image_n_rows = bgr_image.shape[0]
        self.hd_image_n_cols = bgr_image.shape[1]
        if self.use_hd:
            self.sd_image_n_rows = int(self.hd_image_n_rows * self.hd_to_sd_ratio)
            self.sd_image_n_cols = int(self.hd_image_n_cols * self.hd_to_sd_ratio)
        else:
            self.sd_image_n_rows = depth_image.shape[0] 
            self.sd_image_n_cols = depth_image.shape[1]
        normal_points = pc2.read_points(normal_msg, skip_nans=False)
        normal_points = np.array(list(normal_points))
        if not self.use_hd:
            normal_points = np.reshape(normal_points, (self.sd_image_n_rows, self.sd_image_n_cols, 4))
        else:
            normal_points = np.reshape(normal_points, (self.hd_image_n_rows, self.hd_image_n_cols, 4))

        normal_points[np.logical_or(np.logical_or(np.isinf(normal_points), np.isneginf(normal_points)),
                                    np.isnan(normal_points))] = 0.  
        if not self.use_hd:
            self.sd_image_n_rows = depth_image.shape[0]
            self.sd_image_n_cols = depth_image.shape[1]

        rgbd_image = np.zeros((self.sd_image_n_rows, self.sd_image_n_cols, self.rgbd_n_channels))
        #rgbd_image = np.zeros((self.hd_image_n_rows, self.hd_image_n_cols, self.n_channels))
        #bgr_image = cv2.resize(bgr_image, (self.hd_image_n_cols, self.hd_image_n_rows))
        
        if not self.use_hd:
            self.hd_to_sd_ratio = float(self.sd_image_n_rows) / self.hd_image_n_rows

            # Resize the hd image height to sd height
            bgr_image = cv2.resize(bgr_image, (int(self.hd_image_n_cols * self.hd_to_sd_ratio), self.sd_image_n_rows))
            # Crop the extra scaled rgb parts out 
            bgr_center = (bgr_image.shape[1] / 2., bgr_image.shape[0] / 2.) 
            bgr_image = cv2.getRectSubPix(bgr_image.astype('float32'), (self.sd_image_n_cols, self.sd_image_n_rows),
                                            bgr_center)
        else:
            bgr_image = cv2.resize(bgr_image, (self.sd_image_n_cols, self.sd_image_n_rows))
            depth_image = cv2.resize(depth_image, (self.sd_image_n_cols, self.sd_image_n_rows))
            normal_points = cv2.resize(normal_points, (self.sd_image_n_cols, self.sd_image_n_rows))

        #cv2.imshow('bgr_resize', bgr_image)
        #cv2.imshow('depth_resize', depth_image)
        #cv2.waitKey(0)
    
        rgbd_image[:, :, :3] = bgr_image
        rgbd_image[:, :, 3] = depth_image
        rgbd_image[:, :, 4:] = normal_points
        
        # Crop hd rgbd to make sure the rgb and depth cover the same area of scene.
        rgbd_image = rgbd_image[:, self.hd_crop_left_width:-self.hd_crop_right_width, :]

        #rgbd_image = cv2.resize(rgbd_image, (self.sd_image_n_cols, self.sd_image_n_rows))

        if self.save_rgbd:
            self.save_rgbd_image(rgbd_image, self.save_rgbd_image_path)
    
        return rgbd_image
    
    def save_rgbd_image(self, rgbd, path):
        '''
            Save rgb + depth + normal + curvature map.
        '''
        rgb = np.copy(rgbd[:, :, :3]).astype('uint8')
        cv2.imwrite(path + '/rgb.png', rgb)
    
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
    
    def rotateRgbd(self, rgbd, angle, rot_center):
        '''
            Rotate rgb + depth + normal + curvature map.
        '''
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
    
    def extract_rgbd_patch(self, rgbd, center, patch_size):
        '''
            Extract rgb + depth + normal + curvature map patches.

        '''
        # To do: replace with a macro
        center = tuple(map(int, center))
        patch = np.zeros((patch_size, patch_size, self.rgbd_channels))
        patch_sizes = (patch_size, patch_size)
        for i in xrange(self.rgbd_channels):
            patch[:, :, i] = cv2.getRectSubPix(rgbd[:, :, i].astype('float32'), patch_sizes, center)
        
        half_width = patch_sizes[0] / 2
        half_height = patch_sizes[1] / 2

        left_out_boarder = half_width - 1 - center[0]
        if left_out_boarder > 0:
            patch[:, 0:left_out_boarder, :] = 0.
        top_out_boarder = half_height - 1 - center[1]
        if top_out_boarder > 0:
            patch[0:top_out_boarder, :, :] = 0.

        cols = rgbd.shape[1]
        rows = rgbd.shape[0]
        right_out_boarder = center[0] + half_width - cols + 1  
        if right_out_boarder > 0:
            patch[:, cols:cols + right_out_boarder, :] = 0.
        bottom_out_boarder = center[1] + half_height - rows + 1  
        if bottom_out_boarder > 0:
            patch[rows:rows + bottom_out_boarder, :, :] = 0.

        #cv2.imshow('rgb', rgbd[:, :, :3].astype('uint8'))
        #cv2.waitKey(0)
    
        #cv2.imshow('patch', patch[:, :, :3].astype('uint8'))
        #cv2.waitKey(0)
        return patch
   
    def resize_finger_palm_loc(self, palm_rgb_loc, finger_tip_rgb_locs):
        '''
            Resize finger tip and palm center locations from hd to sd.
        '''
        palm_sd_loc = np.zeros(palm_rgb_loc.shape)
        palm_sd_loc[0] = np.array(palm_rgb_loc[0]) * self.hd_to_sd_ratio 
        palm_sd_loc[1] = np.array(palm_rgb_loc[1]) * self.hd_to_sd_ratio

        finger_tip_sd_locs = np.zeros(finger_tip_rgb_locs.shape)
        finger_tip_sd_locs[:, 0] = np.array(finger_tip_rgb_locs[:, 0]) * self.hd_to_sd_ratio
        finger_tip_sd_locs[:, 1] = np.array(finger_tip_rgb_locs[:, 1]) * self.hd_to_sd_ratio

        palm_sd_loc[0] -= self.hd_crop_left_width
        finger_tip_sd_locs[:, 0] -= self.hd_crop_left_width
        return palm_sd_loc, finger_tip_sd_locs

    def resize_palm_loc(self, palm_rgb_loc):
        '''
            Resize palm center locations from hd to sd.
        '''
        palm_sd_loc = np.zeros(palm_rgb_loc.shape)
        palm_sd_loc[0] = np.array(palm_rgb_loc[0]) * self.hd_to_sd_ratio 
        palm_sd_loc[1] = np.array(palm_rgb_loc[1]) * self.hd_to_sd_ratio

        palm_sd_loc[0] -= self.hd_crop_left_width
        return palm_sd_loc


    def plot_palm_to_finger_lines(self, rgb, palm_sd_loc, finger_tip_sd_locs):
        '''
            Plot lines between the palm center and the finger tips.
        '''
        rgb_copy = rgb.copy().astype('uint8')
        for i, finger_loc in enumerate(finger_tip_sd_locs):
            color = [0, 0, 0]
            if i < 3:
                color[i] = 255
            else:
                color = [0, 255, 255]
            color = tuple(color)
            cv2.line(rgb_copy, tuple(palm_sd_loc.astype('uint32')), 
                    tuple(finger_loc.astype('uint32')), color, 2)
        return rgb_copy

    def save_rgbd_into_h5(self, rgbd, init_hand_pcd_true_config, init_hand_pcd_goal_config, 
                            close_hand_pcd_config, object_id, grasp_id, object_name):
        '''
            Save rgbd into h5 file.
        '''
        rgbd_file = h5py.File(self.rgbd_file_name, 'r+') 
        grasp_sample_id = 'grasp_' + str(rgbd_file['grasps_number'][()])

        object_grasp_id_key = grasp_sample_id + '_object_grasp_id'
        if object_grasp_id_key not in rgbd_file:
            object_grasp_id = 'object_' + str(object_id) + '_grasp_' + str(grasp_id)
            rgbd_file.create_dataset(object_grasp_id_key, data=object_grasp_id)
        rgbd_key = grasp_sample_id + '_rgbd'
        if rgbd_key not in rgbd_file:
            rgbd_file.create_dataset(rgbd_key, data=rgbd)

        preshape_palm_pcd_pose_list = [init_hand_pcd_goal_config.palm_pose.pose.position.x, init_hand_pcd_goal_config.palm_pose.pose.position.y,
               init_hand_pcd_goal_config.palm_pose.pose.position.z, init_hand_pcd_goal_config.palm_pose.pose.orientation.x,
               init_hand_pcd_goal_config.palm_pose.pose.orientation.y, init_hand_pcd_goal_config.palm_pose.pose.orientation.z, 
               init_hand_pcd_goal_config.palm_pose.pose.orientation.w]
        palm_pcd_pose_key = grasp_sample_id + '_preshape_palm_pcd_pose'
        if palm_pcd_pose_key not in rgbd_file:
            rgbd_file.create_dataset(palm_pcd_pose_key, 
                    data=preshape_palm_pcd_pose_list)

        true_preshape_palm_pcd_pose_list = [init_hand_pcd_true_config.palm_pose.pose.position.x, init_hand_pcd_true_config.palm_pose.pose.position.y,
               init_hand_pcd_true_config.palm_pose.pose.position.z, init_hand_pcd_true_config.palm_pose.pose.orientation.x,
               init_hand_pcd_true_config.palm_pose.pose.orientation.y, init_hand_pcd_true_config.palm_pose.pose.orientation.z, 
               init_hand_pcd_true_config.palm_pose.pose.orientation.w]
        true_palm_pcd_pose_key = grasp_sample_id + '_true_preshape_palm_pcd_pose'
        if true_palm_pcd_pose_key not in rgbd_file:
            rgbd_file.create_dataset(true_palm_pcd_pose_key, 
                    data=true_preshape_palm_pcd_pose_list)

        close_shape_palm_pcd_pose_list = [close_hand_pcd_config.palm_pose.pose.position.x, close_hand_pcd_config.palm_pose.pose.position.y,
               close_hand_pcd_config.palm_pose.pose.position.z, close_hand_pcd_config.palm_pose.pose.orientation.x,
               close_hand_pcd_config.palm_pose.pose.orientation.y, close_hand_pcd_config.palm_pose.pose.orientation.z, 
               close_hand_pcd_config.palm_pose.pose.orientation.w]
        palm_pcd_pose_key = grasp_sample_id + '_close_shape_palm_pcd_pose'
        if palm_pcd_pose_key not in rgbd_file:
            rgbd_file.create_dataset(palm_pcd_pose_key, 
                    data=close_shape_palm_pcd_pose_list)

        preshape_js_position_key = grasp_sample_id + '_preshape_js_position'
        if preshape_js_position_key not in rgbd_file:
            rgbd_file.create_dataset(preshape_js_position_key, 
                    data=init_hand_pcd_goal_config.hand_joint_state.position)

        true_preshape_js_position_key = grasp_sample_id + '_true_preshape_js_position'
        if true_preshape_js_position_key not in rgbd_file:
            rgbd_file.create_dataset(true_preshape_js_position_key, 
                    data=init_hand_pcd_true_config.hand_joint_state.position)

        close_shape_js_position_key = grasp_sample_id + '_close_shape_js_position'
        if close_shape_js_position_key not in rgbd_file:
            rgbd_file.create_dataset(close_shape_js_position_key, 
                    data=close_hand_pcd_config.hand_joint_state.position)

        rgbd_file['grasps_number'][()] += 1
        rgbd_file.close()

    def save_grasp_patches_into_h5(self, grasp_patch, close_grasp_patch, preshape_true_config, 
                                   preshape_goal_config, close_hand_config, grasp_label,
                                   object_id, grasp_id, object_name):
        '''
            Save the grasp rgbd patch and grasp preshape configuration into the h5 file.
        '''
        grasp_patches_file = h5py.File(self.grasp_patches_file_name, 'r+') 
        grasp_sample_id = 'grasp_' + str(grasp_patches_file['grasps_number'][()])

        object_grasp_id_key = grasp_sample_id + '_object_grasp_id'
        if object_grasp_id_key not in grasp_patches_file:
            object_grasp_id = 'object_' + str(object_id) + '_grasp_' + str(grasp_id)
            grasp_patches_file.create_dataset(object_grasp_id_key, data=object_grasp_id)
        grasp_patch_key = grasp_sample_id + '_grasp_patch'
        if grasp_patch_key not in grasp_patches_file:
            grasp_patches_file.create_dataset(grasp_patch_key, data=grasp_patch)
        close_grasp_patch_key = grasp_sample_id + '_close_grasp_patch'
        if close_grasp_patch_key not in grasp_patches_file:
            grasp_patches_file.create_dataset(close_grasp_patch_key, data=close_grasp_patch)
        grasp_label_key = grasp_sample_id + '_grasp_label'
        if grasp_label_key not in grasp_patches_file:
            grasp_patches_file.create_dataset(grasp_label_key, data=grasp_label)
        grasp_true_config_key = grasp_sample_id + '_preshape_true_config'
        if grasp_true_config_key not in grasp_patches_file:
            grasp_patches_file.create_dataset(grasp_true_config_key, data=preshape_true_config)
        grasp_goal_config_key = grasp_sample_id + '_preshape_goal_config'
        if grasp_goal_config_key not in grasp_patches_file:
            grasp_patches_file.create_dataset(grasp_goal_config_key, data=preshape_goal_config)
        close_grasp_config_key = grasp_sample_id + '_close_hand_config'
        if close_grasp_config_key not in grasp_patches_file:
            grasp_patches_file.create_dataset(close_grasp_config_key, data=close_hand_config)
 
        grasp_patches_file['grasps_number'][()] += 1
        grasp_patches_file.close()

    def get_grasp_patch(self, rgbd, palm_rgb_loc, close_palm_rgb_loc, preshape_true_config, preshape_goal_config,
                        close_hand_config, patch_size=400, object_id=None, grasp_id=None, object_name=None, 
                        grasp_label=None, save=False):
        '''
            Given the palm center in rgbd image, extract the rgbd patches for 
            palm center and finger tips.
        '''
        grasp_patch = self.extract_rgbd_patch(rgbd, tuple(palm_rgb_loc), patch_size)  
    
        if save:
            path = self.save_rgbd_image_path + 'patches/'
            path += str(object_id) + '_' + str(grasp_id) + '_' + str(object_name)
            if not os.path.exists(path):
                os.makedirs(path)
    
            grasp_path = path + '/grasp/'
            if not os.path.exists(grasp_path):
                os.makedirs(grasp_path)
            self.save_rgbd_image(grasp_patch, grasp_path)

        close_grasp_patch = self.extract_rgbd_patch(rgbd, tuple(close_palm_rgb_loc), patch_size)  
    
        if save:
            path = self.save_rgbd_image_path + 'patches_close/'
            path += str(object_id) + '_' + str(grasp_id) + '_' + str(object_name)
            if not os.path.exists(path):
                os.makedirs(path)
    
            grasp_path = path + '/grasp/'
            if not os.path.exists(grasp_path):
                os.makedirs(grasp_path)
            self.save_rgbd_image(grasp_patch, grasp_path)

        self.save_grasp_patches_into_h5(grasp_patch, close_grasp_patch, preshape_true_config, 
                                        preshape_goal_config, close_hand_config, grasp_label,
                                        object_id, grasp_id, object_name)

        return grasp_patch
 
    def get_palm_patch(self, rgbd_image, palm_image_loc, patch_size):
        palm_image_loc = self.resize_palm_loc(palm_image_loc)
        palm_patch = self.extract_rgbd_patch(rgbd_image, tuple(palm_image_loc), patch_size)  
        return palm_patch

    def save_rgbd_patches_into_h5(self, patches_file_name, palm_patch, finger_tip_patches, grasp_label, 
                                    object_id, grasp_id, object_name):
        '''
            Save palm and finger tip rgbd patches and grasp labels into the h5 file.
        '''
        #rgbd_patches_file = h5py.File(self.finger_patches_file_name, 'r+') 
        rgbd_patches_file = h5py.File(patches_file_name, 'r+') 
        grasp_sample_id = 'grasp_' + str(rgbd_patches_file['grasps_number'][()])
        object_grasp_id_key = grasp_sample_id + '_object_grasp_id'
        if object_grasp_id_key not in rgbd_patches_file:
            object_grasp_id = 'object_' + str(object_id) + '_grasp_' + str(grasp_id)
            rgbd_patches_file.create_dataset(object_grasp_id_key, data=object_grasp_id)
        palm_patch_key = grasp_sample_id + '_palm_patch'
        if palm_patch_key not in rgbd_patches_file:
            rgbd_patches_file.create_dataset(palm_patch_key, data=palm_patch)
        finger_tip_patches_key = grasp_sample_id + '_finger_tip_patches'
        if finger_tip_patches_key not in rgbd_patches_file:
            rgbd_patches_file.create_dataset(finger_tip_patches_key, data=finger_tip_patches)
        grasp_label_key = grasp_sample_id + '_grasp_label'
        if grasp_label_key not in rgbd_patches_file:
            rgbd_patches_file.create_dataset(grasp_label_key, data=grasp_label)

        rgbd_patches_file['grasps_number'][()] += 1
        rgbd_patches_file.close()


    def get_finger_palm_patches(self, rgbd, palm_rgb_loc, finger_tip_rgb_locs, close_hand=False, patch_size=200, 
                                finer_level=2, object_id=None, grasp_id=None, object_name=None, grasp_label=None,
                                save=True):
        '''
            Given the palm center and finger tip locations in rgb image, extract the rgbd patches for 
            palm center and finger tips with rotation. The patches axis are along the line connecting 
            the palm and finger tip.
        '''
        if self.plot_palm_fingers:
            rgbd[:, :, :3] = self.plot_palm_to_finger_lines(rgbd[:, :, :3], palm_rgb_loc, finger_tip_rgb_locs)

        #if close_hand:
        #    rgbd[:, :, :3] = self.plot_palm_to_finger_lines(rgbd[:, :, :3], palm_rgb_loc, finger_tip_rgb_locs)

        if self.save_rgbd:
            self.save_rgbd_image(rgbd, self.save_rgbd_image_path)

        palm_patch = self.extract_rgbd_patch(rgbd, tuple(palm_rgb_loc), patch_size)  
       
        finger_tip_patches = []
        rgbd_rot_fingers = []
        for i, finger_loc in enumerate(finger_tip_rgb_locs):
            dxy = finger_loc - palm_rgb_loc
            angle = np.math.atan2(dxy[1], dxy[0])
            angle = np.degrees(angle)
            rgbd_rot_finger = self.rotateRgbd(rgbd, angle, tuple(finger_loc))
            finger_patch = self.extract_rgbd_patch(rgbd_rot_finger, tuple(finger_loc), 
                                                    patch_size/finer_level)
            finger_tip_patches.append(finger_patch)
            rgbd_rot_fingers.append(rgbd_rot_finger)
    
        if save:
            if not close_hand:
                path = self.save_rgbd_image_path + 'patches/'
            else:
                path = self.save_rgbd_image_path + 'patches_close/'
            path += str(object_id) + '_' + str(grasp_id) + '_' + str(object_name)
            if not os.path.exists(path):
                os.makedirs(path)
    
            palm_path = path + '/palm/'
            if not os.path.exists(palm_path):
                os.makedirs(palm_path)
            self.save_rgbd_image(palm_patch, palm_path)
            
            for i, finger_patch in enumerate(finger_tip_patches):
                rgbd_rot_finger = rgbd_rot_fingers[i]
                finger_rgbd_rot_path = path + '/rgbd_rot_f' + str(i) + '/'
                if not os.path.exists(finger_rgbd_rot_path):
                    os.makedirs(finger_rgbd_rot_path)
                self.save_rgbd_image(rgbd_rot_finger, finger_rgbd_rot_path)
    
                finger_path = path + '/finger' + str(i) + '/'
                if not os.path.exists(finger_path):
                    os.makedirs(finger_path)
                self.save_rgbd_image(finger_patch, finger_path)
        
            if not close_hand:
                patches_file_name = self.finger_patches_file_name
            else:
                patches_file_name = self.close_hand_patches_file_name
            self.save_rgbd_patches_into_h5(patches_file_name, palm_patch, finger_tip_patches, grasp_label, 
                                    object_id, grasp_id, object_name)

        return palm_patch, finger_tip_patches
    
    def get_finger_palm_no_rot_patches(self, rgbd, palm_rgb_loc, finger_tip_rgb_locs, close_hand=False, 
                                       patch_size=200, finer_level=2, object_id=None, grasp_id=None, 
                                       object_name=None, grasp_label=None, save=False):
        '''
            Given the palm center and finger tip locations in rgb image, extract the rgbd patches for 
            palm center and finger tips without rotation. The patches axis are along the image orientation.
        '''
        if self.plot_palm_fingers:
            rgbd[:, :, :3] = self.plot_palm_to_finger_lines(rgbd[:, :, :3], palm_rgb_loc, finger_tip_rgb_locs)

        if self.save_rgbd:
            self.save_rgbd_image(rgbd, self.save_rgbd_image_path)

        palm_patch = self.extract_rgbd_patch(rgbd, tuple(palm_rgb_loc), patch_size)  
       
        finger_tip_patches = []
        rgbd_rot_fingers = []
        for i, finger_loc in enumerate(finger_tip_rgb_locs):
            finger_patch = self.extract_rgbd_patch(rgbd, tuple(finger_loc), 
                                                    patch_size/finer_level)
            finger_tip_patches.append(finger_patch)
    
        if save:
            if not close_hand:
                path = self.save_rgbd_image_path + 'patches_no_rot/'
            else:
                path = self.save_rgbd_image_path + 'patches_no_rot_close/'
            path += str(object_id) + '_' + str(grasp_id) + '_' + str(object_name)
            if not os.path.exists(path):
                os.makedirs(path)
    
            palm_path = path + '/palm/'
            if not os.path.exists(palm_path):
                os.makedirs(palm_path)
            self.save_rgbd_image(palm_patch, palm_path)
            
            for i, finger_patch in enumerate(finger_tip_patches):
                finger_path = path + '/finger' + str(i) + '/'
                if not os.path.exists(finger_path):
                    os.makedirs(finger_path)
                self.save_rgbd_image(finger_patch, finger_path)
        
            if not close_hand:
                patches_file_name = self.finger_no_rot_patches_file_name
            else:
                patches_file_name = self.close_hand_no_rot_patches_file_name
            self.save_rgbd_patches_into_h5(patches_file_name, palm_patch, finger_tip_patches, grasp_label, 
                                    object_id, grasp_id, object_name)

        return palm_patch, finger_tip_patches

