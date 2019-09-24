#!/usr/bin/env python
import numpy as np
import cv2
import h5py


def test_rgbd():
    data_path = '/media/kai/multi_finger_sim_data_complete_v4/'
    grasp_rgbd_file_path = data_path + 'grasp_rgbd.h5'
    rgbd_file = h5py.File(grasp_rgbd_file_path, 'r')
    grasp_0_rgbd = rgbd_file['grasp_0_rgbd'][()]
    print grasp_0_rgbd
    grasp_1_rgbd = rgbd_file['grasp_1_rgbd'][()]
    print grasp_1_rgbd
    grasp_2_rgbd = rgbd_file['grasp_2_rgbd'][()]
    print grasp_2_rgbd
    grasp_3_rgbd = rgbd_file['grasp_3_rgbd'][()]
    print grasp_3_rgbd
    grasp_4_rgbd = rgbd_file['grasp_4_rgbd'][()]
    print grasp_4_rgbd

    print np.array_equal(grasp_0_rgbd, grasp_1_rgbd)
    print np.array_equal(grasp_2_rgbd, grasp_1_rgbd)
    print np.array_equal(grasp_2_rgbd, grasp_3_rgbd)
    print np.array_equal(grasp_3_rgbd, grasp_4_rgbd)
   

    rgbd_file.close()

if __name__ == '__main__':
    test_rgbd()
