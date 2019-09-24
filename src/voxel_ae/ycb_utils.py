import show_voxel
import h5py
import pickle
import numpy as np
#from ycb_objects import _OBJECTS, _OBJECTS_TEST
from plyfile import PlyData, PlyElement


def ply_to_voxel(object_path, voxel_res=(32,32,32), voxel_size=(0.01, 0.01, 0.01)):
    '''
    Read in an object ply and return the associated voxel grid
    '''
    # read in object ply file
    object_cloud = PlyData.read(object_path)
    obj_cloud = []
    num_pts = len(object_cloud['vertex'].data)
    for i in xrange(num_pts):
        pt = np.array((object_cloud['vertex']['x'][i],
                       object_cloud['vertex']['y'][i],
                       object_cloud['vertex']['z'][i]))
        #print pt
        obj_cloud.append(pt)

    # Find object frame and centroid
    #axes, centroid = find_obj_axes(np.asarray(obj_cloud))

    # Build rotation matrix from axes
    #r = np.matrix(np.stack(axes))

    # Set voxel limits
    voxel_min_loc = [0.0, 0.0, 0.0]
    voxel_min_loc[0] = -voxel_res[0] / 2 * voxel_size[0] # x_min
    voxel_min_loc[1] = -voxel_res[1] / 2 * voxel_size[1] # y_min
    voxel_min_loc[2] = -voxel_res[2] / 2 * voxel_size[2] # z_min

    # convert ply to voxel
    object_voxel = np.zeros(voxel_res)
    #for i in xrange(num_pts):
        # Center and rotate points
        #pt = r*(np.matrix(([object_cloud['vertex']['x'][i]],
        #                   [object_cloud['vertex']['y'][i]],
        #                   [object_cloud['vertex']['z'][i]]))-centroid)
        #x_idx = int((pt[0,0] - voxel_min_loc[0]) / voxel_size[0])
        #y_idx = int((pt[1,0] - voxel_min_loc[1]) / voxel_size[1])
        #z_idx = int((pt[2,0] - voxel_min_loc[2]) / voxel_size[2])

    for pt in obj_cloud:
        x_idx = int((pt[0] - voxel_min_loc[0]) / voxel_size[0])
        y_idx = int((pt[1] - voxel_min_loc[1]) / voxel_size[1])
        z_idx = int((pt[2] - voxel_min_loc[2]) / voxel_size[2])

        # Check voxel bounds
        if (x_idx >= 0 and x_idx < voxel_res[0] and
            y_idx >= 0 and y_idx < voxel_res[1] and
            z_idx >= 0 and z_idx < voxel_res[2]):
            object_voxel[x_idx, y_idx, z_idx] = 1.0
    return object_voxel

