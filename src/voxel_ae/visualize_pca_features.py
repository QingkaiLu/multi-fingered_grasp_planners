import show_voxel
import h5py
import pickle
import numpy as np
from sklearn.decomposition import PCA
from ycb_objects import _OBJECTS, _OBJECTS_TEST
from plyfile import PlyData, PlyElement

_DEBUG = False
_DEBUG_AXES = False
_DEBUG_OBJ_BUILDING = False
_VIZ_INPUT = True
_VIZ_COMPONENTS = True
_RUN_VIZ = True
_CLOUD_BERKELEY_SFX = '/clouds/merged_cloud.ply'
_CLOUD_GOOGLE_SFX = '/google_64k/nontextured.ply'

def convert_to_sparse_voxel_grid(voxel_grid):
    sparse_voxel_grid = []
    voxel_dim = voxel_grid.shape
    for i in xrange(voxel_dim[0]):
        for j in xrange(voxel_dim[1]):
            for k in xrange(voxel_dim[2]):
                if voxel_grid[i, j, k] > 0.5: # Binarize the sparse representation
                    sparse_voxel_grid.append([i, j, k])
    return np.asarray(sparse_voxel_grid)


def visualize_pca_feats(pca_model, object_voxel, select_dim=-1):
    # Encode object voxel grid to latent features
    latent_voxel = encode_voxel(object_voxel, pca_model)

    if _DEBUG:
        print 'Latent features, pre-filter:', latent_voxel
    # Zero out features not related to select dim
    if select_dim >= 0 and select_dim < pca_model.n_components:
        for i in xrange(pca_model.n_components):
            if i != select_dim:
                latent_voxel[0,i] = 0.0

    if _DEBUG:
        print 'Latent features, post-filter:', latent_voxel
    # Decode latent feature vector to voxel grid
    object_voxel_prime = decode_voxel(latent_voxel, pca_model,
                                      object_voxel.shape)

    # Visualize reprojected voxel grid
    if _RUN_VIZ:
        sparse_prime = convert_to_sparse_voxel_grid(object_voxel_prime)
        show_voxel.plot_voxel(sparse_prime,title='PCA Feat '+str(select_dim))

    return object_voxel_prime

def flatten_voxel(object_voxel):
    '''
    Flatten voxel grid into vector
    '''
    voxel_grid_dim = object_voxel.shape
    voxel_num = voxel_grid_dim[0] * voxel_grid_dim[1] * voxel_grid_dim[2]
    voxel_1d = np.reshape(object_voxel, voxel_num)
    return voxel_1d

def encode_voxel(object_voxel, pca_model):
    '''
    Encode an object voxel grid into the associated pca features
    '''
    voxel_1d = flatten_voxel(object_voxel)

    # Project into latent space
    latent_voxel = pca_model.transform([voxel_1d])

    return latent_voxel

def decode_voxel(latent_voxel, pca_model, voxel_shape):
    '''
    Decode an object pca vector into an object voxel grid
    '''
    # Project from latent space into voxel space and binarize output
    object_flat = pca_model.inverse_transform(latent_voxel) > 0.5

    # Rearrange voxel structure into correct shape, convert from bool to float
    object_voxel = np.reshape(object_flat.astype('float'), voxel_shape)

    return object_voxel

def ply_to_voxel(object_path, voxel_res=(20,20,20), voxel_size=(0.01, 0.01, 0.01)):
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
        obj_cloud.append(pt)

    # Find object frame and centroid
    axes, centroid = find_obj_axes(np.asarray(obj_cloud))

    # Build rotation matrix from axes
    r = np.matrix(np.stack(axes))

    # Set voxel limits
    voxel_min_loc = [0.0, 0.0, 0.0]
    voxel_min_loc[0] = -voxel_res[0] / 2 * voxel_size[0] # x_min
    voxel_min_loc[1] = -voxel_res[1] / 2 * voxel_size[1] # y_min
    voxel_min_loc[2] = -voxel_res[2] / 2 * voxel_size[2] # z_min

    # convert ply to voxel
    object_voxel = np.zeros(voxel_res)
    for i in xrange(num_pts):
        # Center and rotate points
        pt = r*(np.matrix(([object_cloud['vertex']['x'][i]],
                           [object_cloud['vertex']['y'][i]],
                           [object_cloud['vertex']['z'][i]]))-centroid)
        x_idx = int((pt[0,0] - voxel_min_loc[0]) / voxel_size[0])
        y_idx = int((pt[1,0] - voxel_min_loc[1]) / voxel_size[1])
        z_idx = int((pt[2,0] - voxel_min_loc[2]) / voxel_size[2])

        # Check voxel bounds
        if (x_idx >= 0 and x_idx < voxel_res[0] and
            y_idx >= 0 and y_idx < voxel_res[1] and
            z_idx >= 0 and z_idx < voxel_res[2]):
            object_voxel[x_idx, y_idx, z_idx] = 1.0
    return object_voxel

def find_obj_axes(obj_cloud):
    '''
    Given a point cloud determine a valid, right-handed coordinate frame
    '''
    pca_operator = PCA(n_components=3, svd_solver='full')
    pca_operator.fit(obj_cloud)
    centroid = np.matrix(pca_operator.mean_).T
    x_axis = pca_operator.components_[0]
    y_axis = pca_operator.components_[1]
    z_axis = np.cross(x_axis,y_axis)

    if _DEBUG_AXES:
        print 'PCA centroid', centroid
        print 'x_axis', x_axis
        print 'y_axis', y_axis
        print 'z_axis', z_axis
    return (x_axis, y_axis, z_axis), centroid

def fit_pca_full_objects(base_obj_path='/media/thermans/data/ycb/',latent_dim=15,
                         pca_model_path='../../models/grasp_type_planner/train_models/pca/pca_full_obj_mesh.model',
                         whiten=True):
    '''
    Fit PCA data to full object point clouds
    '''
    object_voxels1D = []
    for object_name in _OBJECTS:
        # Read in data from ply files in ycb data set
        try:
            object_path = base_obj_path+object_name+_CLOUD_GOOGLE_SFX
            if _DEBUG:
                print 'Request to open file', object_path
            object_voxel = ply_to_voxel(object_path)
        except IOError:
            try:
                object_path = base_obj_path+object_name+_CLOUD_BERKELEY_SFX
                if _DEBUG:
                    print 'Request to open file', object_path
                object_voxel = ply_to_voxel(object_path)
            except IOError:
                continue
        # Visualize for debugging
        if _DEBUG_OBJ_BUILDING:
            sparse_voxel = convert_to_sparse_voxel_grid(object_voxel)
            show_voxel.plot_voxel(sparse_voxel, show=True, title='Object Model Debug')

        # Compile_into_list
        flat_object = flatten_voxel(object_voxel)
        object_voxels1D.append(flat_object)

    object_voxels = np.asarray(object_voxels1D)

    # Fit PCA features to full dataset
    pca_model = PCA(n_components=latent_dim, whiten=whiten, svd_solver='full')
    pca_model.fit(object_voxels)
    object_voxels_latent = pca_model.transform(object_voxels)
    if _DEBUG:
        print object_voxels.shape
        print object_voxels_latent.shape

    # Save PCA model to disk using pickle
    pickle.dump(pca_model, open(pca_model_path, 'wb'))
    return pca_model

def run(pca_model_path='../../models/grasp_type_planner/train_models/pca/pca_full_obj_mesh.model'):

    voxel_path = '/media/thermans/data/grasp_type_data/train_data/precision/prec_grasps_1/prec_align_failure_grasps.h5'
    grasp_voxel_file = h5py.File(voxel_path, 'r')

    # read in input voxel data / or point cloud
    grasp_voxel_grids = grasp_voxel_file['grasp_voxel_grids'][()]
    grasp_voxel_file.close()

    # Load PCA model
    pca_model = pickle.load(open(pca_model_path, 'rb'))

    for i, object_voxel in enumerate(grasp_voxel_grids):
        # object_voxel = grasp_voxel_grids[i]
        if i > 20 and _DEBUG:
            break
        if _VIZ_COMPONENTS:
            for k in xrange(pca_model.n_components):
                print 'Showing object', i, 'dimension', k

                # Visualize input
                if _VIZ_INPUT:
                    sparse_voxel = convert_to_sparse_voxel_grid(object_voxel)
                    show_voxel.plot_voxel(sparse_voxel, show=False, title='Input Voxel')

                # Visualize feats
                object_prime = visualize_pca_feats(pca_model, object_voxel, k)
        else:
            if _VIZ_INPUT:
                sparse_voxel = convert_to_sparse_voxel_grid(object_voxel)
                show_voxel.plot_voxel(sparse_voxel, show=False, title='Input voxel')

            # Visualize feats
            print 'Showing reprojected object', i
            object_prime = visualize_pca_feats(pca_model, object_voxel)

