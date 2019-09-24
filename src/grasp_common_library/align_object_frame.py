import numpy as np
import tf
from geometry_msgs.msg import Quaternion, QuaternionStamped
import copy


#Compute angles between two vectors, code is from:
#https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def find_min_ang_vec(world_vec, cam_vecs):
    min_ang = float('inf')
    min_ang_idx = -1
    min_ang_vec = None
    for i in xrange(cam_vecs.shape[1]):
        angle = angle_between(world_vec, cam_vecs[:, i])
        larger_half_pi = False
        if angle > np.pi * 0.5:
            angle = np.pi - angle
            larger_half_pi = True
        if angle < min_ang:
            min_ang = angle
            min_ang_idx = i
            if larger_half_pi:
                min_ang_vec = -cam_vecs[:, i]
            else:
                min_ang_vec = cam_vecs[:, i]

    return min_ang_vec, min_ang_idx


def align_object(object_world, listener):
    '''
    Align the object frame coordinates to be consistent relative to the world frame.
    '''

    object_world_aligned = copy.deepcopy(object_world)
    #Transform object frame from camera frame to world frame
    #listener = tf.TransformListener()
    obj_qnt_stamped_world = QuaternionStamped()
    obj_qnt_stamped_world.header.frame_id = object_world.header.frame_id
    obj_qnt_stamped_world.quaternion = copy.deepcopy(object_world.pose.orientation)
    #object_qnt_world = listener.transformQuaternion('world', obj_qnt_stamped_cam)

    #Convert from quarternion to matrix
    quarternion = [obj_qnt_stamped_world.quaternion.x, obj_qnt_stamped_world.quaternion.y,
                    obj_qnt_stamped_world.quaternion.z, obj_qnt_stamped_world.quaternion.w]
    trans_mat = tf.transformations.quaternion_matrix(quarternion)
    rot_mat = trans_mat[:3, :3]

    #Convert from matrix to quarternion
    # Adapt to the new robot arm pose: 
    # robot arm 1st joint zero angles faces the table.
    align_trans_matrix = np.identity(4)
    object_size = [object_world.width, object_world.height, object_world.depth]

    #Find and align x axes.
    x_axis = [1., 0., 0.]
    align_x_axis, min_ang_axis_idx = find_min_ang_vec(x_axis, rot_mat) 
    rot_mat = np.delete(rot_mat, min_ang_axis_idx, axis=1)
    #align_trans_matrix[:3, 1] = align_x_axis
    align_trans_matrix[:3, 0] = align_x_axis
    object_world_aligned.height = object_size[min_ang_axis_idx]

    #y axes
    y_axis = [0., 1., 0.]
    align_y_axis, min_ang_axis_idx = find_min_ang_vec(y_axis, rot_mat) 
    rot_mat = np.delete(rot_mat, min_ang_axis_idx, axis=1)
    #align_trans_matrix[:3, 0] = -align_y_axis
    align_trans_matrix[:3, 1] = align_y_axis
    object_world_aligned.width = object_size[min_ang_axis_idx]

    #z axes
    z_axis = [0., 0., 1.]
    align_z_axis, min_ang_axis_idx = find_min_ang_vec(z_axis, rot_mat) 
    align_trans_matrix[:3, 2] = align_z_axis
    object_world_aligned.depth = object_size[min_ang_axis_idx]

    align_qtn_array = tf.transformations.quaternion_from_matrix(align_trans_matrix)

    align_qnt_stamped = QuaternionStamped()
    align_qnt_stamped.header.frame_id = object_world.header.frame_id
    align_qnt_stamped.quaternion.x, align_qnt_stamped.quaternion.y, \
            align_qnt_stamped.quaternion.z, align_qnt_stamped.quaternion.w = align_qtn_array

    object_world_aligned.pose.orientation = align_qnt_stamped.quaternion
    return object_world_aligned


def align_obj_ort(object_world_pose, listener):
    '''
    Align the object frame coordinates to be consistent relative to the world frame.
    '''
    obj_qnt_stamped_world = QuaternionStamped()
    obj_qnt_stamped_world.header.frame_id = object_world_pose.header.frame_id
    obj_qnt_stamped_world.quaternion = copy.deepcopy(object_world_pose.pose.orientation)

    #Convert from quarternion to matrix
    quarternion = [obj_qnt_stamped_world.quaternion.x, obj_qnt_stamped_world.quaternion.y,
                    obj_qnt_stamped_world.quaternion.z, obj_qnt_stamped_world.quaternion.w]
    trans_mat = tf.transformations.quaternion_matrix(quarternion)
    rot_mat = trans_mat[:3, :3]

    #Convert from matrix to quarternion
    # Adapt to the new robot arm pose: 
    # robot arm 1st joint zero angles faces the table.
    align_trans_matrix = np.identity(4)

    #Find and align x axes.
    x_axis = [1., 0., 0.]
    align_x_axis, min_ang_axis_idx = find_min_ang_vec(x_axis, rot_mat) 
    rot_mat = np.delete(rot_mat, min_ang_axis_idx, axis=1)
    align_trans_matrix[:3, 1] = align_x_axis

    #y axes
    y_axis = [0., 1., 0.]
    align_y_axis, min_ang_axis_idx = find_min_ang_vec(y_axis, rot_mat) 
    rot_mat = np.delete(rot_mat, min_ang_axis_idx, axis=1)
    align_trans_matrix[:3, 0] = -align_y_axis

    #z axes
    z_axis = [0., 0., 1.]
    align_z_axis, min_ang_axis_idx = find_min_ang_vec(z_axis, rot_mat) 
    align_trans_matrix[:3, 2] = align_z_axis

    align_qtn_array = tf.transformations.quaternion_from_matrix(align_trans_matrix)

    align_qnt_stamped = QuaternionStamped()
    align_qnt_stamped.header.frame_id = object_world_pose.header.frame_id
    align_qnt_stamped.quaternion.x, align_qnt_stamped.quaternion.y, \
            align_qnt_stamped.quaternion.z, align_qnt_stamped.quaternion.w = align_qtn_array

    return align_qnt_stamped

