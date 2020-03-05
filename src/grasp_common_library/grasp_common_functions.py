import numpy as np
from geometry_msgs.msg import Pose, Quaternion, PoseStamped, PointStamped
from prob_grasp_planner.msg import VisualInfo, HandConfig
from prob_grasp_planner.srv import *
import tf
import rospy
from prob_grasp_planner.srv import *
from prob_grasp_planner.msg import *
from point_cloud_segmentation.srv import *
import roslib.packages as rp
pkg_path = rp.get_pkg_dir('prob_grasp_planner')
sys.path.append(pkg_path + '/src/grasp_common_library')
import align_object_frame as align_obj
from data_proc_lib import DataProcLib 
from grasp_kdl import GraspKDL 
import PyKDL


class ConfigConvertFunctions():


    def __init__(self, data_proc_lib=None):
        # 3 dof location of the palm + 3 dof orientation of the palm (4 parameters for quaternion, 
        # 3 for Euler angles) + 1st two joints of the thumb + 1st joint of other three fingers
        # Other joint angles are fixed for grasp preshape inference.
        self.palm_loc_dof_dim = 3
        self.palm_dof_dim = 6
        self.arm_joints_dof = 7
        self.grasp_kin = GraspKDL()
        self.dpl = data_proc_lib


    def convert_array_to_pose(self, pose_array, frame_id):
        '''
        Convert pose Quaternion array to ROS PoseStamped.
    
        Args:
            pose_array
    
        Returns:
            ROS pose.
        '''
        pose_stamp = PoseStamped()
        pose_stamp.header.frame_id = frame_id
        pose_stamp.pose.position.x, pose_stamp.pose.position.y, \
                pose_stamp.pose.position.z = pose_array[:self.palm_loc_dof_dim]    
    
        pose_stamp.pose.orientation.x, pose_stamp.pose.orientation.y, \
            pose_stamp.pose.orientation.z, pose_stamp.pose.orientation.w = \
                                            pose_array[self.palm_loc_dof_dim:]    
    
        return pose_stamp
    

    def convert_preshape_to_full_config(self, preshape_config, config_frame_id):
        '''
        Convert preshape grasp configuration to full grasp configuration by filling zeros for 
        uninferred finger joints.
        '''
        hand_config = HandConfig()
        hand_config.palm_pose.header.frame_id = config_frame_id
        hand_config.palm_pose.pose.position.x, hand_config.palm_pose.pose.position.y, \
                hand_config.palm_pose.pose.position.z = preshape_config[:self.palm_loc_dof_dim]    
    
        palm_euler = preshape_config[self.palm_loc_dof_dim:self.palm_dof_dim] 
        palm_quaternion = tf.transformations.quaternion_from_euler(
                                            palm_euler[0], palm_euler[1], palm_euler[2])
        #hand_config.palm_pose.pose.orientation = palm_quaternion
        hand_config.palm_pose.pose.orientation.x, hand_config.palm_pose.pose.orientation.y, \
                hand_config.palm_pose.pose.orientation.z, \
                hand_config.palm_pose.pose.orientation.w = palm_quaternion 
    
        hand_config.hand_joint_state.name = \
                    ['index_joint_0','index_joint_1','index_joint_2', 'index_joint_3',
                   'middle_joint_0','middle_joint_1','middle_joint_2', 'middle_joint_3',
                   'ring_joint_0','ring_joint_1','ring_joint_2', 'ring_joint_3',
                   'thumb_joint_0','thumb_joint_1','thumb_joint_2', 'thumb_joint_3']
        hand_config.hand_joint_state.position = [preshape_config[self.palm_dof_dim], 
                                                 preshape_config[self.palm_dof_dim + 1], 0., 0.,
                                                 preshape_config[self.palm_dof_dim + 2], 
                                                 preshape_config[self.palm_dof_dim + 3], 0., 0.,
                                                 preshape_config[self.palm_dof_dim + 4], 
                                                 preshape_config[self.palm_dof_dim + 5], 0., 0.,
                                                 preshape_config[self.palm_dof_dim + 6], 
                                                 preshape_config[self.palm_dof_dim + 7], 0., 0.]
    
        return hand_config
    
    
    def convert_full_to_preshape_config(self, hand_config):
       '''
       Convert full grasp configuration to preshape grasp configuration by deleting uninferred joint
       angles.
       '''
       palm_quaternion = (hand_config.palm_pose.pose.orientation.x, 
                           hand_config.palm_pose.pose.orientation.y,
                           hand_config.palm_pose.pose.orientation.z, 
                           hand_config.palm_pose.pose.orientation.w) 
       palm_euler = tf.transformations.euler_from_quaternion(palm_quaternion)
   
       preshape_config = [hand_config.palm_pose.pose.position.x, hand_config.palm_pose.pose.position.y,
               hand_config.palm_pose.pose.position.z, palm_euler[0], palm_euler[1], palm_euler[2],
               hand_config.hand_joint_state.position[0], hand_config.hand_joint_state.position[1],
               hand_config.hand_joint_state.position[4], hand_config.hand_joint_state.position[5],
               hand_config.hand_joint_state.position[8], hand_config.hand_joint_state.position[9],
               hand_config.hand_joint_state.position[12], hand_config.hand_joint_state.position[13]]
   
       return np.array(preshape_config)


    def convert_to_ik_config_tf(self, config):
        palm_pose_obj = Pose()
        palm_pose_obj.position.x, palm_pose_obj.position.y, \
                palm_pose_obj.position.z = config[:self.palm_loc_dof_dim]    
    
        palm_euler = config[self.palm_loc_dof_dim:
                                        self.palm_dof_dim] 
        palm_quaternion = tf.transformations.quaternion_from_euler(
                                palm_euler[0], palm_euler[1], palm_euler[2])
        palm_pose_obj.orientation.x, palm_pose_obj.orientation.y, \
                palm_pose_obj.orientation.z, \
                palm_pose_obj.orientation.w = palm_quaternion 

        # Transform the pose from object frame to world frame
        palm_pose_stamp = PoseStamped() 
        palm_pose_stamp.header.frame_id = self.dpl.object_frame_id
        palm_pose_stamp.pose = palm_pose_obj
        palm_pose_world = self.dpl.trans_pose_to_world_tf(palm_pose_stamp, 
                                                    return_array=False)

        arm_js = self.grasp_kin.inverse(palm_pose_world.pose)
        
        ik_config = None
        if arm_js is not None:
            ik_config = np.concatenate((arm_js, 
                            config[self.palm_dof_dim:]))

        return ik_config


    def convert_to_ik_config(self, config, mat_obj_to_world):
        config_trans = config[:self.palm_loc_dof_dim]    
        config_euler = config[self.palm_loc_dof_dim:self.palm_dof_dim] 
        config_trans_mat = tf.transformations.translation_matrix(config_trans)
        config_rot_mat = tf.transformations.euler_matrix(config_euler[0], 
                                        config_euler[1], config_euler[2])
        mat_palm_to_obj = np.copy(config_trans_mat)
        mat_palm_to_obj[:3, :3] = config_rot_mat[:3, :3]

        mat_palm_to_world = np.matmul(mat_obj_to_world, mat_palm_to_obj)

        trans_palm_to_world = tf.transformations.translation_from_matrix(mat_palm_to_world)
        quat_palm_to_world = tf.transformations.quaternion_from_matrix(mat_palm_to_world)

        palm_pose_world = Pose()
        palm_pose_world.position.x, palm_pose_world.position.y, \
                palm_pose_world.position.z = trans_palm_to_world 
        palm_pose_world.orientation.x, palm_pose_world.orientation.y, \
                palm_pose_world.orientation.z, \
                palm_pose_world.orientation.w = quat_palm_to_world 
        
        arm_js = self.grasp_kin.inverse(palm_pose_world)
        
        ik_config = None
        if arm_js is not None:
            ik_config = np.concatenate((arm_js, 
                            config[self.palm_dof_dim:]))

        return ik_config

    
    def convert_to_fk_config_tf(self, ik_config):
        palm_pose_mat = self.grasp_kin.forward(ik_config[:self.arm_joints_dof])

        # Transform the pose from world frame to object frame
        palm_pose_stamp = PoseStamped() 
        palm_pose_stamp.header.frame_id = self.dpl.world_frame_id
        palm_pose_stamp.pose.position.x, palm_pose_stamp.pose.position.y, \
                palm_pose_stamp.pose.position.z = tf.transformations.translation_from_matrix(palm_pose_mat)
        palm_pose_stamp.pose.orientation.x, palm_pose_stamp.pose.orientation.y, \
                palm_pose_stamp.pose.orientation.z, palm_pose_stamp.pose.orientation.w \
                                            = tf.transformations.quaternion_from_matrix(palm_pose_mat)
        pose_obj_array = self.dpl.trans_pose_to_obj_tf(palm_pose_stamp)

        fk_config = np.concatenate((pose_obj_array, ik_config[self.arm_joints_dof:]))

        return fk_config


    def convert_to_fk_config(self, ik_config, mat_world_to_obj):
        mat_palm_to_world = self.grasp_kin.forward(ik_config[:self.arm_joints_dof])

        mat_palm_to_obj = np.matmul(mat_world_to_obj, mat_palm_to_world)

        palm_obj_trans = tf.transformations.translation_from_matrix(mat_palm_to_obj) 
        palm_obj_euler = tf.transformations.euler_from_matrix(mat_palm_to_obj)

        pose_obj_array = np.concatenate((palm_obj_trans, palm_obj_euler)) 
        fk_config = np.concatenate((pose_obj_array, ik_config[self.arm_joints_dof:]))

        return fk_config


    def compute_num_jac(self, func, ik_config, mat_world_to_obj):
        eps = 10**-6
        jac = np.zeros((self.palm_dof_dim, self.arm_joints_dof)) 
        
        for i in xrange(self.arm_joints_dof):
            ik_config_plus = np.copy(ik_config)
            ik_config_plus[i] += eps
            config_plus = func(ik_config_plus, mat_world_to_obj)

            ik_config_minus = np.copy(ik_config)
            ik_config_minus[i] -= eps
            config_minus = func(ik_config_minus, mat_world_to_obj)

            # ith_grad = (config_plus[:6] - config_minus[:6]) / (2. * eps)

            trans_plus = PyKDL.Vector(config_plus[0], 
                        config_plus[1], config_plus[2])
            rot_plus = PyKDL.Rotation.RPY(config_plus[3], 
                        config_plus[4], config_plus[5])
            frame_plus = PyKDL.Frame(rot_plus, trans_plus)

            trans_minus = PyKDL.Vector(config_minus[0], 
                        config_minus[1], config_minus[2])
            rot_minus = PyKDL.Rotation.RPY(config_minus[3], 
                        config_minus[4], config_minus[5])
            frame_minus = PyKDL.Frame(rot_minus, trans_minus)

            twist = PyKDL.diff(frame_minus, frame_plus, 1.0)
            ith_grad = np.array([twist.vel[0], twist.vel[1], twist.vel[2],
                                twist.rot[0], twist.rot[1], twist.rot[2]]) / (2. * eps)
            jac[:, i] = ith_grad

        return jac


def seg_obj_from_file_client(pcd_file_path, 
                            listener, align_obj_frame=True):
    rospy.loginfo('Waiting for service object_segmenter.')
    rospy.wait_for_service('object_segmenter')
    rospy.loginfo('Calling service object_segmenter.')
    try:
        object_segment_proxy = rospy.ServiceProxy('object_segmenter', 
                                                    SegmentGraspObject)
        object_segment_request = SegmentGraspObjectRequest()
        object_segment_request.client_cloud_path = pcd_file_path
        object_segment_response = object_segment_proxy(object_segment_request) 
        if align_obj_frame:
            object_segment_response.obj = \
                    align_obj.align_object(object_segment_response.obj, listener)
    except rospy.ServiceException, e:
        rospy.loginfo('Service object_segmenter call failed: %s'%e)
    rospy.loginfo('Service object_segmenter is executed.')
    if not object_segment_response.object_found:
        rospy.logerr('No object found from segmentation!')
        # return None
    return object_segment_response


def segment_object_client(listener, align_obj_frame=True):
    rospy.loginfo('Waiting for service object_segmenter.')
    rospy.wait_for_service('object_segmenter')
    rospy.loginfo('Calling service object_segmenter.')
    try:
        object_segment_proxy = rospy.ServiceProxy('object_segmenter', 
                                                    SegmentGraspObject)
        object_segment_request = SegmentGraspObjectRequest()
        object_segment_response = object_segment_proxy(object_segment_request) 
        if align_obj_frame:
            object_segment_response.obj = \
                    align_obj.align_object(object_segment_response.obj, listener)
    except rospy.ServiceException, e:
        rospy.loginfo('Service object_segmenter call failed: %s'%e)
    rospy.loginfo('Service object_segmenter is executed.')
    if not object_segment_response.object_found:
        rospy.logerr('No object found from segmentation!')
        # return False
    return object_segment_response

