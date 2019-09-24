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


class ConfigConvertFunctions():


    def __init__(self):
        # 3 dof location of the palm + 3 dof orientation of the palm (4 parameters for quaternion, 
        # 3 for Euler angles) + 1st two joints of the thumb + 1st joint of other three fingers
        # Other joint angles are fixed for grasp preshape inference.
        self.palm_loc_dof_dim = 3
        self.palm_dof_dim = 6


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

