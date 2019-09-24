#!/usr/bin/env python
import rospy
import tf
import numpy as np
from geometry_msgs.msg import Pose, Quaternion, PoseStamped, PointStamped
from prob_grasp_planner.srv import *

class BroadcastPalmTf:
    '''
    Broadcast the grasp palm pose in object tf for all grasps to analyze grasps.
    To compare the grasp poses in object frame, treat the world frame as the common object frame
    for all grasps.
    '''
    def __init__(self):
        rospy.init_node('grasp_palm_tf_in_obj_br')
        # Grasp pose in blensor camera frame.
        self.palm_poses_in_obj = []
        self.tf_prefix = ''
        self.prior_poses = []
        self.tf_br = tf.TransformBroadcaster()

    def broadcast_palm_tf(self):
        '''
        Broadcast the grasp palm tf in a common object frame.
        '''
        for i, palm_pose in enumerate(self.palm_poses_in_obj):
            self.tf_br.sendTransform((palm_pose.pose.position.x, palm_pose.pose.position.y, 
                    palm_pose.pose.position.z),
                    (palm_pose.pose.orientation.x, palm_pose.pose.orientation.y, 
                    palm_pose.pose.orientation.z, palm_pose.pose.orientation.w),
                    rospy.Time.now(), self.tf_prefix + str(i), 'world')

    def handle_update_palm_poses(self, req):
        '''
        Handler to update the palm pose tf.
        '''
        self.palm_poses_in_obj = req.palm_poses_in_obj
        self.tf_prefix = req.tf_prefix
        response = UpdatePalmPosesInObjResponse()
        response.success = True
        return response

    def update_palm_poses_server(self):
        '''
        Create the ROS server to update the palm tf.
        '''
        rospy.Service('update_grasp_palm_poses', UpdatePalmPosesInObj, self.handle_update_palm_poses) 
        rospy.loginfo('Service update_grasp_palm_poses:')
        rospy.loginfo('Ready to update grasp palm poses in object frame:')

    def broadcast_prior_tf(self):
        '''
        Broadcast the grasp palm tf in a common object frame.
        '''
        for i, prior_pose in enumerate(self.prior_poses):
            self.tf_br.sendTransform((prior_pose.pose.position.x, prior_pose.pose.position.y, 
                    prior_pose.pose.position.z),
                    (prior_pose.pose.orientation.x, prior_pose.pose.orientation.y, 
                    prior_pose.pose.orientation.z, prior_pose.pose.orientation.w),
                    #rospy.Time.now(), 'prior_' + str(i), 'world')
                    rospy.Time.now(), 'prior_' + str(i), 'blensor_camera')

    def handle_update_prior_poses(self, req):
        '''
        Handler to update the prior pose tf.
        '''
        self.prior_poses = req.prior_poses
        response = UpdatePriorPosesResponse()
        response.success = True
        return response

    def update_prior_poses_server(self):
        '''
        Create the ROS server to update the prior tf.
        '''
        rospy.Service('update_grasp_prior_poses', UpdatePriorPoses, self.handle_update_prior_poses) 
        rospy.loginfo('Service update_grasp_prior_poses:')
        #rospy.loginfo('Ready to update grasp palm poses in object frame:')
        rospy.loginfo('Ready to update grasp palm poses in blensor_camera frame:')


if __name__ == '__main__':
    broadcast_tf = BroadcastPalmTf() 
    broadcast_tf.update_palm_poses_server()
    broadcast_tf.update_prior_poses_server()
    while not rospy.is_shutdown():
        broadcast_tf.broadcast_palm_tf()
        broadcast_tf.broadcast_prior_tf()

        rospy.sleep(1)
