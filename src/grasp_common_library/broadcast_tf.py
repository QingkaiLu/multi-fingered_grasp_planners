#!/usr/bin/env python
import rospy
import tf
import numpy as np
from geometry_msgs.msg import Pose, Quaternion, PoseStamped, PointStamped
from prob_grasp_planner.srv import *

class BroadcastTf:
    '''
    The class to 1. create ROS services to update the grasp palm tf;
    2. broadcast the updated grasp palm tf; 3. update and broadcast the object tf.
    '''
    def __init__(self):
        rospy.init_node('grasp_palm_tf_br')
        # Grasp pose in blensor camera frame.
        self.palm_pose = None
        self.object_pose_world = None
        self.tf_br = tf.TransformBroadcaster()

    def broadcast_tf(self):
        '''
        Broadcast the grasp palm tf.
        '''
        if self.palm_pose is not None:
            self.tf_br.sendTransform((self.palm_pose.pose.position.x, self.palm_pose.pose.position.y, 
                    self.palm_pose.pose.position.z),
                    (self.palm_pose.pose.orientation.x, self.palm_pose.pose.orientation.y, 
                    self.palm_pose.pose.orientation.z, self.palm_pose.pose.orientation.w),
                    rospy.Time.now(), 'grasp_palm_pose', self.palm_pose.header.frame_id)

        if self.object_pose_world is not None:
            self.tf_br.sendTransform((self.object_pose_world.pose.position.x, self.object_pose_world.pose.position.y, 
                    self.object_pose_world.pose.position.z),
                    (self.object_pose_world.pose.orientation.x, self.object_pose_world.pose.orientation.y, 
                    self.object_pose_world.pose.orientation.z, self.object_pose_world.pose.orientation.w),
                    rospy.Time.now(), 'object_pose', self.object_pose_world.header.frame_id)


    def handle_update_palm_pose(self, req):
        '''
        Handler to update the palm pose tf.
        '''
        self.palm_pose = req.palm_pose
        response = UpdatePalmPoseResponse()
        response.success = True
        return response

    def update_palm_pose_server(self):
        '''
        Create the ROS server to update the palm tf.
        '''
        rospy.Service('update_grasp_palm_pose', UpdatePalmPose, self.handle_update_palm_pose) 
        rospy.loginfo('Service update_grasp_palm_pose:')
        rospy.loginfo('Ready to update grasp palm pose:')

    def handle_update_object_pose(self, req):
        '''
        Handler to update the object pose tf.
        '''
        self.object_pose_world = req.object_pose_world
        response = UpdateObjectPoseResponse()
        response.success = True
        return response

    def update_object_pose_server(self):
        '''
        Create the ROS server to update the object tf.
        '''
        rospy.Service('update_grasp_object_pose', UpdateObjectPose, self.handle_update_object_pose) 
        rospy.loginfo('Service update_grasp_object_pose:')
        rospy.loginfo('Ready to update grasp object pose:')

if __name__ == '__main__':
    broadcast_tf = BroadcastTf() 
    broadcast_tf.update_palm_pose_server()
    broadcast_tf.update_object_pose_server()
    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        broadcast_tf.broadcast_tf()
        rate.sleep()


