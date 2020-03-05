#!/usr/bin/env python


from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_kinematics import KDLKinematics
import numpy as np
import rospy
from trac_ik_python.trac_ik import IK


class GraspKDL():
    '''
    Grasp KDL class.
    '''

    def __init__(self):
        # rospy.init_node('grasp_kdl')
        self.robot = URDF.from_parameter_server()
        base_link = 'world'
        end_link = 'palm_link'
        self.kdl_kin = KDLKinematics(self.robot, base_link, end_link)
        self.ik_solver = IK(base_link, end_link)
        self.seed_state = [0.0] * self.ik_solver.number_of_joints


    def forward(self, q):
        pose = self.kdl_kin.forward(q)
        return pose


    def jacobian(self, q):
        J = self.kdl_kin.jacobian(q)
        return J


    def inverse(self, pose):
        ik_js = self.ik_solver.get_ik(self.seed_state, pose.position.x, 
                                        pose.position.y, pose.position.z,
                                        pose.orientation.x, pose.orientation.y, 
                                        pose.orientation.z, pose.orientation.w)
        if ik_js is None:
            rospy.logerr('No IK solution for grasp planning!')
            return None
        return ik_js

