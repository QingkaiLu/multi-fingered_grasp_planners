#!/usr/bin/env python
import rospy
import tf
from sensor_msgs.msg import JointState
import numpy as np

def broadcast_tf(tf_br):
    tf_br.sendTransform((0, 0, 0.2), (0, 0, 0, 1), 
            rospy.Time.now(), 'allegro_mount', 'world')


if __name__ == '__main__':
    rospy.init_node('broadcast_hand_tf')
    tf_br = tf.TransformBroadcaster()
    pub = rospy.Publisher('/allegro_hand_right/joint_states', JointState, queue_size=1)
    jc = JointState()
    jc.name = ['index_joint_0','index_joint_1','index_joint_2', 'index_joint_3',
                'middle_joint_0','middle_joint_1','middle_joint_2', 'middle_joint_3',
                'ring_joint_0','ring_joint_1','ring_joint_2', 'ring_joint_3',
                'thumb_joint_0','thumb_joint_1','thumb_joint_2', 'thumb_joint_3']
    jc.position = 0.3 * np.ones(16)

    while not rospy.is_shutdown():
        broadcast_tf(tf_br)
        pub.publish(jc)
        rospy.sleep(1)


