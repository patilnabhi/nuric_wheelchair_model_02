#!/usr/bin/env python
import rospy
import sys
from sensor_msgs.msg import JointState
from nuric_wheelchair_model_02.msg import FloatArray

from math import atan2, sin, cos

class GetCasterJoints:

    def __init__(self):
        rospy.init_node('get_caster_joints')

        # Set rospy to execute a shutdown function when exiting       
        rospy.on_shutdown(self.shutdown)

        self.caster_joints = FloatArray()

        #Setup subscriber
        self.joints = rospy.Subscriber('/joint_states', JointState, self.joints_callback)

        # Setup publisher
        self.pub_caster_joints = rospy.Publisher('/caster_joints', FloatArray, queue_size=20)

        r = rospy.Rate(20)
        while not rospy.is_shutdown():
            self.pub()
            r.sleep()

    def joints_callback(self, joints):
        self.caster_joints = [atan2(sin(joints.position[0]), cos(joints.position[0])), atan2(sin(joints.position[1]), cos(joints.position[1]))]
        # print self.caster_joints

    def pub(self):
        self.pub_caster_joints.publish(self.caster_joints)

  
    def shutdown(self):
        # shutting down the node.
        rospy.loginfo("Shutting down node...")
        
        rospy.sleep(1)


if __name__ == '__main__':
    
    try:
        GetCasterJoints()
        # rospy.spin()
    except rospy.ROSInterruptException:
        pass
