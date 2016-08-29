#!/usr/bin/env python
import rospy
import sys
from sensor_msgs.msg import JointState
from nuric_wheelchair_model_02.msg import FloatArray

from math import atan2, sin, cos

class CasterOrient:

    def __init__(self):
        rospy.init_node('caster_orient')

        self.caster_orient = FloatArray()

        #Setup subscriber
        self.joints = rospy.Subscriber('/joint_states', JointState, self.joints_callback)

        # Setup publisher
        self.pub_caster_orient = rospy.Publisher('/caster_orient', FloatArray, queue_size=20)

        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            self.pub()
            rate.sleep()

    def joints_callback(self, joints):
        self.caster_orient = [atan2(sin(joints.position[0]), cos(joints.position[0])), atan2(sin(joints.position[1]), cos(joints.position[1]))]

    def pub(self):
        self.pub_caster_orient.publish(self.caster_orient)


def main():
    caster_orient_node = CasterOrient()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    sys.exit(main())
