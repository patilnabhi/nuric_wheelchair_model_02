#!/usr/bin/env python
import rospy
import sys
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy

class JoyTeleop:

	def __init__(self):
		rospy.init_node('joy_teleop')
		self.joy_cmd = Twist()
		#Setup subscriber
		self.joy = rospy.Subscriber('joy', Joy, self.joy_callback)

		# Setup publisher
		self.pub_twist = rospy.Publisher('/cmd_vel', Twist, queue_size=20)

		rate = rospy.Rate(20)
		while not rospy.is_shutdown():
			self.pub()
			rate.sleep()

	def pub(self):
		self.pub_twist.publish(self.joy_cmd)

	def joy_callback(self, joy):

		self.joy_cmd.linear.x = 0.5 * joy.axes[1]
		self.joy_cmd.angular.z = 0.5 * joy.axes[0]


def main():
	joy_teleop_node = JoyTeleop()
	try:
		rospy.spin()
	except rospy.ROSInterruptException:
		pass

if __name__ == '__main__':
	sys.exit(main())
