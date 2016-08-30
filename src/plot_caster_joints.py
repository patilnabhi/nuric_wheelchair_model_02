#!/usr/bin/env python
import rospy
import sys
from nuric_wheelchair_model_02.msg import FloatArray
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
import random
import matplotlib.pyplot as plt

class PlotCasterJoints:

    def __init__(self):
        rospy.init_node('plot_caster_joints')

        # Set rospy to execute a shutdown function when exiting       
        rospy.on_shutdown(self.shutdown)

        self.l_caster_angle = Float32()
        self.r_caster_angle = Float32()
        self.wheel_cmd = Twist()

        self.move_time = 3.0

        self.l_caster_data = []
        self.r_caster_data = []

        # Setup subscriber
        self.caster_joints = rospy.Subscriber('/caster_joints', FloatArray, self.caster_joints_callback)

        # Setup publisher
        self.pub_twist = rospy.Publisher('/cmd_vel', Twist, queue_size=20)

        self.r = rospy.Rate(100)

        # Initial movement to orient caster wheels randomly
        self.move_wheelchair_init()

        # Move wheelchair straight
        self.move_wheelchair()

        # Plot simulated caster joints data
        self.plot_sim_data()

        # Plot estimated caster joints data
        # self.plot_est_data()
        
    def caster_joints_callback(self, caster_joints):
        self.l_caster_angle, self.r_caster_angle = caster_joints.data[0], caster_joints.data[1]
        

    def move_wheelchair_init(self):
        self.wheel_cmd.linear.x = random.uniform(-0.8, 0.8)
        self.wheel_cmd.angular.z = random.uniform(-0.5, 0.5)

        while rospy.get_time() == 0.0:
            continue
        start = rospy.get_time()

        rospy.loginfo("Initial random movement...")
        while (rospy.get_time() - start < random.uniform(4.0, 6.0)) and not rospy.is_shutdown():
            self.pub_twist.publish(self.wheel_cmd)
            
            self.r.sleep()

        # Stop the robot
        self.pub_twist.publish(Twist())
        
        rospy.sleep(2)
        self.print_caster_joints()

    def move_wheelchair(self):
        self.wheel_cmd.linear.x = 0.5
        self.wheel_cmd.angular.z = 0.0

        start = rospy.get_time()

        rospy.loginfo("Second movement...")
        while (rospy.get_time() - start < self.move_time) and not rospy.is_shutdown():
            self.save_data()
            self.pub_twist.publish(self.wheel_cmd)        
            self.r.sleep()

        # Stop the robot
        self.pub_twist.publish(Twist())
        self.print_caster_joints()
        

    def save_data(self):
        self.l_caster_data.append(self.l_caster_angle)
        self.r_caster_data.append(self.r_caster_angle)

    def plot_sim_data(self):
        plt.plot(self.l_caster_data)
        plt.show()

    # def plot_est_data(self):

    def print_caster_joints(self):
        print 'Left caster: {:.3f} rad'.format(self.l_caster_angle)
        print 'Right caster: {:.3f} rad'.format(self.r_caster_angle)

    def shutdown(self):
        # Stop the robot when shutting down the node.
        rospy.loginfo("Stopping the robot...")
        self.pub_twist.publish(Twist())
        rospy.sleep(1)


if __name__ == '__main__':
    
    try:
        PlotCasterJoints()
    except rospy.ROSInterruptException:
        pass