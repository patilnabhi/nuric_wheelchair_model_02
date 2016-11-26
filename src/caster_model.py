#!/usr/bin/env python
import rospy
import sys
from nuric_wheelchair_model_02.msg import FloatArray
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import random
import matplotlib.pyplot as plt
from math import sin, cos, atan2
import numpy as np
from scipy.integrate import odeint, ode
from tf.transformations import euler_from_quaternion
from ukf_helper import normalize_angle

class SolveCasterModel:
    def __init__(self):
        rospy.init_node('solve_caster_model')

        rospy.on_shutdown(self.shutdown)

        self.wheel_cmd = Twist()

        self.wheel_cmd.linear.x = 0.4 
        self.wheel_cmd.angular.z = 0.001


        self.move_time = 10.0
        self.rate = 50

        self.l_caster_data = []
        self.r_caster_data = []

        self.asol = []

        self.sol_alpha1 = []
        self.sol_alpha2 = []

        self.error_alpha1 = []
        self.error_alpha2 = []

        self.angle_adj = normalize_angle

        # wheelchair constants
        self.wh_consts = [0.58, 0.19, 0.06]

        # self.actual_pose = rospy.Subscriber('/odom', Odometry, self.actual_pose_callback)

        self.caster_joints = rospy.Subscriber('/caster_joints', FloatArray, self.caster_joints_callback)
        self.pub_twist = rospy.Publisher('/cmd_vel', Twist, queue_size=20)

        self.r = rospy.Rate(self.rate)

        self.move_wheelchair()

        self.plot_data()

    def caster_joints_callback(self, caster_joints):       
        self.l_caster_angle, self.r_caster_angle = caster_joints.data[0], caster_joints.data[1]

    def move_wheelchair(self):
        
        while rospy.get_time() == 0.0:
            continue
        start = rospy.get_time()

        rospy.sleep(1)

        rospy.loginfo("Moving robot...")
        
        while (rospy.get_time() - start < self.move_time) and not rospy.is_shutdown():

            self.pub_twist.publish(self.wheel_cmd)    

            self.l_caster_data.append(self.l_caster_angle)
            self.r_caster_data.append(self.r_caster_angle)
            
            self.r.sleep()


        # Stop the robot
        self.pub_twist.publish(Twist())
        rospy.sleep(1)

    def solvr(self, x, t):

        dl = self.wh_consts[0]
        df = self.wh_consts[1]
        dc = self.wh_consts[2]

        vx = self.wheel_cmd.linear.x
        vth = self.wheel_cmd.angular.z

        eq1 = (vth*(dl*cos(x[0]) - (df*sin(x[0])/2) - dc)/dc) + (vx*sin(x[0])/dc)
        eq2 = (vth*(dl*cos(x[1]) + (df*sin(x[1])/2) - dc)/dc) + (vx*sin(x[1])/dc)

        return [eq1, eq2]


    def ode_int(self):

        a_t = np.arange(0.0, self.move_time, 1./self.rate)
        ini_val = [normalize_angle(self.r_caster_angle-np.pi), normalize_angle(self.l_caster_angle-np.pi)]
        
        asol = odeint(self.solvr, ini_val, a_t)
        self.asol = asol
    
    def calc_error(self):

        for i in xrange(min(len(self.sol_alpha1), len(self.r_caster_data))):
            self.error_alpha1.append(self.sol_alpha1[i]-self.r_caster_data[i])

        for i in xrange(min(len(self.sol_alpha2), len(self.l_caster_data))):
            self.error_alpha2.append(self.sol_alpha2[i]-self.l_caster_data[i])

    def plot_data(self):

        self.ode_int()

        for i in xrange(len(self.asol)):
            self.sol_alpha1.append(normalize_angle(self.asol[i][0]))
            self.sol_alpha2.append(normalize_angle(self.asol[i][1]))


        self.sol_alpha1 = [normalize_angle((angle+np.pi)) for angle in self.sol_alpha1]
        self.sol_alpha2 = [normalize_angle((angle+np.pi)) for angle in self.sol_alpha2]

        self.calc_error()


        plt.figure(1)

        plt.subplot(221)
        plt.title("R Caster Orientation (rad)")
        xaxis = [x/100. for x in xrange(len(self.sol_alpha1))]
        plt.plot(xaxis, self.sol_alpha1, label="est")
        xaxis = [x/100. for x in xrange(len(self.r_caster_data))]
        plt.plot(xaxis, self.r_caster_data, label="actual")
        plt.legend()

        plt.subplot(222)
        plt.title("L Caster Orientation (rad)")
        xaxis = [x/100. for x in xrange(len(self.sol_alpha2))]
        plt.plot(xaxis, self.sol_alpha2, label="est")
        xaxis = [x/100. for x in xrange(len(self.l_caster_data))]
        plt.plot(xaxis, self.l_caster_data, label="actual")
        plt.legend()

        plt.subplot(223)
        plt.title("Error R Caster Orientation (rad)")
        xaxis = [x/100. for x in xrange(len(self.error_alpha1))]
        plt.plot(xaxis, self.error_alpha1)

        plt.subplot(224)
        plt.title("Error L Caster Orientation (rad)")
        xaxis = [x/100. for x in xrange(len(self.error_alpha2))]
        plt.plot(xaxis, self.error_alpha2)

        plt.show()


    def shutdown(self):
        # Stop the robot when shutting down the node.
        rospy.loginfo("Stopping the robot...")
        self.pub_twist.publish(Twist())
        rospy.sleep(1)

if __name__ == '__main__':

    try:
        SolveCasterModel()
    except rospy.ROSInterruptException:
        pass