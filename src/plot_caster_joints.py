#!/usr/bin/env python
import rospy
import sys
from nuric_wheelchair_model_02.msg import FloatArray
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist
import random
import matplotlib.pyplot as plt
from math import sin, cos, atan2
import numpy as np
from scipy.integrate import odeint

class PlotCasterJoints:

    def __init__(self):
        rospy.init_node('plot_caster_joints')

        # Set rospy to execute a shutdown function when exiting       
        rospy.on_shutdown(self.shutdown)

        self.l_caster_angle = Float64()
        self.r_caster_angle = Float64()
        self.wheel_cmd = Twist()

        self.pi = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348

        # time to move wheelchair 
        self.move_time = 4.0

        self.rate = 100

        self.save = 0

        self.l_caster_data = []
        self.r_caster_data = []

        # wheelchair constants
        self.wh_consts = [0.58, 0.19, 0.06]

        # Setup subscriber
        self.caster_joints = rospy.Subscriber('/caster_joints', FloatArray, self.caster_joints_callback)

        # Setup publisher
        self.pub_twist = rospy.Publisher('/cmd_vel', Twist, queue_size=20)

        self.r = rospy.Rate(self.rate)

        # Initialize (wait for caster joints to be subscribed)
        self.initialize()

        # Move wheelchair
        self.move_wheelchair()

        # print len(self.l_caster_data)

        # Plot caster joints data
        self.plot_data()

        

        
    def caster_joints_callback(self, caster_joints):
        if self.save:
            self.l_caster_data.append(caster_joints.data[0])
            self.r_caster_data.append(caster_joints.data[1])
        else:
            self.l_caster_angle = caster_joints.data[0]
            self.r_caster_angle = caster_joints.data[1]

    def initialize(self):
        rospy.loginfo("Initializing...")        
        rospy.sleep(1)
        self.print_caster_joints()

    def move_wheelchair(self):
        self.wheel_cmd.linear.x = -0.3
        self.wheel_cmd.angular.z = 0.0

        while rospy.get_time() == 0.0:
            continue
        start = rospy.get_time()

        rospy.loginfo("Moving robot...")
        while (rospy.get_time() - start < self.move_time) and not rospy.is_shutdown():
            self.save = 1
            self.pub_twist.publish(self.wheel_cmd)        
            # self.r.sleep()

        # Stop the robot
        self.pub_twist.publish(Twist())
        self.save = 0
        self.print_caster_joints()
        
        rospy.sleep(1)
        

    def save_data(self):
        self.l_caster_data.append(self.l_caster_angle)
        self.r_caster_data.append(self.r_caster_angle)

    def plot_data(self):
        plt.figure(1)
        plt.subplot(211)
        plt.title("Left caster orientations")
        plt.plot(self.l_caster_data, label="sim")

        self.ode_int(0)
        plt.plot(self.sol, label="est")
        plt.legend()

        # plt.subplot(412)
        # data_temp = []
        # for i in range(len(self.l_caster_data)-1):
        #     data_temp.append(self.l_caster_data[i] - self.sol[i])
        # plt.plot(data_temp, label="error left")
        # plt.legend()

        plt.subplot(212)
        plt.title("Right caster orientations")
        plt.plot(self.r_caster_data, label="sim")
        self.ode_int(1)
        plt.plot(self.sol, label="est")

        # plt.subplot(414)
        # data_temp = []
        # for i in range(len(self.r_caster_data)-1):
        #     data_temp.append(self.r_caster_data[i] - self.sol[i])
        # plt.plot(data_temp, label="error right")
        # plt.legend()

        plt.show()

    def solvr_left(self, Y, t):
        th = self.wheel_cmd.angular.z
        x = self.wheel_cmd.linear.x
        dl = self.wh_consts[0]
        df = self.wh_consts[1]
        dc = self.wh_consts[2]
        return [(th*(dl*cos(Y[0]) + (df*sin(Y[0])/2) - dc)/dc) + (x*sin(Y[0])/dc)]

    def solvr_right(self, Y, t):
        th = self.wheel_cmd.angular.z
        x = self.wheel_cmd.linear.x
        dl = self.wh_consts[0]
        df = self.wh_consts[1]
        dc = self.wh_consts[2]
        return [(th*(dl*cos(Y[0]) - (df*sin(Y[0])/2) - dc)/dc) + (x*sin(Y[0])/dc)]

    def ode_int(self, lr):
        a_t = np.arange(0.0, self.move_time, 1./self.rate)
        if lr == 0:
            ini_val = self.angle_adj(self.l_caster_data[0]+self.pi)
            # if abs(ini_val) < 0.0005:
            #     ini_val = 0.0
            print ini_val
            asol = odeint(self.solvr_left, [ini_val], a_t)
            # self.sol = [(self.angle_adj(item+self.pi)) for sublist in asol.tolist() for item in sublist]
            self.sol = [self.angle_adj((item+self.pi)%(2.*self.pi)) for sublist in asol.tolist() for item in sublist]
        elif lr == 1:
            ini_val = self.angle_adj(self.r_caster_data[0]+self.pi)
            # if abs(ini_val) < 0.0005:
            #     ini_val = 0.0
            print ini_val
            asol = odeint(self.solvr_right, [ini_val], a_t)
            # self.sol = [(self.angle_adj(item+self.pi)) for sublist in asol.tolist() for item in sublist]
            self.sol = [self.angle_adj((item+self.pi)%(2.*self.pi)) for sublist in asol.tolist() for item in sublist]


    def angle_adj(self, angle):
        if angle > self.pi:
            return -(2.*self.pi - angle)
        return angle

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