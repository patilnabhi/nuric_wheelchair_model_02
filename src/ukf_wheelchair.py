#!/usr/bin/env python

import rospy
import sys
from ukf_helper import MerweScaledSigmaPoints, state_mean, meas_mean, residual_x, residual_z
from ukf import UKF
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion
import numpy as np 
from scipy.integrate import odeint

class UKFWheelchair(object):

    def __init__(self):

        rospy.init_node('ukf_wheelchair')

        rospy.on_shutdown(self.shutdown)

        self.wheel_cmd = Twist()
        self.wheel_cmd.linear.x = 0.0
        self.wheel_cmd.angular.z = 0.0

        self._move_time = 5.0
        self._rate = 1000

        # constants for ode equations
        # (approximations)
        self.Iz = 15.0
        self.mu = .01
        self.ep = 0.2
        self.m = 5.0
        self.g = 9.81/50.
        self.pi = np.pi
        self.two_pi = 2.*self.pi

        # wheelchair constants
        self.wh_consts = [0.58, 0.19, 0.06]


        self.ini_val = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.odom_data = rospy.Subscriber('/odom', Odometry, self.odom_cb)

        self.pub_twist = rospy.Publisher('/cmd_vel', Twist, queue_size=20)

        self.r = rospy.Rate(self._rate)

        self.run_ukf_wheelchair()



    def odom_cb(self, odom_data):

        (_,_,yaw) = euler_from_quaternion([odom_data.pose.pose.orientation.x, odom_data.pose.pose.orientation.y, odom_data.pose.pose.orientation.z, odom_data.pose.pose.orientation.w])

        self.odom_x, self.odom_y, self.odom_th = odom_data.pose.pose.position.x, odom_data.pose.pose.position.y, yaw


    def run_ukf_wheelchair(self):

        def fx(x, dt):
            # for i in xrange(len(x)):
            x[4], x[5], x[6] = self.angle_adj(x[4]), self.angle_adj(x[5]), self.angle_adj(x[6])
            self.ode_solve(x, dt)
            return np.array(self.asol)


        def hx(x):
            return np.array([x[3], x[2], x[4]])

        dt = 1./self._rate
        points = MerweScaledSigmaPoints(n=7, alpha=.5, beta=2., kappa=0.)
        kf = UKF(dim_x=7, dim_z=3, dt=dt, fx=fx, hx=hx, points=points, sqrt_fn=None, x_mean_fn=state_mean, z_mean_fn=meas_mean, residual_x=residual_x, residual_z=residual_z)

        kf.x = np.array(self.ini_val)   # initial mean state
        kf.P *= 0.0001  # kf.P = eye(dim_x) ; adjust covariances if necessary
        # kf.R *= 0
        # kf.Q *= 0

        zs = []
        xs = []

        rospy.sleep(1)
        print "Initialized..."

        while rospy.get_time() == 0.0:
            continue

        start = rospy.get_time()

        rospy.loginfo("Moving robot...")

        count = 0
        while (rospy.get_time() - start < self._move_time) and not rospy.is_shutdown():
            
            z = np.array([self.odom_x, self.odom_y, self.odom_th])
            zs.append(z)

            kf.predict()
            kf.update(z)

            xs.append(kf.x)

            if count%100 == 0:
                print "Updated x: ", kf.x

                
            self.pub_twist.publish(self.wheel_cmd)
            self.r.sleep()

            count += 1
        # stop the robot
        self.pub_twist.publish(Twist())
        


    def solvr(self, x, t):

        omega1 = self.omegas(self.delta(x[5]),self.delta(x[6]))[0]
        omega2 = self.omegas(self.delta(x[5]),self.delta(x[6]))[1]
        omega3 = self.omegas(self.delta(x[5]),self.delta(x[6]))[2]

        dl = self.wh_consts[0]
        df = self.wh_consts[1]
        dc = self.wh_consts[2]

        # Assume v_w = 0  ==>  ignore lateral movement of wheelchair
        # ==>  remove function/equation involving v_w from the model
        eq1 = omega3/self.Iz
        eq2 = ((-omega1*np.sin(x[4]) + omega2*np.cos(x[4]))/self.m) - 0.*x[0]*x[2]
        eq3 = ((-omega1*np.cos(x[4]) - omega2*np.sin(x[4]))/self.m) + x[0]*x[1]
        eq4 = x[1]*np.sin(x[4]) - 0.*x[2]*np.cos(x[4])
        eq5 = -x[1]*np.cos(x[4]) - 0.*x[2]*np.sin(x[4])
        eq6 = x[0]
        eq7 = (x[0]*(dl*np.cos(x[5]) - (df*np.sin(x[5])/2) - dc)/dc) + (-x[1]*np.sin(x[5])/dc)
        eq8 = (x[0]*(dl*np.cos(x[6]) + (df*np.sin(x[6])/2) - dc)/dc) + (-x[1]*np.sin(x[6])/dc)

        return [eq1, eq2, eq4, eq5, eq6, eq7, eq8]



    def ode_solve(self, ini_val, dt):

        a_t = np.arange(0.0, dt)
        # ini_val = [self.wheel_cmd.angular.z, -self.wheel_cmd.linear.x, 0.0, 0.0, 0.0, self.pi, self.pi]
        # ini_val = [self.wheel_cmd.angular.z, -self.wheel_cmd.linear.x, -self.pose_y, self.pose_x, self.pose_th, self.angle_adj(self.r_caster_angle+self.pi), self.angle_adj(self.l_caster_angle+self.pi)]

        
        asol = odeint(self.solvr, ini_val, a_t)
        self.asol = asol


    def omegas(self, delta1, delta2):

        N = self.m*self.g

        F1u = self.mu*self.ep*N/2.
        F1w = 0.0      
        F2u = self.mu*self.ep*N/2.
        F2w = 0.0
        F3u = self.mu*(1-self.ep)*N/2.
        F3w = 0.0
        F4u = self.mu*(1-self.ep)*N/2.
        F4w = 0.0

        d = 0.0
        L = 0.58
        Rr = 0.27*2
        s = 0.0


        omega1 = (F3u*np.cos(delta1)) + (F3w*np.sin(delta1)) + F1u + F2u + (F4u*np.cos(delta2)) + (F4w*np.sin(delta2))
        omega2 = F1w - (F3u*np.sin(delta1)) + (F3w*np.cos(delta1)) - (F4u*np.sin(delta2)) + (F4w*np.cos(delta2)) + F2w
        omega3 = (F2u*(Rr/2.-s))-(F1u*(Rr/2.-s))-((F2w+F1w)*d)+((F4u*np.cos(delta2)+F4w*np.sin(delta2))*(Rr/2.-s))-((F3u*np.cos(delta1)-F3w*np.sin(delta1))*(Rr/2.+s))+((F4w*np.cos(delta2)-F4u*np.sin(delta2)+F3w*np.cos(delta1)-F3u*np.sin(delta1))*(L-d))

        return [omega1, omega2, omega3]

    def angle_adj(self, angle):
        angle = angle%self.two_pi
        angle = (angle+self.two_pi)%(self.two_pi)

        if angle > self.pi:
            angle -= self.two_pi
        return angle

    def delta(self, alpha):
        # return self.angle_adj(self.two_pi - (alpha%self.two_pi))
        return self.angle_adj(-alpha)



    def shutdown(self):
        # Stop the robot when shutting down the node.
        rospy.loginfo("Stopping the robot...")
        self.pub_twist.publish(Twist())
        rospy.sleep(1)


if __name__ == '__main__':

    try:
        UKFWheelchair()
    except rospy.ROSInterruptException:
        pass