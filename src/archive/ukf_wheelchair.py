#!/usr/bin/env python

import rospy
import sys
from ukf_helper import MerweScaledSigmaPoints, state_mean, meas_mean, residual_x, residual_z, normalize_angle, rKN
from ukf import UKF
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion
import numpy as np 
from scipy.integrate import odeint, ode
from math import sin, cos

class UKFWheelchair(object):

    def __init__(self):

        rospy.init_node('ukf_wheelchair')

        rospy.on_shutdown(self.shutdown)

        self.wheel_cmd = Twist()
        self.wheel_cmd.linear.x = 0.3
        self.wheel_cmd.angular.z = 0.0

        self._move_time = 10.0
        self._rate = 50

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

        self.dt = 0.02



        self.ini_val = [0.1, 0.2, 0.0, 0.0, 0.0, 0.0+self.pi, 0.0+self.pi]

        self.odom_data = rospy.Subscriber('/odom', Odometry, self.odom_cb)

        self.pub_twist = rospy.Publisher('/cmd_vel', Twist, queue_size=20)

        self.r = rospy.Rate(self._rate)

        self.run_ukf_wheelchair()



    def odom_cb(self, odom_data):

        # if self.save:
        (_,_,yaw) = euler_from_quaternion([odom_data.pose.pose.orientation.x, odom_data.pose.pose.orientation.y, odom_data.pose.pose.orientation.z, odom_data.pose.pose.orientation.w])

        self.odom_x, self.odom_y, self.odom_th = odom_data.pose.pose.position.x, odom_data.pose.pose.position.y, yaw


    def run_ukf_wheelchair(self):

        def fx(x, dt):
            x[4], x[5], x[6] = normalize_angle(x[4]), normalize_angle(x[5]), normalize_angle(x[6])
            sol = self.ode_solve(x)
            return np.array(sol)


        def hx(x):                        
            return np.array([x[3], x[2], x[4]])

        dt = 1./self._rate
        points = MerweScaledSigmaPoints(n=7, alpha=.5, beta=2., kappa=4.)
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

            
            # print len(zs)
            z = np.array([self.odom_x, self.odom_y, self.odom_th])
            zs.append(z)

            kf.predict()
            kf.update(z)

            # xs.append(kf.x)

            print kf.x

                
            self.pub_twist.publish(self.wheel_cmd)

            count += 1
            self.r.sleep()
        
        # stop the robot
        self.pub_twist.publish(Twist())
        rospy.sleep(1)


    def fun(self, t, x):
        thdot, ydot, x, y, th, alpha1, alpha2 = x 

        omega1 = self.omegas(self.delta(alpha1),self.delta(alpha2))[0]
        omega2 = self.omegas(self.delta(alpha1),self.delta(alpha2))[1]
        omega3 = self.omegas(self.delta(alpha1),self.delta(alpha2))[2]

        dl = self.wh_consts[0]
        df = self.wh_consts[1]
        dc = self.wh_consts[2]

        # Assume v_w = 0  ==>  ignore lateral movement of wheelchair
        # ==>  remove function/equation involving v_w from the model
        eq1 = omega3/self.Iz
        eq2 = ((-omega1*sin(th) + omega2*cos(th))/self.m)
        # eq3 = ((-omega1*cos(th) - omega2*sin(th))/self.m) + thdot*ydot
        eq4 = ydot*sin(th)
        eq5 = -ydot*cos(th) 
        eq6 = thdot
        eq7 = (thdot*(dl*cos(alpha1) - (df*sin(alpha1)/2) - dc)/dc) + (-ydot*sin(alpha1)/dc)
        eq8 = (thdot*(dl*cos(alpha2) + (df*sin(alpha2)/2) - dc)/dc) + (-ydot*sin(alpha2)/dc)

        f = [eq1, eq2, eq4, eq5, eq6, eq7, eq8]

        return f


    def ode_solve(self, x0):

        solver = ode(self.fun)
        solver.set_integrator('dopri5')

        t0 = 0.0
        # x0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        solver.set_initial_value(x0, t0)

        t1 = self.dt
        N = 20
        t = np.linspace(t0, t1, N)
        sol = np.empty((N, 7))
        sol[0] = x0

        k=1
        while solver.successful() and solver.t < t1:
            solver.integrate(t[k])
            sol[k] = solver.y
            k += 1

        return sol[-1]


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

    

    def delta(self, alpha):
        return normalize_angle(-alpha)



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