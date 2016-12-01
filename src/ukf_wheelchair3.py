#!/usr/bin/env python

import rospy
import sys
from ukf_helper import MerweScaledSigmaPoints, state_mean, meas_mean, residual_x, residual_z, normalize_angle, rKN, sub_angle
from ukf import UKF
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from nuric_wheelchair_model_02.msg import FloatArray
from tf.transformations import euler_from_quaternion
import numpy as np
from scipy.integrate import odeint, ode
from math import sin, cos
import matplotlib.pyplot as plt


class UKFWheelchair3(object):

    def __init__(self):
        rospy.init_node('ukf_wheelchair3')

        rospy.on_shutdown(self.shutdown)

        self.wheel_cmd = Twist()

        self.wheel_cmd.linear.x = 0.2 
        self.wheel_cmd.angular.z = 0.1


        self.move_time = 4.0
        self.rate = 50
        self.factor = 1
        self.dt = self.factor*1.0/self.rate
        # self.dt = 0.1 

        self.zs = []

        # constants for ode equations
        # (approximations)
        self.Iz = 15.0
        self.mu = .01
        self.ep = 0.2
        self.m = 5.0
        self.g = 9.81/50.

        self.save_caster_data = []
        self.save_pose_data = []

        self.asol = []

        


        # wheelchair constants
        self.wh_consts = [0.58, 0.19, 0.06]

        self.odom_data = rospy.Subscriber('/odom', Odometry, self.odom_cb)
        self.caster_data = rospy.Subscriber('/caster_joints', FloatArray, self.caster_cb)
        self.pub_twist = rospy.Publisher('/cmd_vel', Twist, queue_size=20)

        self.r = rospy.Rate(self.rate)

        self.move_wheelchair()

        # self.solve_ukf()

        self.plot_data()



    def caster_cb(self, caster_joints):       
        self.l_caster_angle, self.r_caster_angle = caster_joints.data[0], caster_joints.data[1]

    def odom_cb(self, odom_data):
        self.odom_vx, self.odom_vth = odom_data.twist.twist.linear.x, odom_data.twist.twist.angular.z
        (_,_,yaw) = euler_from_quaternion([odom_data.pose.pose.orientation.x, odom_data.pose.pose.orientation.y, odom_data.pose.pose.orientation.z, odom_data.pose.pose.orientation.w])
        self.odom_x, self.odom_y, self.odom_th = odom_data.pose.pose.position.x, odom_data.pose.pose.position.y, yaw


    def move_wheelchair(self):

        while rospy.get_time() == 0.0:
            continue
        

        rospy.sleep(1)

        
        self.ini_val = [0.1, 0.2, 0.0, 0.0, 0.0, np.pi, np.pi]

        count = 0

        rospy.loginfo("Moving robot...")

        start = rospy.get_time()
        self.r.sleep()
        
        while (rospy.get_time() - start < self.move_time) and not rospy.is_shutdown():
            
            z = np.array([self.odom_x, self.odom_y, self.odom_th])
            
            self.zs.append(z)

            print len(self.zs)

            self.save_caster_data.append([self.l_caster_angle, self.r_caster_angle])


            self.pub_twist.publish(self.wheel_cmd)    
            
            count += 1
            self.r.sleep()


        # Stop the robot
        self.pub_twist.publish(Twist())


        rospy.sleep(1)

    def solve_ukf(self):

        def fx(x, dt):

            x[4], x[5], x[6] = normalize_angle(x[4]), normalize_angle(x[5]), normalize_angle(x[6])
            sol = self.ode_solve(x)
            return np.array(sol)

        def hx(x):
            return np.array([x[3], x[2], normalize_angle(x[4])])


        points = MerweScaledSigmaPoints(n=7, alpha=.4, beta=2., kappa=1.)
        kf = UKF(dim_x=7, dim_z=3, dt=self.dt, fx=fx, hx=hx, points=points, sqrt_fn=None, x_mean_fn=self.state_mean, z_mean_fn=self.meas_mean, residual_x=self.residual_x, residual_z=self.residual_z)

        kf.x = np.array(self.ini_val)   # initial mean state
        kf.P *= 0.0001  # kf.P = eye(dim_x) ; adjust covariances if necessary
        # kf.R *= 0
        # kf.Q *= 0

        # Ms, Ps = kf.batch_filter(self.zs)
        # Ms2 = self.solve_est()

        # print self.l_caster_angle, self.r_caster_angle
        # # print Ms[-1,5], Ms[-1,6]
        # print Ms2[-1][5], Ms[-1][6]


    def solve_est(self):
        count=0
        Ms = []
        x = np.array(self.ini_val)
        while count < 200:
            sol = self.ode_solve(x)
            Ms.append(sol)
            x = sol
            count += 1

        return Ms



    def plot_data(self):


        plt.figure(1)

        # Plot left caster
        plt.subplot(221)
        plt.title("L Caster Orientation (rad)")

        data = np.array(self.solve_est())
        data = data[:,6]

        data_est_l = [normalize_angle(angle-np.pi) for angle in data]
        xaxis = [x/50. for x in xrange(len(data_est_l))]
        plt.plot(xaxis, data_est_l, label="est")


        data_act = np.array(self.save_caster_data)
        data_act = data_act[:,0]

        xaxis = [x/50. for x in xrange(len(data_act))]
        plt.plot(xaxis, data_act, label="act")

        plt.legend()

        # plt.subplot(223)
        # plt.title("Error L Caster Orientation (rad)")
        # error_data = self.calc_error(data_act, data_ukf_l)
        # xaxis = [x/5. for x in xrange(len(error_data))]
        # plt.plot(xaxis, error_data, label="act - ukf")

        # # plt.subplot(325)
        # # plt.title("Error L Caster Orientation (rad)")
        # error_data = self.calc_error2(data_act, data_est_l)
        # xaxis = [x/50. for x in xrange(len(error_data))]
        # plt.plot(xaxis, error_data, label="act - est")

        # plt.legend()


        
        plt.show()



    def state_mean(self, sigmas, Wm):
        x = np.zeros(7)

        sum_sin1, sum_cos1 = 0., 0.
        sum_sin2, sum_cos2 = 0., 0.
        sum_sin3, sum_cos3 = 0., 0.

        for i in xrange(len(sigmas)):
            s = sigmas[i]

            x[0] += s[0] * Wm[i]
            x[1] += s[1] * Wm[i]
            x[2] += s[2] * Wm[i]
            x[3] += s[3] * Wm[i]
            x[5] += s[5] * Wm[i]
            x[6] += s[6] * Wm[i]

            sum_sin1 += np.sin(s[4])*Wm[i]
            sum_cos1 += np.cos(s[4])*Wm[i]

            # sum_sin2 += np.sin(s[5])*Wm[i]
            # sum_cos2 += np.cos(s[5])*Wm[i]

            # sum_sin3 += np.sin(s[6])*Wm[i]
            # sum_cos3 += np.cos(s[6])*Wm[i]

        x[4] = np.arctan2(sum_sin1, sum_cos1)
        # x[5] = np.arctan2(sum_sin2, sum_cos2)
        # x[6] = np.arctan2(sum_sin3, sum_cos3)

        return x

    def meas_mean(self, sigmas, Wm):
        z = np.zeros(3)

        sum_sin1, sum_cos1 = 0., 0.

        for i in xrange(len(sigmas)):
            s = sigmas[i]

            z[0] += s[0] * Wm[i]
            z[1] += s[1] * Wm[i]
            

            sum_sin1 += np.sin(s[2])*Wm[i]
            sum_cos1 += np.cos(s[2])*Wm[i]

        z[2] = np.arctan2(sum_sin1, sum_cos1)

        return z

    def residual_x(self, a, b):
        y = np.zeros(7)

        y[0] = a[0] - b[0]
        y[1] = a[1] - b[1]
        y[2] = a[2] - b[2]
        y[3] = a[3] - b[3]
        y[4] = sub_angle(a[4] - b[4])
        y[5] = sub_angle(a[5] - b[5])
        y[6] = sub_angle(a[6] - b[6])


        y[4] = normalize_angle(y[4])

        y[5] = normalize_angle(y[5])
        y[6] = normalize_angle(y[6])

        return y 

    def residual_z(self, a, b):
        y = np.zeros(3)

        y[0] = a[0] - b[0]
        y[1] = a[1] - b[1]
        y[2] = sub_angle(a[2] - b[2])

        y[2] = normalize_angle(y[2])

        return y

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
        eq2 = ((-omega1*sin(x[4]) + omega2*cos(x[4]))/self.m) - 0.*x[0]*x[2]
        eq3 = ((-omega1*cos(x[4]) - omega2*sin(x[4]))/self.m) + x[0]*x[1]
        eq4 = -x[1]*sin(x[4]) - 0.*x[2]*cos(x[4])
        eq5 = x[1]*cos(x[4]) - 0.*x[2]*sin(x[4])
        eq6 = x[0]
        eq7 = (x[0]*(dl*cos(x[5]) - (df*sin(x[5])/2) - dc)/dc) + (-x[1]*sin(x[5])/dc)
        eq8 = (x[0]*(dl*cos(x[6]) + (df*sin(x[6])/2) - dc)/dc) + (-x[1]*sin(x[6])/dc)

        return [eq1, eq2, eq4, eq5, eq6, eq7, eq8]


    def ode_solve(self, x0):
        a_t = np.arange(0.0, 0.02, 0.005)
        ini_val = x0
        asol = odeint(self.solvr, ini_val, a_t)

        sol = asol[-1]

        sol[4], sol[5], sol[6] = normalize_angle(sol[4]), normalize_angle(sol[5]), normalize_angle(sol[6])
        
        return sol

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


        omega1 = (F3u*cos(delta1)) + (F3w*sin(delta1)) + F1u + F2u + (F4u*cos(delta2)) + (F4w*sin(delta2))
        omega2 = F1w - (F3u*sin(delta1)) + (F3w*cos(delta1)) - (F4u*sin(delta2)) + (F4w*cos(delta2)) + F2w
        omega3 = (F2u*(Rr/2.-s))-(F1u*(Rr/2.-s))-((F2w+F1w)*d)+((F4u*cos(delta2)+F4w*sin(delta2))*(Rr/2.-s))-((F3u*cos(delta1)-F3w*sin(delta1))*(Rr/2.+s))+((F4w*cos(delta2)-F4u*sin(delta2)+F3w*cos(delta1)-F3u*sin(delta1))*(L-d))

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
        UKFWheelchair3()
    except rospy.ROSInterruptException:
        pass