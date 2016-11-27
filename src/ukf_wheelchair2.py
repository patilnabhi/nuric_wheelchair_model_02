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


class UKFWheelchair2(object):

    def __init__(self):
        rospy.init_node('ukf_wheelchair2')

        rospy.on_shutdown(self.shutdown)

        self.wheel_cmd = Twist()

        self.wheel_cmd.linear.x = -0.3 
        self.wheel_cmd.angular.z = 0.2


        self.move_time = 3.0
        self.rate = 50
        self.factor = 10
        self.dt = self.factor*1.0/self.rate 

        self.save_caster_data = []

        self.asol = []

        self.sol_alpha1 = []
        self.sol_alpha2 = []

        self.error_alpha1 = []
        self.error_alpha2 = []


        # wheelchair constants
        self.wh_consts = [0.58, 0.19, 0.06]

        self.odom_data = rospy.Subscriber('/odom', Odometry, self.odom_cb)
        self.caster_data = rospy.Subscriber('/caster_joints', FloatArray, self.caster_cb)
        self.pub_twist = rospy.Publisher('/cmd_vel', Twist, queue_size=20)

        self.r = rospy.Rate(self.rate)

        self.ukf_move_wheelchair()

        self.caster_model()

        self.plot_data()



    def caster_cb(self, caster_joints):       
        self.l_caster_angle, self.r_caster_angle = caster_joints.data[0], caster_joints.data[1]

    def odom_cb(self, odom_data):
        self.odom_vx, self.odom_vth = odom_data.twist.twist.linear.x, odom_data.twist.twist.angular.z
        (_,_,yaw) = euler_from_quaternion([odom_data.pose.pose.orientation.x, odom_data.pose.pose.orientation.y, odom_data.pose.pose.orientation.z, odom_data.pose.pose.orientation.w])
        self.odom_x, self.odom_th = odom_data.pose.pose.position.x, yaw


    def ukf_move_wheelchair(self):

        while rospy.get_time() == 0.0:
            continue
        

        rospy.sleep(1)

        def fx(x, dt):
            x[0], x[1] = normalize_angle(x[0]), normalize_angle(x[1])
            self.prev_alpha1 = x[0]
            self.prev_alpha2 = x[1]

            return self.caster_model_ukf(x, dt)

        def hx(x):
            # print "2: ", self.prev_alpha1
            delta_alpha1 = x[0] - self.prev_alpha1
            delta_alpha2 = x[1] - self.prev_alpha2
            alpha1dot = delta_alpha1/self.dt
            alpha2dot = delta_alpha2/self.dt

            sol = self.meas_model(x[0], x[1], alpha1dot, alpha2dot)
            return sol
        
        points = MerweScaledSigmaPoints(n=2, alpha=.1, beta=1., kappa=-1.)
        kf = UKF(dim_x=2, dim_z=2, dt=self.dt, fx=fx, hx=hx, points=points, sqrt_fn=None, x_mean_fn=self.state_mean, z_mean_fn=self.meas_mean, residual_x=self.residual_x, residual_z=self.residual_z)

        # self.ini_val = [normalize_angle(self.l_caster_angle-np.pi), normalize_angle(self.r_caster_angle-np.pi)]
        self.ini_val = [3.1, -3.14]
        self.save_caster_data.append(self.ini_val)

        kf.x = np.array(self.ini_val)   # initial mean state
        kf.P *= 0.0001  # kf.P = eye(dim_x) ; adjust covariances if necessary
        # kf.R *= 0
        # kf.Q *= 0

        zs = []
        xs = []

        # xs.append(self.ini_val)

        count = 0

        # print "Est1: ", normalize_angle(kf.x[0]+np.pi), normalize_angle(kf.x[1]+np.pi) 

        rospy.loginfo("Moving robot...")
        
        last_odom_x = self.odom_x
        last_odom_th = self.odom_th
        start = rospy.get_time()
        self.r.sleep()
        
        while (rospy.get_time() - start < self.move_time) and not rospy.is_shutdown():
            curr_odom_x, curr_odom_th = self.odom_x, self.odom_th
            delta_x, delta_th = curr_odom_x - last_odom_x, curr_odom_th - last_odom_th
            z = np.array([delta_x/self.dt, delta_th/self.dt])
            # z = np.array([self.odom_vx, self.odom_vth])
            
            if count%self.factor==0:

                zs.append(z)

                kf.predict()   
                kf.update(z)

                xs.append([normalize_angle(kf.x[0]+np.pi), normalize_angle(kf.x[1]+np.pi)])
    

                # print "Est: ", normalize_angle(kf.x[0]+np.pi), normalize_angle(kf.x[1]+np.pi)
                # print "Act: ", normalize_angle(self.l_caster_angle), normalize_angle(self.r_caster_angle)

            self.save_caster_data.append([self.l_caster_angle, self.r_caster_angle])


            self.pub_twist.publish(self.wheel_cmd)    
            
            count += 1
            last_odom_x, last_odom_th = curr_odom_x, curr_odom_th
            self.r.sleep()


        # Stop the robot
        self.pub_twist.publish(Twist())

        self.caster_sol_ukf = np.array(xs)
        self.caster_sol_act = np.array(self.save_caster_data)

        rospy.sleep(1)

    def solvr_caster_model(self, x, t, vx=None, vth=None):

        dl = self.wh_consts[0]
        df = self.wh_consts[1]
        dc = self.wh_consts[2]

        if vx is None:
            vx = self.wheel_cmd.linear.x
        if vth is None:
            vth = self.wheel_cmd.angular.z

        eq1 = (vth*(dl*cos(x[0]) - (df*sin(x[0])/2) - dc)/dc) + (vx*sin(x[0])/dc)
        eq2 = (vth*(dl*cos(x[1]) + (df*sin(x[1])/2) - dc)/dc) + (vx*sin(x[1])/dc)

        return [eq1, eq2]


    def caster_model_ukf(self, x0, dt):

        a_t = np.arange(0.0, dt, 0.001)
        # ini_val = [normalize_angle(self.r_caster_angle-np.pi), normalize_angle(self.l_caster_angle-np.pi)]
        ini_val = x0
        asol = odeint(self.solvr_caster_model, ini_val, a_t)
        
        
        return asol[-1]


    def caster_model(self):

        a_t = np.arange(0.0, self.move_time, self.dt)
        ini_val = self.ini_val
        asol = odeint(self.solvr_caster_model, ini_val, a_t)


        self.caster_sol = asol


    def meas_model(self, alpha1, alpha2, alpha1dot, alpha2dot):
        dl = self.wh_consts[0]
        df = self.wh_consts[1]
        dc = self.wh_consts[2]

        # eq1 = (vth*(dl*cos(x[0]) - (df*sin(x[0])/2) - dc)/dc) + (vx*sin(x[0])/dc)
        # eq2 = (vth*(dl*cos(x[1]) + (df*sin(x[1])/2) - dc)/dc) + (vx*sin(x[1])/dc)

        a2 = (dl*cos(alpha1) - (df*sin(alpha1)/2) - dc)/dc
        a4 = (dl*cos(alpha2) + (df*sin(alpha2)/2) - dc)/dc
        a1 = sin(alpha1)/dc
        a3 = sin(alpha2)/dc

        b1 = alpha1dot
        b2 = alpha2dot

        A = np.array([[a1,a2], [a3,a4]])
        B = np.array([b1,b2])

        sol = np.linalg.solve(A,B)

        return sol

    def state_mean(self, sigmas, Wm):
        x = np.zeros(2)

        sum_sin1, sum_cos1 = 0., 0.
        sum_sin2, sum_cos2 = 0., 0.
        
        for i in range(len(sigmas)):
            s = sigmas[i] 

            x[0] += s[0] * Wm[i]
            x[1] += s[1] * Wm[i] 

            

        #     sum_sin1 += np.sin(s[0])*Wm[i]
        #     sum_cos1 += np.cos(s[0])*Wm[i]

        #     sum_sin2 += np.sin(s[1])*Wm[i]
        #     sum_cos2 += np.cos(s[1])*Wm[i]

        # x[0] = normalize_angle(x[0])
        # x[1] = normalize_angle(x[1])

        # x[0] = np.arctan2(sum_sin1, sum_cos1)
        # x[1] = np.arctan2(sum_sin2, sum_cos2)
        
        return x

    def meas_mean(self, sigmas, Wm):
        z = np.zeros(2)

        for i in range(len(sigmas)):
            s = sigmas[i]
            z[0] += s[0] * Wm[i]
            z[1] += s[1] * Wm[i]

        return z

    def residual_x(self, a, b):
        y = np.zeros(2)
        
        # for i in xrange(len(a)):
        # y[0] = a[0] - b[0]
        # y[1] = a[1] - b[1]
        y[0] = sub_angle(a[0] - b[0])
        y[1] = sub_angle(a[1] - b[1])
        # y[0] = normalize_angle(y[0])
        # y[1] = normalize_angle(y[1])

        return y

    def residual_z(self, a, b):
        y = np.zeros(2)
        
        # for i in xrange(len(a)):
        y[0] = a[0] - b[0]
        y[1] = a[1] - b[1]
        # y[:2] = a[:2] - b[:2]
            
        return y
    
    def calc_error(self, data1, data2):

        error = []

        for i in xrange(min(len(data1), len(data2))):
            error.append(data1[i]-data2[i])

        return error

        # for i in xrange(min(len(self.sol_alpha2), len(self.l_caster_data))):
        #     self.error_alpha2.append(self.sol_alpha2[i]-self.l_caster_data[i])

    def generate_data(self, data, side):

        out_data = []

        if side == 'left':
            side = 0
        elif side == 'right':
            side = 1
        else:
            print "Check arguments!"
            return

        for i in xrange(len(data)):
            out_data.append(data[i,side])

        return out_data

    def plot_data(self):


        # self.calc_error()


        

        plt.figure(1)

        # Plot left caster
        plt.subplot(221)
        plt.title("L Caster Orientation (rad)")

        data = self.generate_data(self.caster_sol, 'left')
        data_est = [normalize_angle(angle+np.pi) for angle in data]
        xaxis = [x/5. for x in xrange(len(data_est))]
        plt.plot(xaxis, data_est, label="est")

        data_ukf = self.generate_data(self.caster_sol_ukf, 'left')
        xaxis = [x/5. for x in xrange(len(data_ukf))]
        plt.plot(xaxis, data_ukf, label="ukf")

        data = self.generate_data(self.caster_sol_act, 'left')
        xaxis = [x/self.rate for x in xrange(len(data))]
        plt.plot(xaxis, data, label="act")

        plt.legend()



        # plt.subplot(222)
        # plt.title("L Caster Orientation (rad)")
        # xaxis = [x/100. for x in xrange(len(self.sol_alpha2))]
        # plt.plot(xaxis, self.sol_alpha2, label="est")
        # xaxis = [x/100. for x in xrange(len(self.l_caster_data))]
        # plt.plot(xaxis, self.l_caster_data, label="actual")
        # plt.legend()

        plt.subplot(223)
        plt.title("Error L Caster Orientation (rad)")
        error_data = self.calc_error(data_est, data_ukf)
        xaxis = [x/5. for x in xrange(len(error_data))]
        plt.plot(xaxis, error_data)

        # plt.subplot(224)
        # plt.title("Error L Caster Orientation (rad)")
        # xaxis = [x/100. for x in xrange(len(self.error_alpha2))]
        # plt.plot(xaxis, self.error_alpha2)

        plt.show()


    def shutdown(self):
        # Stop the robot when shutting down the node.
        rospy.loginfo("Stopping the robot...")
        self.pub_twist.publish(Twist())
        rospy.sleep(1)


if __name__ == '__main__':

    try:
        UKFWheelchair2()
    except rospy.ROSInterruptException:
        pass