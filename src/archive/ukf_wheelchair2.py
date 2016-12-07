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

        self.wheel_cmd.linear.x = 0.2 
        self.wheel_cmd.angular.z = 0.3


        self.move_time = 10.0
        self.rate = 50
        self.factor = 10
        self.dt = self.factor*1.0/self.rate
        # self.dt = 0.1 

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
        self.odom_x, self.odom_y, self.odom_th = odom_data.pose.pose.position.x, odom_data.pose.pose.position.y, yaw


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
        
        self.ini_pose = [self.wheel_cmd.angular.z, -self.wheel_cmd.linear.x, -self.odom_y, self.odom_x, self.odom_th]
        self.save_pose_data.append([self.ini_pose[2], self.ini_pose[3], self.ini_pose[4]])

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
            self.save_pose_data.append([self.odom_x, self.odom_y, self.odom_th])
            # print len(self.save_caster_data)


            self.pub_twist.publish(self.wheel_cmd)    
            
            count += 1
            last_odom_x, last_odom_th = curr_odom_x, curr_odom_th
            self.r.sleep()


        # Stop the robot
        self.pub_twist.publish(Twist())

        self.caster_sol_ukf = np.array(xs)
        self.caster_sol_act = np.array(self.save_caster_data)
        self.pose_act = np.array(self.save_pose_data)

        # self.pose_act = np.reshape(self.pose_act, (len(self.save_pose_data), 3))

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

    def solvr_dynamic_model(self, x, t):
        omega1 = self.omegas(self.delta(self.alpha1),self.delta(self.alpha2))[0]
        omega2 = self.omegas(self.delta(self.alpha1),self.delta(self.alpha2))[1]
        omega3 = self.omegas(self.delta(self.alpha1),self.delta(self.alpha2))[2]

        dl = self.wh_consts[0]
        df = self.wh_consts[1]
        dc = self.wh_consts[2]

        # Assume v_w = 0  ==>  ignore lateral movement of wheelchair
        # ==>  remove function/equation involving v_w from the model
        eq1 = omega3/self.Iz
        eq2 = ((-omega1*sin(x[4]) + omega2*cos(x[4]))/self.m) - 0.*x[0]*x[2]
        eq3 = ((-omega1*cos(x[4]) - omega2*sin(x[4]))/self.m) + x[0]*x[1]
        eq4 = x[1]*sin(x[4]) - 0.*x[2]*cos(x[4])
        eq5 = -x[1]*cos(x[4]) - 0.*x[2]*sin(x[4])
        eq6 = x[0]

        return [eq1, eq2, eq4, eq5, eq6]


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


    def dynamic_model(self, x0, alpha1, alpha2):
        a_t = np.arange(0.0, self.dt, 0.001)
        ini_val = x0

        self.alpha1 = alpha1
        self.alpha2 = alpha2

        asol = odeint(self.solvr_dynamic_model, ini_val, a_t)
        sol = asol[-1]

        sol[4] = normalize_angle(sol[4])
        return sol

    def get_est_pose(self, alpha1s, alpha2s, factor):

        x0 = self.ini_pose
        t = 0.0
        est_pose = np.array(x0)
        est_pose = np.reshape(est_pose, (1,5))

        for i in xrange(len(alpha1s)/factor):
            alpha1, alpha2 = alpha1s[i*factor], alpha2s[i*factor]

            out = self.dynamic_model(x0, alpha1, alpha2)
            x0 = out
            out = np.reshape(out, (1,5))
            est_pose = np.append(est_pose, out, axis=0)
             
            t += self.dt

        return est_pose


    def caster_model_ukf(self, x0, dt):

        a_t = np.arange(0.0, dt, 0.001)
        # ini_val = [normalize_angle(self.r_caster_angle-np.pi), normalize_angle(self.l_caster_angle-np.pi)]
        ini_val = x0
        asol = odeint(self.solvr_caster_model, ini_val, a_t)
        
        
        return asol[-1]


    def caster_model(self):

        a_t = np.arange(0.0, self.move_time, 0.01)
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
            error.append(data1[i*10]-data2[i])

        return error

    def calc_error2(self, data1, data2):

        error = []

        for i in xrange(500):
            error.append(data1[i]-data2[i*2])

        return error

    def calc_error3(self, data1, data2):

        error = []

        for i in xrange(min(len(data1), len(data2))):
            error.append(data1[i*10]-data2[i])

        return error

        

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




        

        plt.figure(1)

        # Plot left caster
        plt.subplot(221)
        plt.title("L Caster Orientation (rad)")

        data = self.generate_data(self.caster_sol, 'left')
        data_est_l = [normalize_angle(angle+np.pi) for angle in data]
        xaxis = [x/100. for x in xrange(len(data_est_l))]
        plt.plot(xaxis, data_est_l, label="est")

        data_ukf_l = self.generate_data(self.caster_sol_ukf, 'left')
        xaxis = [x/5. for x in xrange(len(data_ukf_l))]
        plt.plot(xaxis, data_ukf_l, label="ukf")

        data_act = self.generate_data(self.caster_sol_act, 'left')
        xaxis = [x/self.rate for x in xrange(len(data_act))]
        plt.plot(xaxis, data_act, label="act")

        plt.legend()

        plt.subplot(223)
        plt.title("Error L Caster Orientation (rad)")
        error_data = self.calc_error(data_act, data_ukf_l)
        xaxis = [x/5. for x in xrange(len(error_data))]
        plt.plot(xaxis, error_data, label="act - ukf")

        # plt.subplot(325)
        # plt.title("Error L Caster Orientation (rad)")
        error_data = self.calc_error2(data_act, data_est_l)
        xaxis = [x/50. for x in xrange(len(error_data))]
        plt.plot(xaxis, error_data, label="act - est")

        plt.legend()


        # Plot right caster
        plt.subplot(222)
        plt.title("R Caster Orientation (rad)")

        data = self.generate_data(self.caster_sol, 'right')
        data_est_r = [normalize_angle(angle+np.pi) for angle in data]
        xaxis = [x/100. for x in xrange(len(data_est_r))]
        plt.plot(xaxis, data_est_r, label="est")

        data_ukf_r = self.generate_data(self.caster_sol_ukf, 'right')
        xaxis = [x/5. for x in xrange(len(data_ukf_r))]
        plt.plot(xaxis, data_ukf_r, label="ukf")

        data_act = self.generate_data(self.caster_sol_act, 'right')
        xaxis = [x/self.rate for x in xrange(len(data_act))]
        plt.plot(xaxis, data_act, label="act")

        plt.legend()

        plt.subplot(224)
        plt.title("Error R Caster Orientation (rad)")
        error_data = self.calc_error(data_act, data_ukf_r)
        xaxis = [x/5. for x in xrange(len(error_data))]
        plt.plot(xaxis, error_data, label="act - ukf")

        # plt.subplot(326)
        # plt.title("Error R Caster Orientation (rad)")
        error_data = self.calc_error2(data_act, data_est_r)
        xaxis = [x/50. for x in xrange(len(error_data))]
        plt.plot(xaxis, error_data, label="act - est")

        plt.legend()


        est_poses = self.get_est_pose(data_est_l, data_est_r, 20)
        ukf_poses = self.get_est_pose(data_ukf_l, data_ukf_r, 1)

        x_est = [est_poses[i,3] for i in xrange(len(est_poses))]
        y_est = [-est_poses[i,2] for i in xrange(len(est_poses))]
        th_est = [est_poses[i,4] for i in xrange(len(est_poses))]

        x_ukf = [ukf_poses[i,3] for i in xrange(len(ukf_poses))]
        y_ukf = [-ukf_poses[i,2] for i in xrange(len(ukf_poses))]
        th_ukf = [ukf_poses[i,4] for i in xrange(len(ukf_poses))]


        x_act = [self.pose_act[i,0] for i in xrange(len(self.pose_act))]
        y_act = [self.pose_act[i,1] for i in xrange(len(self.pose_act))]
        th_act = [self.pose_act[i,2] for i in xrange(len(self.pose_act))]

        


        plt.figure(2)

        plt.subplot(231)
        plt.title("Pose x (m)")
        xaxis = [x/5. for x in xrange(len(x_est))]
        plt.plot(xaxis, x_est, label="est")
        xaxis = [x/5. for x in xrange(len(x_ukf))]
        plt.plot(xaxis, x_ukf, label="ukf")
        xaxis = [x/self.rate for x in xrange(len(x_act))]
        plt.plot(xaxis, x_act, label="act")

        plt.legend()

        plt.subplot(234)

        plt.title("Error pose x (m)")
        error_data = self.calc_error3(x_act, x_ukf)
        xaxis = [x/5. for x in xrange(len(error_data))]
        plt.plot(xaxis, error_data, label="act - ukf")

        error_data = self.calc_error3(x_act, x_est)
        xaxis = [x/5. for x in xrange(len(error_data))]
        plt.plot(xaxis, error_data, label="act - est")

        plt.legend()

        plt.subplot(232)
        plt.title("Pose y (m)")
        xaxis = [x/5. for x in xrange(len(y_est))]
        plt.plot(xaxis, y_est, label="est")
        xaxis = [x/5. for x in xrange(len(y_ukf))]
        plt.plot(xaxis, y_ukf, label="ukf")
        xaxis = [x/self.rate for x in xrange(len(y_act))]
        plt.plot(xaxis, y_act, label="act")

        plt.legend()

        plt.subplot(235)

        plt.title("Error pose y (m)")
        error_data = self.calc_error3(y_act, y_ukf)
        xaxis = [x/5. for x in xrange(len(error_data))]
        plt.plot(xaxis, error_data, label="act - ukf")

        error_data = self.calc_error3(y_act, y_est)
        xaxis = [x/5. for x in xrange(len(error_data))]
        plt.plot(xaxis, error_data, label="act - est")

        plt.legend()

        plt.subplot(233)
        plt.title("Orientation (rad)")
        xaxis = [x/5. for x in xrange(len(th_est))]
        plt.plot(xaxis, th_est, label="est")
        xaxis = [x/5. for x in xrange(len(th_ukf))]
        plt.plot(xaxis, th_ukf, label="ukf")
        xaxis = [x/self.rate for x in xrange(len(th_act))]
        plt.plot(xaxis, th_act, label="act")

        plt.legend()

        plt.subplot(236)

        plt.title("Error orientation (rad)")
        error_data = self.calc_error3(th_act, th_ukf)
        xaxis = [x/5. for x in xrange(len(error_data))]
        plt.plot(xaxis, error_data, label="act - ukf")

        error_data = self.calc_error3(th_act, th_est)
        xaxis = [x/5. for x in xrange(len(error_data))]
        plt.plot(xaxis, error_data, label="act - est")

        plt.legend()


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