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

class SolveDynamicModel2:
    def __init__(self):
        rospy.init_node('solve_dynamic_model2')

        rospy.on_shutdown(self.shutdown)
        self.wheel_cmd = Twist()

        self.wheel_cmd.linear.x = 0.4 # Driving back w/o turn and a non-zero caster orientation
        self.wheel_cmd.angular.z = 0.2


        self.move_time = 4.0
        self.rate = 50

        self.pose_x_data = []
        self.pose_y_data = []
        self.pose_th_data = []
        self.l_caster_data = []
        self.r_caster_data = []

        self.asol = []

        self.solx = []
        self.soly = []
        self.solth = []
        self.soldel1 = []
        self.soldel2 = []

        self.errorx = []
        self.errory = []
        self.errorth = []
        self.errordel1 = []
        self.errordel2 = []

        self.angle_adj = normalize_angle


        # constants for ode equations
        # (approximations)
        self.Iz = 15.0
        self.mu = .01
        self.ep = 0.2
        self.m = 5.0
        self.g = 9.81/50.
        self.pi = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348
        self.two_pi = 2.*self.pi

        # wheelchair constants
        self.wh_consts = [0.58, 0.19, 0.06]

        self.save = 0
        self.get_caster_data = 0
        self.save_caster_data = 0
        self.get_pose = 0

        self.count = 0

        self.dt = 1./self.rate

        self.actual_pose = rospy.Subscriber('/odom', Odometry, self.actual_pose_callback)

        self.caster_joints = rospy.Subscriber('/caster_joints', FloatArray, self.caster_joints_callback)

        self.pub_twist = rospy.Publisher('/cmd_vel', Twist, queue_size=20)

        self.r = rospy.Rate(self.rate)

        self.move_wheelchair()

        self.plot_data()
       


    def actual_pose_callback(self, actual_pose):
        # (roll,pitch,yaw) = euler_from_quaternion([actual_pose.pose.pose.orientation.x, actual_pose.pose.pose.orientation.y, actual_pose.pose.pose.orientation.z, actual_pose.pose.pose.orientation.w])

        # self.pose_x = actual_pose.pose.pose.position.x
        # self.pose_y = actual_pose.pose.pose.position.y
        # self.pose_th = yaw

        if self.save:
            (_,_,yaw) = euler_from_quaternion([actual_pose.pose.pose.orientation.x, actual_pose.pose.pose.orientation.y, actual_pose.pose.pose.orientation.z, actual_pose.pose.pose.orientation.w])

            self.pose_x_data.append(actual_pose.pose.pose.position.x)
            self.pose_y_data.append(actual_pose.pose.pose.position.y)
            self.pose_th_data.append(yaw)

        # if self.get_pose:
        (_,_,yaw) = euler_from_quaternion([actual_pose.pose.pose.orientation.x, actual_pose.pose.pose.orientation.y, actual_pose.pose.pose.orientation.z, actual_pose.pose.pose.orientation.w])

        self.pose_x = actual_pose.pose.pose.position.x
        self.pose_y = actual_pose.pose.pose.position.y
        self.pose_th = yaw
        # print self.pose_x, self.pose_y

    def caster_joints_callback(self, caster_joints):       
        
        if self.save_caster_data:
            self.l_caster_data.append(caster_joints.data[0])
            self.r_caster_data.append(caster_joints.data[1])
        # if self.get_caster_data:
        self.l_caster_angle, self.r_caster_angle = caster_joints.data[0], caster_joints.data[1]

    def move_wheelchair(self):

        

        self.get_caster_data = 1
        self.get_pose = 1
        self.r.sleep()
        
        while rospy.get_time() == 0.0:
            continue
        start = rospy.get_time()
        self.get_caster_data = 0
        self.get_pose = 0
        rospy.sleep(1)
        self.ini_val = [self.wheel_cmd.angular.z, -self.wheel_cmd.linear.x, -self.pose_y, self.pose_x, self.pose_th, self.angle_adj(self.r_caster_angle+self.pi), self.angle_adj(self.l_caster_angle+self.pi)]

        x0=np.array(self.ini_val)
        x0 = np.reshape(x0, (1,7))
        sol = x0

        # print sol.shape

        # print self.pose_x
        count = 0
        rospy.loginfo("Moving robot...")
        while (rospy.get_time() - start < self.move_time) and not rospy.is_shutdown():
            
            self.pub_twist.publish(self.wheel_cmd)    
            
            sol1 = self.fx(x0)
            sol1 = np.reshape(sol1, (1,7))
            # print sol1.shape
            sol = np.append(sol, sol1, axis=0)
            x0 = sol1

            self.pose_x_data.append(self.pose_x)
            self.pose_y_data.append(self.pose_y)
            self.pose_th_data.append(self.pose_th)
            self.l_caster_data.append(self.l_caster_angle)
            self.r_caster_data.append(self.r_caster_angle)

            # print len(self.pose_x_data)
            # print len(sol)

            count += 1

            self.r.sleep()
            # self.save, self.save_caster_data = 1, 1



        # Stop the robot
        self.pub_twist.publish(Twist())
        self.save, self.save_caster_data = 0, 0
        # print len(sol)
        self.asol = sol
        

        rospy.sleep(1)


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
        eq3 = ((-omega1*cos(th) - omega2*sin(th))/self.m) + thdot*ydot
        eq4 = ydot*sin(th)
        eq5 = -ydot*cos(th) 
        eq6 = thdot
        eq7 = (thdot*(dl*cos(alpha1) - (df*sin(alpha1)/2) - dc)/dc) + (-ydot*sin(alpha1)/dc)
        eq8 = (thdot*(dl*cos(alpha2) + (df*sin(alpha2)/2) - dc)/dc) + (-ydot*sin(alpha2)/dc)

        f = [eq1, eq2, eq4, eq5, eq6, eq7, eq8]

        return f

    def fx(self, x0):
        solver = ode(self.fun)
        solver.set_integrator('dop853')

        t0 = 0.0
        # x0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        x0 = np.reshape(x0, (7,))
        x0 = x0.tolist()
        # print x0
        solver.set_initial_value(x0, t0)

        t1 = self.dt
        N = 50
        t = np.linspace(t0, t1, N)
        sol = np.empty((N, 7))
        sol[0] = x0

        k=1
        while solver.successful() and solver.t < t1:
            solver.integrate(t[k])
            sol[k] = solver.y
            k += 1

        # out = sol[-1]
        # out[5] = self.angle_adj(out[5])
        # out[6] = self.angle_adj(out[6])
        return sol[-1]



    def solvr(self, q, t):

        # delta2 = self.pi - self.l_caster_angle
        # delta1 = self.pi - self.r_caster_angle

        omega1 = self.omegas(self.delta(q[5]),self.delta(q[6]))[0]
        omega2 = self.omegas(self.delta(q[5]),self.delta(q[6]))[1]
        omega3 = self.omegas(self.delta(q[5]),self.delta(q[6]))[2]

        dl = self.wh_consts[0]
        df = self.wh_consts[1]
        dc = self.wh_consts[2]

        # Assume v_w = 0  ==>  ignore lateral movement of wheelchair
        # ==>  remove function/equation involving v_w from the model
        eq1 = omega3/self.Iz
        eq2 = ((-omega1*sin(q[4]) + omega2*cos(q[4]))/self.m) - 0.*q[0]*q[2]
        eq3 = ((-omega1*cos(q[4]) - omega2*sin(q[4]))/self.m) + q[0]*q[1]
        eq4 = q[1]*sin(q[4]) - 0.*q[2]*cos(q[4])
        eq5 = -q[1]*cos(q[4]) - 0.*q[2]*sin(q[4])
        eq6 = q[0]
        eq7 = (q[0]*(dl*cos(q[5]) - (df*sin(q[5])/2) - dc)/dc) + (-q[1]*sin(q[5])/dc)
        eq8 = (q[0]*(dl*cos(q[6]) + (df*sin(q[6])/2) - dc)/dc) + (-q[1]*sin(q[6])/dc)

        return [eq1, eq2, eq4, eq5, eq6, eq7, eq8]


    def ode_int(self):

        a_t = np.arange(0.0, self.move_time, 1./self.rate)
        # ini_val = [self.wheel_cmd.angular.z, -self.wheel_cmd.linear.x, 0.0, 0.0, 0.0, self.pi, self.pi]
        ini_val = [self.wheel_cmd.angular.z, -self.wheel_cmd.linear.x, -self.pose_y, self.pose_x, self.pose_th, self.angle_adj(self.r_caster_angle+self.pi), self.angle_adj(self.l_caster_angle+self.pi)]

        
        asol = odeint(self.solvr, ini_val, a_t)
        self.asol = asol
        

        print self.asol
        
             
    
    def calc_error(self):
        for i in xrange(min(len(self.solx), len(self.pose_x_data))):
            self.errorx.append(self.solx[i]-self.pose_x_data[i])

        for i in xrange(min(len(self.soly), len(self.pose_y_data))):
            self.errory.append(self.soly[i]-self.pose_y_data[i])

        for i in xrange(min(len(self.solth), len(self.pose_th_data))):
            self.errorth.append(self.solth[i]-self.pose_th_data[i])

        for i in xrange(min(len(self.soldel1), len(self.r_caster_data))):
            self.errordel1.append(self.soldel1[i]-self.r_caster_data[i])

        for i in xrange(min(len(self.soldel2), len(self.l_caster_data))):
            self.errordel2.append(self.soldel2[i]-self.l_caster_data[i])


    # def angle_adj(self, angle):
    #     angle = angle%self.two_pi
    #     angle = (angle+self.pi)%(self.two_pi)

    #     if angle > self.pi:
    #         angle -= self.two_pi
    #     return angle

    def delta(self, alpha):
        # return self.angle_adj(self.two_pi - (alpha%self.two_pi))
        return self.angle_adj(-alpha)

    def plot_data(self):
        # self.asol = self.fx()

        for i in xrange(len(self.asol)):
            self.solx.append(self.asol[i][3])
            self.soly.append(-self.asol[i][2])
            self.solth.append(self.angle_adj(self.asol[i][4])) 
            self.soldel1.append(self.angle_adj(self.asol[i][5]))
            self.soldel2.append(self.angle_adj(self.asol[i][6]))


        self.soldel1 = [self.angle_adj((angle+self.pi)) for angle in self.soldel1]
        self.soldel2 = [self.angle_adj((angle+self.pi)) for angle in self.soldel2]

        self.calc_error()


        plt.figure(1)
        plt.subplot(431)
        plt.title("Pose x (m)")
        xaxis = [x/self.rate for x in xrange(len(self.solx))]
        plt.plot(xaxis, self.solx, label="est")
        xaxis = [x/self.rate for x in xrange(len(self.pose_x_data))]
        plt.plot(xaxis, self.pose_x_data, label="actual")
        plt.legend()

        plt.subplot(432)
        plt.title("Orientation (rad)")
        xaxis = [x/self.rate for x in xrange(len(self.solth))]
        plt.plot(xaxis, self.solth, label="est")
        xaxis = [x/self.rate for x in xrange(len(self.pose_th_data))]
        plt.plot(xaxis, self.pose_th_data, label="actual")
        plt.legend()

        plt.subplot(433)
        plt.title("R Caster Orientation (rad)")
        xaxis = [x/self.rate for x in xrange(len(self.soldel1))]
        plt.plot(xaxis, self.soldel1, label="est")
        xaxis = [x/self.rate for x in xrange(len(self.r_caster_data))]
        plt.plot(xaxis, self.r_caster_data, label="actual")
        plt.legend()

        plt.subplot(434)
        plt.title("Error pose x (m)")
        xaxis = [x/self.rate for x in xrange(len(self.errorx))]
        plt.plot(xaxis, self.errorx)

        plt.subplot(435)
        plt.title("Error Orientation (rad)")
        xaxis = [x/self.rate for x in xrange(len(self.errorth))]
        plt.plot(xaxis, self.errorth)

        plt.subplot(436)
        plt.title("Error R Caster Orientation (rad)")
        xaxis = [x/self.rate for x in xrange(len(self.errordel1))]
        plt.plot(xaxis, self.errordel1)

        plt.subplot(437)
        plt.title("Pose y (m)")
        xaxis = [x/self.rate for x in xrange(len(self.soly))]
        plt.plot(xaxis, self.soly, label="est")
        xaxis = [x/self.rate for x in xrange(len(self.pose_y_data))]
        plt.plot(xaxis, self.pose_y_data, label="actual")
        plt.legend()


        plt.subplot(4,3,10)
        plt.title("Error pose y (m)")
        xaxis = [x/self.rate for x in xrange(len(self.errory))]
        plt.plot(xaxis, self.errory)


        plt.subplot(438)
        plt.title("L Caster Orientation (rad)")
        xaxis = [x/self.rate for x in xrange(len(self.soldel2))]
        plt.plot(xaxis, self.soldel2, label="est")
        xaxis = [x/self.rate for x in xrange(len(self.l_caster_data))]
        plt.plot(xaxis, self.l_caster_data, label="actual")
        plt.legend()

        plt.subplot(4,3,11)
        plt.title("Error L Caster Orientation (rad)")
        xaxis = [x/self.rate for x in xrange(len(self.errordel2))]
        plt.plot(xaxis, self.errordel2)

        plt.show()


    def shutdown(self):
        # Stop the robot when shutting down the node.
        rospy.loginfo("Stopping the robot...")
        self.pub_twist.publish(Twist())
        rospy.sleep(1)

if __name__ == '__main__':

    try:
        SolveDynamicModel2()
    except rospy.ROSInterruptException:
        pass