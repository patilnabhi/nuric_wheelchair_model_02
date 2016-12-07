#!/usr/bin/env python

import rospy
import sys
from ukf_helper import MerweScaledSigmaPoints, SimplexSigmaPoints, JulierSigmaPoints, state_mean, meas_mean, residual_x, residual_z, normalize_angle, rKN, sub_angle
from ukf import UKF
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from nuric_wheelchair_model_02.msg import FloatArray
from tf.transformations import euler_from_quaternion
import numpy as np
from scipy.integrate import odeint, ode
from math import sin, cos
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm


class UKFWheelchair3(object):

    def __init__(self):
        rospy.init_node('ukf_wheelchair3')

        rospy.on_shutdown(self.shutdown)

        self.wheel_cmd = Twist()

        self.wheel_cmd.linear.x = 0.4 
        self.wheel_cmd.angular.z = 0.2


        self.move_time = 6.0
        self.rate = 100
        self.dt = 1.0/self.rate

        self.zs = []

        # constants for ode equations
        # (approximations)
        self.Iz = 15.0
        self.mu = .01
        self.ep = 0.2
        self.m = 5.0
        self.g = 9.81/50.


        self.pose_x_data = []
        self.pose_y_data = []
        self.pose_th_data = []
        self.l_caster_data = []
        self.r_caster_data = []

        self.asol = []

        # wheelchair constants
        self.wh_consts = [0.58, 0.19, 0.06]

        self.odom_data = rospy.Subscriber('/odom', Odometry, self.odom_cb)
        self.caster_data = rospy.Subscriber('/caster_joints', FloatArray, self.caster_cb)
        self.pub_twist = rospy.Publisher('/cmd_vel', Twist, queue_size=20)

        self.r = rospy.Rate(self.rate)

        self.move_wheelchair()

        self.save_data()

        # self.plot_data()



    def caster_cb(self, caster_joints):       
        self.l_caster_angle, self.r_caster_angle = caster_joints.data[0], caster_joints.data[1]

    def odom_cb(self, odom_data):
        # self.odom_vx, self.odom_vth = odom_data.twist.twist.linear.x, odom_data.twist.twist.angular.z
        (_,_,yaw) = euler_from_quaternion([odom_data.pose.pose.orientation.x, odom_data.pose.pose.orientation.y, odom_data.pose.pose.orientation.z, odom_data.pose.pose.orientation.w])
        self.odom_x, self.odom_y, self.odom_th = odom_data.pose.pose.position.x, odom_data.pose.pose.position.y, yaw


    def move_wheelchair(self):

        # self.wheel_cmd.linear.x = 1.*np.random.random_sample() * -.5 
        # self.wheel_cmd.angular.z = .6*np.random.random_sample() * -.3 

        self.ini_cwo_l = 2*np.pi*np.random.random_sample() * -np.pi
        self.ini_cwo_r = 2*np.pi*np.random.random_sample() * -np.pi

        while rospy.get_time() == 0.0:
            continue
        

        rospy.sleep(1)

        
        # self.ini_val = [self.wheel_cmd.angular.z, -self.wheel_cmd.linear.x, -self.odom_y, self.odom_x, self.odom_th, self.th_to_al(self.l_caster_angle), self.th_to_al(self.r_caster_angle)]
        self.ini_val = [0.0, 0.0, 0.0, 0.0, 0.0, self.ini_cwo_l, self.ini_cwo_r]
        # self.ini_val = np.random.uniform(low=-1.0, high=1.0, size=(7,)).tolist()

        count = 0

        rospy.loginfo("Moving robot...")

        start = rospy.get_time()
        self.r.sleep()
        
        while (rospy.get_time() - start < self.move_time) and not rospy.is_shutdown():
            
            z = np.array([self.odom_x, -self.odom_y, self.odom_th])

            self.pub_twist.publish(self.wheel_cmd)
            
            self.zs.append(z)

            self.pose_x_data.append(self.odom_x)
            self.pose_y_data.append(self.odom_y)
            self.pose_th_data.append(self.odom_th)
            self.l_caster_data.append(self.l_caster_angle)
            self.r_caster_data.append(self.r_caster_angle)

            count += 1
            self.r.sleep()


        # Stop the robot
        self.pub_twist.publish(Twist())


        rospy.sleep(1)

    def solve_ukf(self):

        def fx(x, dt):

            # x[4], x[5], x[6] = x[4], normalize_angle(x[5]), normalize_angle(x[6])
            sol = self.ode2(x)
            # sol = self.ode_int(x)
            return np.array(sol)

        def hx(x):
            return np.array([x[3], x[2], normalize_angle(x[4])])


        # points = MerweScaledSigmaPoints(n=7, alpha=.00001, beta=2., kappa=-4.)
        points = JulierSigmaPoints(n=7, kappa=-4., sqrt_method=None)
        # points = SimplexSigmaPoints(n=7)
        kf = UKF(dim_x=7, dim_z=3, dt=self.dt, fx=fx, hx=hx, points=points, sqrt_fn=None, x_mean_fn=self.state_mean, z_mean_fn=self.meas_mean, residual_x=self.residual_x, residual_z=self.residual_z)

        x0 = np.array(self.ini_val)
        # x0 = np.reshape(x0, (1,7))

        kf.x = x0   # initial mean state
        kf.Q *= np.diag([.0001, .0001, .0001, .0001, .0001, .01, .01])
        kf.P *= 0.000001  # kf.P = eye(dim_x) ; adjust covariances if necessary
        kf.R *= 0.0001
        

        Ms, Ps = kf.batch_filter(self.zs)
        Ms[:,5] = self.al_to_th(Ms[:,5])
        Ms[:,6] = self.al_to_th(Ms[:,6])

        return Ms


    def solve_est(self):
        count=0

        x0=np.array(self.ini_val)
        # x0 = np.reshape(x0, (1,7))
        sol = np.reshape(x0, (1,7))

        while count < int(self.rate*self.move_time):

            # sol1 = self.ode_int(x0)
            sol1 = self.ode2(x0)
            x0 = sol1
            sol1 = np.reshape(sol1, (1,7))
            sol = np.append(sol, sol1, axis=0)
            # x0 = sol1
            
            count += 1

        return sol



    def save_data(self):

        rospy.loginfo("Saving data...")

        np.savetxt('data.csv', np.c_[self.pose_x_data, self.pose_y_data, self.pose_th_data, self.l_caster_data, self.r_caster_data])

        ukf_data = self.solve_ukf()
        x0 = [item for item in ukf_data[:,0].tolist()]
        x1 = [item for item in ukf_data[:,1].tolist()]
        x2 = [-item for item in ukf_data[:,2].tolist()]
        x3 = [item for item in ukf_data[:,3].tolist()]
        x4 = [normalize_angle(item) for item in ukf_data[:,4].tolist()]
        x5 = [normalize_angle(item) for item in ukf_data[:,5].tolist()]
        x6 = [normalize_angle(item) for item in ukf_data[:,6].tolist()]
        np.savetxt('data_ukf.csv', np.c_[x0,x1,x2,x3,x4,x5,x6])

        sol = self.solve_est()
        sol[:,5] = self.al_to_th(sol[:,5])
        sol[:,6] = self.al_to_th(sol[:,6])

        x00 = [item for item in sol[:,0].tolist()]
        x11 = [item for item in sol[:,1].tolist()]
        x22 = [-item for item in sol[:,2].tolist()]
        x33 = [item for item in sol[:,3].tolist()]
        x44 = [normalize_angle(item) for item in sol[:,4].tolist()]
        x55 = [normalize_angle(item) for item in sol[:,5].tolist()]
        x66 = [normalize_angle(item) for item in sol[:,6].tolist()]



        np.savetxt('data_est.csv', np.c_[x00,x11,x22,x33,x44,x55,x66])

    def ode2(self, x0):

        x0 = np.array(x0)

        a, b, c, d, e, f, g = x0.tolist()
        
        self._dl = self.wh_consts[0]
        self._df = self.wh_consts[1]
        self._dc = self.wh_consts[2]

        self._omega1 = self.omegas(self.delta(f),self.delta(g))[0]
        self._omega2 = self.omegas(self.delta(f),self.delta(g))[1]
        self._omega3 = self.omegas(self.delta(f),self.delta(g))[2]

        def fa(a, b, c, d, e, f, g):
            # omega3 = self.omegas(self.delta(f),self.delta(g), self._i)[2]
            return self._omega3/self.Iz

        def fb(a, b, c, d, e, f, g):
            # omega1 = self.omegas(self.delta(f),self.delta(g), self._i)[0]
            # omega2 = self.omegas(self.delta(f),self.delta(g), self._i)[1]
            return ((-self._omega1*np.sin(e) + self._omega2*np.cos(e))/self.m)

        def fc(a, b, c, d, e, f, g):
            return b*np.sin(e)

        def fd(a, b, c, d, e, f, g):
            return -b*np.cos(e)

        def fe(a, b, c, d, e, f, g):
            return a

        def ff(a, b, c, d, e, f, g):
            return (a*(self._dl*np.cos(f) - (self._df*np.sin(f)/2) - self._dc)/self._dc) - (b*np.sin(f)/self._dc)

        def fg(a, b, c, d, e, f, g):
            return (a*(self._dl*np.cos(g) + (self._df*np.sin(g)/2) - self._dc)/self._dc) - (b*np.sin(g)/self._dc)

        return np.array(self.rK7(a, b, c, d, e, f, g, fa, fb, fc, fd, fe, ff, fg, self.dt))


    def rK7(self, a, b, c, d, e, f, g, fa, fb, fc, fd, fe, ff, fg, hs):

        a1 = fa(a, b, c, d, e, f, g)*hs
        b1 = fb(a, b, c, d, e, f, g)*hs
        c1 = fc(a, b, c, d, e, f, g)*hs
        d1 = fd(a, b, c, d, e, f, g)*hs
        e1 = fe(a, b, c, d, e, f, g)*hs
        f1 = ff(a, b, c, d, e, f, g)*hs
        g1 = fg(a, b, c, d, e, f, g)*hs

        ak = a + a1*0.5
        bk = b + b1*0.5
        ck = c + c1*0.5
        dk = d + d1*0.5
        ek = e + e1*0.5
        fk = f + f1*0.5
        gk = g + g1*0.5

        a2 = fa(ak, bk, ck, dk, ek, fk, gk)*hs
        b2 = fb(ak, bk, ck, dk, ek, fk, gk)*hs
        c2 = fc(ak, bk, ck, dk, ek, fk, gk)*hs
        d2 = fd(ak, bk, ck, dk, ek, fk, gk)*hs
        e2 = fe(ak, bk, ck, dk, ek, fk, gk)*hs
        f2 = ff(ak, bk, ck, dk, ek, fk, gk)*hs
        g2 = fg(ak, bk, ck, dk, ek, fk, gk)*hs

        ak = a + a2*0.5
        bk = b + b2*0.5
        ck = c + c2*0.5
        dk = d + d2*0.5
        ek = e + e2*0.5
        fk = f + f2*0.5
        gk = g + g2*0.5

        a3 = fa(ak, bk, ck, dk, ek, fk, gk)*hs
        b3 = fb(ak, bk, ck, dk, ek, fk, gk)*hs
        c3 = fc(ak, bk, ck, dk, ek, fk, gk)*hs
        d3 = fd(ak, bk, ck, dk, ek, fk, gk)*hs
        e3 = fe(ak, bk, ck, dk, ek, fk, gk)*hs
        f3 = ff(ak, bk, ck, dk, ek, fk, gk)*hs
        g3 = fg(ak, bk, ck, dk, ek, fk, gk)*hs

        ak = a + a3
        bk = b + b3
        ck = c + c3
        dk = d + d3
        ek = e + e3
        fk = f + f3
        gk = g + g3

        a4 = fa(ak, bk, ck, dk, ek, fk, gk)*hs
        b4 = fb(ak, bk, ck, dk, ek, fk, gk)*hs
        c4 = fc(ak, bk, ck, dk, ek, fk, gk)*hs
        d4 = fd(ak, bk, ck, dk, ek, fk, gk)*hs
        e4 = fe(ak, bk, ck, dk, ek, fk, gk)*hs
        f4 = ff(ak, bk, ck, dk, ek, fk, gk)*hs
        g4 = fg(ak, bk, ck, dk, ek, fk, gk)*hs

        a = a + (a1 + 2*(a2 + a3) + a4)/6
        b = b + (b1 + 2*(b2 + b3) + b4)/6
        c = c + (c1 + 2*(c2 + c3) + c4)/6
        d = d + (d1 + 2*(d2 + d3) + d4)/6
        e = e + (e1 + 2*(e2 + e3) + e4)/6
        f = f + (f1 + 2*(f2 + f3) + f4)/6
        g = g + (g1 + 2*(g2 + g3) + g4)/6

        return [a, b, c, d, e, f, g]


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
        eq7 = (thdot*(dl*cos(alpha1) - (df*sin(alpha1)/2) - dc)/dc) - (ydot*sin(alpha1)/dc)
        eq8 = (thdot*(dl*cos(alpha2) + (df*sin(alpha2)/2) - dc)/dc) - (ydot*sin(alpha2)/dc)

        f = [eq1, eq2, eq4, eq5, eq6, eq7, eq8]

        return f


    def ode_int(self, x0):
        solver = ode(self.fun)
        solver.set_integrator('dop853')

        t0 = 0.0
        x0 = np.reshape(x0, (7,))
        x0 = x0.tolist()
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

        solf = sol[-1]

        # solf = np.reshape(solf, (1,7))

        return solf


    def th_to_al(self, th):
        return th-np.pi 

    def al_to_th(self, al):
        return al+np.pi

    def delta(self, alpha):
        return -alpha



    def state_mean(self, sigmas, Wm):
        x = np.zeros(7)

        sum_sin4 = np.sum(np.dot(np.sin(sigmas[:,4]), Wm))
        sum_cos4 = np.sum(np.dot(np.cos(sigmas[:,4]), Wm))
        sum_sin5 = np.sum(np.dot(np.sin(sigmas[:,5]), Wm))
        sum_cos5 = np.sum(np.dot(np.cos(sigmas[:,5]), Wm))
        sum_sin6 = np.sum(np.dot(np.sin(sigmas[:,6]), Wm))
        sum_cos6 = np.sum(np.dot(np.cos(sigmas[:,6]), Wm))

        x[0] = np.sum(np.dot(sigmas[:, 0], Wm))
        x[1] = np.sum(np.dot(sigmas[:, 1], Wm))
        x[2] = np.sum(np.dot(sigmas[:, 2], Wm))
        x[3] = np.sum(np.dot(sigmas[:, 3], Wm))
        x[4] = np.arctan2(sum_sin4, sum_cos4)
        x[5] = np.arctan2(sum_sin5, sum_cos5)
        x[6] = np.arctan2(sum_sin6, sum_cos6)


        return x

    def meas_mean(self, sigmas, Wm):
        z = np.zeros(3)

        z[0] = np.sum(np.dot(sigmas[:, 0], Wm))
        z[1] = np.sum(np.dot(sigmas[:, 1], Wm))

        sum_sin = np.sum(np.dot(np.sin(sigmas[:,2]), Wm))
        sum_cos = np.sum(np.dot(np.cos(sigmas[:,2]), Wm))

        z[2] = np.arctan2(sum_sin, sum_cos)

        return z

    def residual_x(self, a, b):
        y = a - b

        y[4], y[5], y[6] = normalize_angle(y[4]), normalize_angle(y[5]), normalize_angle(y[6])

        return y 

    def residual_z(self, a, b):
        y = a - b

        y[2] = normalize_angle(y[2])

        return y


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