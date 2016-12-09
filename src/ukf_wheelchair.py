#!/usr/bin/env python
import rospy
from ukf_helper import MerweScaledSigmaPoints, SimplexSigmaPoints, JulierSigmaPoints, state_mean, meas_mean, residual_x, residual_z, normalize_angle, rK7, al_to_th, th_to_al
from ukf import UKF
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from nuric_wheelchair_model_02.msg import FloatArray
from tf.transformations import euler_from_quaternion
import numpy as np


class UKFWheelchair(object):

    def __init__(self):
        rospy.init_node('ukf_wheelchair')
        rospy.on_shutdown(self.shutdown)

        self.wheel_cmd = Twist()

        self.wheel_cmd.linear.x = -0.4 
        self.wheel_cmd.angular.z = 0.5


        self.move_time = 6.0
        self.rate = 50
        self.dt = 1.0/self.rate

        self.zs = []
        self.xs = []

        self.al_to_th = al_to_th
        self.th_to_al = th_to_al
        self.state_mean = state_mean
        self.meas_mean = meas_mean
        self.residual_x = residual_x
        self.residual_z = residual_z
        self.rK7 = rK7  # Runge-kutta method to solve ode with 7 variables

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

        # wheelchair constants
        self.wh_consts = [0.58, 0.19, 0.06]

        self.odom_data = rospy.Subscriber('/odom', Odometry, self.odom_cb)
        self.caster_data = rospy.Subscriber('/caster_joints', FloatArray, self.caster_cb)
        self.pub_twist = rospy.Publisher('/cmd_vel', Twist, queue_size=20)

        self.r = rospy.Rate(self.rate)

        self.move_wheelchair()
        self.save_data()


    def caster_cb(self, caster_joints):       
        self.l_caster_angle, self.r_caster_angle = caster_joints.data[0], caster_joints.data[1]

    def odom_cb(self, odom_data):
        (_,_,yaw) = euler_from_quaternion([odom_data.pose.pose.orientation.x, odom_data.pose.pose.orientation.y, odom_data.pose.pose.orientation.z, odom_data.pose.pose.orientation.w])
        self.odom_x, self.odom_y, self.odom_th = odom_data.pose.pose.position.x, odom_data.pose.pose.position.y, yaw


    def move_wheelchair(self):

        self.ini_cwo_l = 2*np.pi*np.random.random_sample() * -np.pi
        self.ini_cwo_r = 2*np.pi*np.random.random_sample() * -np.pi

        while rospy.get_time() == 0.0:
            continue
        
        rospy.sleep(1)
        
        # self.ini_val = [self.wheel_cmd.angular.z, -self.wheel_cmd.linear.x, -self.odom_y, self.odom_x, self.odom_th, self.th_to_al(self.l_caster_angle), self.th_to_al(self.r_caster_angle)]
        self.ini_val = [0.0, 0.0, 0.0, 0.0, 0.0, self.ini_cwo_l, self.ini_cwo_r]
        # self.ini_val = np.random.uniform(low=-1.0, high=1.0, size=(7,)).tolist()

        # UKF initialization
        def fx(x, dt):

            sol = self.ode2(x)
            return np.array(sol)

        def hx(x):
            return np.array([x[3], x[2], normalize_angle(x[4])])


        # points = MerweScaledSigmaPoints(n=7, alpha=.00001, beta=2., kappa=-4.)
        points = JulierSigmaPoints(n=7, kappa=-4., sqrt_method=None)
        # points = SimplexSigmaPoints(n=7)
        kf = UKF(dim_x=7, dim_z=3, dt=self.dt, fx=fx, hx=hx, points=points, sqrt_fn=None, x_mean_fn=self.state_mean, z_mean_fn=self.meas_mean, residual_x=self.residual_x, residual_z=self.residual_z)

        x0 = np.array(self.ini_val)

        kf.x = x0   # initial mean state
        kf.Q *= np.diag([.0001, .0001, .0001, .0001, .0001, .01, .01])
        kf.P *= 0.000001  # kf.P = eye(dim_x) ; adjust covariances if necessary
        kf.R *= 0.0001

        count = 0
        rospy.loginfo("Moving robot...")
        start = rospy.get_time()
        self.r.sleep()
        
        while (rospy.get_time() - start < self.move_time) and not rospy.is_shutdown():
            
            self.pub_twist.publish(self.wheel_cmd) 

            z = np.array([self.odom_x, -self.odom_y, self.odom_th])           
            self.zs.append(z)

            kf.predict()
            kf.update(z)

            self.xs.append(kf.x)

            self.pose_x_data.append(self.odom_x)
            self.pose_y_data.append(self.odom_y)
            self.pose_th_data.append(self.odom_th)
            self.l_caster_data.append(self.l_caster_angle)
            self.r_caster_data.append(self.r_caster_angle)

            count += 1
            self.r.sleep()


        # Stop the robot
        self.pub_twist.publish(Twist())
        self.xs = np.array(self.xs)
        rospy.sleep(1)


    def solve_est(self):
        count=0

        x0=np.array(self.ini_val)
        sol = np.reshape(x0, (1,7))

        while count < int(self.rate*self.move_time):

            sol1 = self.ode2(x0)
            x0 = sol1
            sol1 = np.reshape(sol1, (1,7))
            sol = np.append(sol, sol1, axis=0)
            
            count += 1

        return sol


    # saving data to csv files
    def save_data(self):

        rospy.loginfo("Saving data...")

        np.savetxt('data.csv', np.c_[self.pose_x_data, self.pose_y_data, self.pose_th_data, self.l_caster_data, self.r_caster_data])

        ukf_data = self.xs
        ukf_data[:,5] = self.al_to_th(ukf_data[:,5])
        ukf_data[:,6] = self.al_to_th(ukf_data[:,6])
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
            return self._omega3/self.Iz

        def fb(a, b, c, d, e, f, g):
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


    
    # calculating omegas from the dynamic motion model
    ef omegas(self, delta1, delta2):

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
        return -alpha

    

    def shutdown(self):
        # Stop the robot when shutting down the node.
        rospy.loginfo("Shutting node...")
        self.pub_twist.publish(Twist())
        rospy.sleep(1)


if __name__ == '__main__':

    try:
        UKFWheelchair()
    except rospy.ROSInterruptException:
        pass