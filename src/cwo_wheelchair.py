#!/usr/bin/env python
import rospy
from nuric_wheelchair_model_02.msg import FloatArray
from geometry_msgs.msg import Twist
import numpy as np
from ukf_helper import al_to_th, rK2, th_to_al, normalize_angle

class CwoWheelchair:

    def __init__(self):
        # Initialize node
        rospy.init_node('cwo_wheelchair')

        # Set rospy to execute a shutdown function when exiting       
        rospy.on_shutdown(self.shutdown)

        self.wheel_cmd = Twist()

        # Control commands for wheelchair
        self.wheel_cmd.linear.x = -0.4
        self.wheel_cmd.angular.z = 0.5

        # time to move wheelchair 
        self.move_time = 5.0

        self.rate = 50
        self.dt = 1./self.rate

        self.al_to_th = al_to_th
        self.th_to_al = th_to_al
        self.rK2 = rK2

        self.l_caster_data = []
        self.r_caster_data = []

        # wheelchair constants
        self.wh_consts = [0.58, 0.19, 0.06]

        self.count = 0

        # Setup subscriber
        self.caster_joints = rospy.Subscriber('/caster_joints', FloatArray, self.caster_joints_callback)

        # Setup publisher
        self.pub_twist = rospy.Publisher('/cmd_vel', Twist, queue_size=20)

        self.r = rospy.Rate(self.rate)

        # Initialize (wait for caster joints to be subscribed)
        self.initialize()

        # Move wheelchair
        self.move_wheelchair()

        # save caster data
        self.save_data()
       

        
    def caster_joints_callback(self, caster_joints):
        
        self.l_caster_angle, self.r_caster_angle = caster_joints.data[0], caster_joints.data[1]


    def initialize(self):
        rospy.loginfo("Initializing...")        
        rospy.sleep(1)
        self.print_caster_joints()

    def move_wheelchair(self):
        

        while rospy.get_time() == 0.0:
            continue

        # set initial values for the kinematic model
        self.ini_val = [self.th_to_al(self.l_caster_angle), self.th_to_al(self.r_caster_angle)]

        start = rospy.get_time()
        count=0

        rospy.loginfo("Moving robot...")
        while (rospy.get_time() - start < self.move_time) and not rospy.is_shutdown():

            # save data for plotting
            self.l_caster_data.append(self.l_caster_angle)
            self.r_caster_data.append(self.r_caster_angle)
            self.pub_twist.publish(self.wheel_cmd)   

            count += 1

            self.r.sleep()

        # Stop the robot
        self.pub_twist.publish(Twist())
        self.print_caster_joints()
        self.count = count
        rospy.sleep(1)


    # solve kinematic model using ode2 function
    def solve_est(self):

        count=0

        x0 = np.array(self.ini_val)
        sol = np.reshape(x0, (1,2))


        while count < self.count-1:
            sol1 = self.ode2(x0, self.dt)
            x0 = sol1
            sol1 = np.reshape(sol1, (1,2))
            sol = np.append(sol, sol1, axis=0)

            count += 1

        return sol

    # save data to csv files 
    def save_data(self):

        np.savetxt('data_cwo.csv', np.c_[self.l_caster_data, self.r_caster_data])

        sol = self.solve_est()
        sol[:,0] = self.al_to_th(sol[:,0])
        sol[:,1] = self.al_to_th(sol[:,1])

        x0 = [normalize_angle(item) for item in sol[:,0].tolist()]
        x1 = [normalize_angle(item) for item in sol[:,1].tolist()]

        np.savetxt('data_est_cwo.csv', np.c_[x0, x1])

    
    # function to solve ODE (kinematic model for CWOs estimation)
    def ode2(self, x0, dt):

        self.dl = self.wh_consts[0]
        self.df = self.wh_consts[1]
        self.dc = self.wh_consts[2]

        x0 = np.array(x0)

        a, b = x0.tolist()

        def fa(a, b):
            return (self.wheel_cmd.angular.z*(self.dl*np.cos(a) - (self.df*np.sin(a)/2) - self.dc)/self.dc) - (-self.wheel_cmd.linear.x*np.sin(a)/self.dc)


        def fb(a, b):
            return (self.wheel_cmd.angular.z*(self.dl*np.cos(b) - (self.df*np.sin(b)/2) - self.dc)/self.dc) - (-self.wheel_cmd.linear.x*np.sin(b)/self.dc)


        return np.array(self.rK2(a, b, fa, fb, dt))

    

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
        CwoWheelchair()
    except rospy.ROSInterruptException:
        pass