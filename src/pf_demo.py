#!/usr/bin/env python

from pf_wheelchair import PF 
import rospy
import numpy as np 


class PFDemo():

    def __init__(self):

        rospy.init_node('pf_demo')

        self._num_particles = 20
        self._dt = .1
        self._consts = [0.58, 0.19, 0.06]
        self._motion_consts = [15.0, 5., 9.81/50., .01, .1, .2, .0, .58, .27*2, .0]
        self._alpha_var = [1.e2,1.e2]

        self.test_pf()

        rospy.on_shutdown(self.shutdown)


    def test_pf(self):

        mu_initial = [0.0, 0.0, 0.5, 0.0, 0.0, np.pi, np.pi]
        sigma_initial = np.diag([1.e-4,1.e-4,1.e-1,1.e-1,1.e-1,1.e1,1.e1])
        
        pf = PF(7, 3, mu_initial, sigma_initial, self._num_particles, self._dt, self._consts, self._motion_consts, self._alpha_var)

        pf.generate_particles()

        count=0
        while count < 3:
            pf.predict()

            mu_z = np.array([0.0, -0.5, 0.0])
            sig_z = np.diag([1.e-5,1.e-5,1.e-5])

            pf.update(mu_z, sig_z)

            pf.resample()

            count += 1

        print pf.Xt

        rospy.sleep(1)


    def shutdown(self):
        # Stop the robot when shutting down the node.
        rospy.loginfo("Stopping the robot...")
        # self.pub_twist.publish(Twist())
        # rospy.sleep(1)



if __name__ == '__main__':

    try:
        PFDemo()
    except rospy.ROSInterruptException:
        pass