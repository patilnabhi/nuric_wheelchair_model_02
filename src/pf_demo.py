#!/usr/bin/env python

from pf_wheelchair import PF 
import rospy
import numpy as np 


class PFDemo():

    def __init__(self):

        rospy.init_node('pf_demo')

        self._num_particles = 2
        self._dt = 0.02
        self._consts = [0.58, 0.19, 0.06]
        self._motion_consts = [15.0, 5., 9.81/50., .01, .01, .2, .0, .58, .27*2, .0]
        self._alpha_var = [.001,.001]

        self.test_pf()

        rospy.on_shutdown(self.shutdown)


    def test_pf(self):

        mu_initial = [0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]
        sigma_initial = np.diag([0.1,0.1,0.1,0.1,0.1,0.1,0.1])
        
        pf = PF(7, 3, mu_initial, sigma_initial, self._num_particles, self._dt, self._consts, self._motion_consts, self._alpha_var)

        pf.generate_particles()

        pf.predict()

        mu_z = np.array([0.1, 0.1, 0.1])
        sig_z = np.diag([.01,.001,.1])

        pf.update(mu_z, sig_z)

        pf.resample()
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