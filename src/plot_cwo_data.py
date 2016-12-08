#!/usr/bin/env python

import rospy
import sys
import numpy as np 
import matplotlib.pyplot as plt 

def plot_cwo_data():
    
    data = np.genfromtxt('data_cwo.csv', names=['l_caster', 'r_caster'])
    dataEst = np.genfromtxt('data_est_cwo.csv', names=['l_caster', 'r_caster'])


    line_width = 3.0
    alpha_value = 0.6

    plt.figure(1)
    plt.subplot(221)
    plt.title("left-CWO (rad)")
    plt.plot(data['l_caster'], linewidth=line_width, alpha=alpha_value, label='Actual')
    plt.plot(dataEst['l_caster'], linewidth=line_width, alpha=alpha_value, label='Estimated')
    plt.legend()

    plt.subplot(222)
    plt.title("right-CWO (rad)")
    plt.plot(data['r_caster'], linewidth=line_width, alpha=alpha_value, label='Actual')
    plt.plot(dataEst['r_caster'], linewidth=line_width, alpha=alpha_value, label='Estimated')
    plt.legend()

    plt.subplot(223)
    plt.title("Error between actual and estimated left-CWO (rad)")
    plt.plot(data['l_caster']-dataEst['l_caster'], linewidth=line_width, alpha=alpha_value)

    plt.subplot(224)
    plt.title("Error between actual and estimated right-CWO (rad)")
    plt.plot(data['r_caster']-dataEst['r_caster'],  linewidth=line_width, alpha=alpha_value)

    

    plt.show()


if __name__ == '__main__':

    try:
        plot_cwo_data()
    except rospy.ROSInterruptException:
        pass