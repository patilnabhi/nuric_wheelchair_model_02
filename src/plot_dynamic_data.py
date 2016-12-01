#!/usr/bin/env python

import rospy
import sys
import numpy as np 
import matplotlib.pyplot as plt 

def plot_data():
    
    data = np.genfromtxt('/home/abhi/nuric_ws/src/nuric_wheelchair_model_02/src/data.csv', names=['l_caster', 'r_caster'])

    data2 = np.genfromtxt('/home/abhi/nuric_ws/src/nuric_wheelchair_model_02/src/data_est.csv', names=['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6'])


    

    plt.figure(1)
    plt.plot(data['l_caster'], 'ro')
    plt.plot(data2['x5'], 'go')

    plt.show()


if __name__ == '__main__':

    try:
        plot_data()
    except rospy.ROSInterruptException:
        pass