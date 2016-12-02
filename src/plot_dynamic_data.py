#!/usr/bin/env python

import rospy
import sys
import numpy as np 
import matplotlib.pyplot as plt 

def plot_data():
    
    data = np.genfromtxt('/home/abhi/nuric_ws/src/nuric_wheelchair_model_02/src/data.csv', names=['x', 'y', 'th', 'l_caster', 'r_caster'])
    dataEst = np.genfromtxt('/home/abhi/nuric_ws/src/nuric_wheelchair_model_02/src/data_est.csv', names=['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6'])
    dataUkf = np.genfromtxt('/home/abhi/nuric_ws/src/nuric_wheelchair_model_02/src/data_ukf.csv', names=['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6'])


    plt.figure(1)
    plt.subplot(221)
    plt.plot(data['l_caster'], 'ro')
    plt.plot(dataEst['x5'], 'go')
    plt.plot(dataUkf['x5'], 'bo')

    plt.subplot(222)
    plt.plot(data['r_caster'], 'ro')
    plt.plot(dataEst['x6'], 'go')
    plt.plot(dataUkf['x6'], 'bo')

    # plt.subplot(223)
    # plt.plot(data['l_caster']-data2['x5'], 'bo')

    # plt.subplot(224)
    # plt.plot(data['r_caster']-data2['x6'], 'bo')

    # plt.figure(2)
    # plt.subplot(231)
    # plt.plot(data['x'], 'ro')
    # plt.plot(dataEst['x3'], 'go')

    # plt.subplot(232)
    # plt.plot(data['y'], 'ro')
    # plt.plot(dataEst['x2'], 'go')

    # plt.subplot(233)
    # plt.plot(data['th'], 'ro')
    # plt.plot(dataEst['x4'], 'go')


    # plt.subplot(234)
    # plt.plot(data['x']-dataEst['x3'], 'bo')

    # plt.subplot(235)
    # plt.plot(data['y']-dataEst['x2'], 'bo')

    # plt.subplot(236)
    # plt.plot(data['th']-dataEst['x4'], 'bo')



    plt.show()


if __name__ == '__main__':

    try:
        plot_data()
    except rospy.ROSInterruptException:
        pass