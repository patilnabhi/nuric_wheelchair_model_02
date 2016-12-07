#!/usr/bin/env python

import numpy as np 
from numpy import array, asarray, isscalar, eye, dot
from scipy.linalg import inv, cholesky, eigvals

def dot3(A,B,C):
    return dot(A, dot(B,C))

def normalize_angle(x):
    x = x%(2*np.pi)
    if x > np.pi:
        x -= 2*np.pi 
    return x



def state_mean(sigmas, Wm):
    x = np.zeros(7)

    sum_sin1, sum_cos1 = 0., 0.
    sum_sin2, sum_cos2 = 0., 0.
    sum_sin3, sum_cos3 = 0., 0.

    for i in range(len(sigmas)):
        s = sigmas[i]
        x[0] += s[0] * Wm[i]
        x[1] += s[1] * Wm[i]
        x[2] += s[2] * Wm[i]
        x[3] += s[3] * Wm[i]        

        sum_sin1 += np.sin(s[4])*Wm[i]
        sum_cos1 += np.cos(s[4])*Wm[i]

        sum_sin2 += np.sin(s[5])*Wm[i]
        sum_cos2 += np.cos(s[5])*Wm[i]

        sum_sin3 += np.sin(s[6])*Wm[i]
        sum_cos3 += np.cos(s[6])*Wm[i]

        # x[5] += s[5] * Wm[i]
        # x[6] += s[6] * Wm[i]

    x[4] = np.arctan2(sum_sin1, sum_cos1)
    x[5] = np.arctan2(sum_sin2, sum_cos2)
    x[6] = np.arctan2(sum_sin3, sum_cos3)

    return x

def meas_mean(sigmas, Wm):
    z = np.zeros(3)

    sum_sin1, sum_cos1 = 0., 0.
    

    for i in range(len(sigmas)):
        s = sigmas[i]
        z[0] += s[0] * Wm[i]
        z[1] += s[1] * Wm[i]
                

        sum_sin1 += np.sin(s[2])*Wm[i]
        sum_cos1 += np.cos(s[2])*Wm[i]

        

    z[2] = np.arctan2(sum_sin1, sum_cos1)

    return z

def residual_x(a, b):
    y = np.zeros(7)
    
    for i in xrange(len(a)):
        y[:4] = a[:4] - b[:4]
        y[4] = sub_angle(a[4] - b[4])
        y[5] = sub_angle(a[5] - b[5])
        y[6] = sub_angle(a[6] - b[6])

    return y

def residual_z(a, b):
    y = np.zeros(3)
    
    for i in xrange(len(a)):
        y[:2] = a[:2] - b[:2]
        y[2] = sub_angle(a[2] - b[2])
        
    return y

def sub_angle(angle):
    if angle > np.pi:
        angle -= 2*np.pi
    return angle

class MerweScaledSigmaPoints(object):

    def __init__(self, n, alpha, beta, kappa, sqrt_method=None, subtract=None):
        

        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        if sqrt_method is None:
            self.sqrt = cholesky
        else:
            self.sqrt = sqrt_method

        if subtract is None:
            self.subtract= np.subtract
        else:
            self.subtract = subtract


    def num_sigmas(self):
        """ Number of sigma points for each variable in the state x"""
        return 2*self.n + 1


    def sigma_points(self, x, P):
        """ Computes the sigma points for an unscented Kalman filter
        given the mean (x) and covariance(P) of the filter.
        Returns tuple of the sigma points and weights.
        Works with both scalar and array inputs:
        sigma_points (5, 9, 2) # mean 5, covariance 9
        sigma_points ([5, 2], 9*eye(2), 2) # means 5 and 2, covariance 9I
        Parameters
        ----------
        X An array-like object of the means of length n
            Can be a scalar if 1D.
            examples: 1, [1,2], np.array([1,2])
        P : scalar, or np.array
           Covariance of the filter. If scalar, is treated as eye(n)*P.
        Returns
        -------
        sigmas : np.array, of size (n, 2n+1)
            Two dimensional array of sigma points. Each column contains all of
            the sigmas for one dimension in the problem space.
            Ordered by Xi_0, Xi_{1..n}, Xi_{n+1..2n}
        """

        assert self.n == np.size(x), "expected size {}, but size is {}".format(
            self.n, np.size(x))

        n = self.n

        if np.isscalar(x):
            x = np.asarray([x])

        if  np.isscalar(P):
            P = np.eye(n)*P
        else:
            P = np.asarray(P)

        lambda_ = self.alpha**2 * (n + self.kappa) - n
        
        matt = (lambda_ + n)*P
        # print eigvals(matt)
        U = self.sqrt(matt)

        sigmas = np.zeros((2*n+1, n))
        sigmas[0] = x
        for k in range(n):
            sigmas[k+1]   = self.subtract(x, -U[k])
            sigmas[n+k+1] = self.subtract(x, U[k])

        return sigmas


    def weights(self):
        """ Computes the weights for the scaled unscented Kalman filter.
        Returns
        -------
        Wm : ndarray[2n+1]
            weights for mean
        Wc : ndarray[2n+1]
            weights for the covariances
        """

        n = self.n
        lambda_ = self.alpha**2 * (n +self.kappa) - n

        c = .5 / (n + lambda_)
        Wc = np.full(2*n + 1, c)
        Wm = np.full(2*n + 1, c)
        Wc[0] = lambda_ / (n + lambda_) + (1 - self.alpha**2 + self.beta)
        Wm[0] = lambda_ / (n + lambda_)

        return Wm, Wc


""" Generates sigma points and weights according to Van der Merwe's
        2004 dissertation[1]. It parametizes the sigma points using
        alpha, beta, kappa terms, and is the version seen in most publications.
        Unless you know better, this should be your default choice.
        Parameters
        ----------
        n : int
            Dimensionality of the state. 2n+1 weights will be generated.
        alpha : float
            Determins the spread of the sigma points around the mean.
            Usually a small positive value (1e-3) according to [3].
        beta : float
            Incorporates prior knowledge of the distribution of the mean. For
            Gaussian x beta=2 is optimal, according to [3].
        kappa : float, default=0.0
            Secondary scaling parameter usually set to 0 according to [4],
            or to 3-n according to [5].
        sqrt_method : function(ndarray), default=scipy.linalg.cholesky
            Defines how we compute the square root of a matrix, which has
            no unique answer. Cholesky is the default choice due to its
            speed. Typically your alternative choice will be
            scipy.linalg.sqrtm. Different choices affect how the sigma points
            are arranged relative to the eigenvectors of the covariance matrix.
            Usually this will not matter to you; if so the default cholesky()
            yields maximal performance. As of van der Merwe's dissertation of
            2004 [6] this was not a well reseached area so I have no advice
            to give you.
            If your method returns a triangular matrix it must be upper
            triangular. Do not use numpy.linalg.cholesky - for historical
            reasons it returns a lower triangular matrix. The SciPy version
            does the right thing.
        subtract : callable (x, y), optional
            Function that computes the difference between x and y.
            You will have to supply this if your state variable cannot support
            subtraction, such as angles (359-1 degreees is 2, not 358). x and y
            are state vectors, not scalars.
        References
        ----------
        .. [1] R. Van der Merwe "Sigma-Point Kalman Filters for Probabilitic
               Inference in Dynamic State-Space Models" (Doctoral dissertation)
               
        """