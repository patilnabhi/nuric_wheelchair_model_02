#!/usr/bin/env python
import math
import numpy as np 
from numpy import eye, zeros, dot, isscalar, outer
from scipy.linalg import inv, cholesky, sqrtm
from ut import unscented_transform
from ukf_helper import dot3

# UKF algorithm implementation


class UKF(object):

    def __init__(self, dim_x, dim_z, dt, hx, fx, points, sqrt_fn=None, x_mean_fn=None, z_mean_fn=None, residual_x=None, residual_z=None):

        self.Q = eye(dim_x)     # motion model noise Q
        self.R = eye(dim_z)     # measurement noise matrix
        self.x = zeros(dim_x)   # mean values of state
        self.P = eye(dim_x)     # state / motion model noise matrix
        self._dim_x = dim_x
        self._dim_z = dim_z
        self.points_fn = points
        self._dt = dt
        self._num_sigmas = points.num_sigmas()
        self.hx = hx        # measurement function
        self.fx = fx        # motion model function
        self.x_mean = x_mean_fn
        self.z_mean = z_mean_fn
        self.log_likelihood = 0.0

        if sqrt_fn is None:
            self.msqrt = cholesky
        else:
            self.msqrt = sqrt_fn

        self.Wm, self.Wc = self.points_fn.weights()

        if residual_x is None:
            self.residual_x = np.subtract
        else:
            self.residual_x = residual_x

        if residual_z is None:
            self.residual_z = np.subtract
        else:
            self.residual_z = residual_z


        self.sigmas_f = zeros((self._num_sigmas, self._dim_x))
        self.sigmas_h = zeros((self._num_sigmas, self._dim_z))




    def predict(self, UT=None, fx_args=()):

        dt = self._dt

        if not isinstance(fx_args, tuple):
            fx_args = (fx_args,)

        if UT is None:
            UT = unscented_transform

        sigmas = self.points_fn.sigma_points(self.x, self.P)

        for i in xrange(self._num_sigmas):
            self.sigmas_f[i] = self.fx(sigmas[i], dt, *fx_args)
        
        self.x, self.P = UT(self.sigmas_f, self.Wm, self.Wc, self.Q, self.x_mean, self.residual_x)
        # print self.x


    def update(self, z, R=None, UT=None, hx_args=()):

        if z is None:
            return

        if not isinstance(hx_args, tuple):
            hx_args = (hx_args,)

        if UT is None:
            UT = unscented_transform

        R = self.R

        for i in xrange(self._num_sigmas):
            self.sigmas_h[i] = self.hx(self.sigmas_f[i], *hx_args)

        zp, Pz = UT(self.sigmas_h, self.Wm, self.Wc, R, self.z_mean, self.residual_z)

        Pxz = zeros((self._dim_x, self._dim_z))
        for i in xrange(self._num_sigmas):
            dx = self.residual_x(self.sigmas_f[i], self.x)
            dz = self.residual_z(self.sigmas_h[i], zp)
            Pxz += self.Wc[i] * outer(dx, dz)


        self.K = dot(Pxz, inv(Pz))
        self.y = self.residual_z(z, zp)

        self.x = self.x + dot(self.K, self.y)
        self.P = self.P - dot3(self.K, Pz, self.K.T)

        # print zp

        # self.log_likelihood = multivariate_normal.logpdf(x=self.y, mean=np.zeros(len(self.y)), cov=Pz, allow_singular=True)


    def batch_filter(self, zs, Rs=None, UT=None):
        """ Performs the UKF filter over the list of measurement in `zs`.
        Parameters
        ----------
        zs : list-like
            list of measurements at each time step `self._dt` Missing
            measurements must be represented by 'None'.
        Rs : list-like, optional
            optional list of values to use for the measurement error
            covariance; a value of None in any position will cause the filter
            to use `self.R` for that time step.
        UT : function(sigmas, Wm, Wc, noise_cov), optional
            Optional function to compute the unscented transform for the sigma
            points passed through hx. Typically the default function will
            work - you can use x_mean_fn and z_mean_fn to alter the behavior
            of the unscented transform.
        Returns
        -------
        means: ndarray((n,dim_x,1))
            array of the state for each time step after the update. Each entry
            is an np.array. In other words `means[k,:]` is the state at step
            `k`.
        covariance: ndarray((n,dim_x,dim_x))
            array of the covariances for each time step after the update.
            In other words `covariance[k,:,:]` is the covariance at step `k`.
        """

        try:
            z = zs[0]
        except:
            assert not isscalar(zs), 'zs must be list-like'

        if self._dim_z == 1:
            assert isscalar(z) or (z.ndim==1 and len(z) == 1), \
            'zs must be a list of scalars or 1D, 1 element arrays'

        else:
            assert len(z) == self._dim_z, 'each element in zs must be a' \
            '1D array of length {}'.format(self._dim_z)

        z_n = np.size(zs, 0)
        if Rs is None:
            Rs = [None] * z_n

        # mean estimates from Kalman Filter
        if self.x.ndim == 1:
            means = zeros((z_n, self._dim_x))
        else:
            means = zeros((z_n, self._dim_x, 1))


        # state covariances from Kalman Filter
        covariances = zeros((z_n, self._dim_x, self._dim_x))

        for i, (z, r) in enumerate(zip(zs, Rs)):
            self.predict(UT=UT)
            self.update(z, r, UT=UT)
            means[i,:]         = self.x
            covariances[i,:,:] = self.P

        return (means, covariances)






    
