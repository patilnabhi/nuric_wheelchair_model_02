#!/usr/bin/env python

import numpy as np 
from numpy import array, asarray, isscalar, eye, dot
from scipy.linalg import inv, cholesky, eigvals, sqrtm

def dot3(A,B,C):
    return dot(A, dot(B,C))

def normalize_angle(x):
    x = x%(2*np.pi)
    if x > np.pi:
        x -= 2*np.pi 
    # if x < -np.pi:
    #     x += 2*np.pi
    return x

def al_to_th(al):
    return al+np.pi

def th_to_al(th):
    return th-np.pi

# calculating mean to be used in unscented transform (ut.py)
def state_mean(sigmas, Wm):
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

# calculating mean to be used in unscented transform (ut.py)
def meas_mean(sigmas, Wm):
    z = np.zeros(3)

    z[0] = np.sum(np.dot(sigmas[:, 0], Wm))
    z[1] = np.sum(np.dot(sigmas[:, 1], Wm))

    sum_sin = np.sum(np.dot(np.sin(sigmas[:,2]), Wm))
    sum_cos = np.sum(np.dot(np.cos(sigmas[:,2]), Wm))

    z[2] = np.arctan2(sum_sin, sum_cos)

    return z

# calculating the residual gain values (motion model)
def residual_x(a, b):
    y = a - b

    y[4], y[5], y[6] = normalize_angle(y[4]), normalize_angle(y[5]), normalize_angle(y[6])

    return y 

# calculating the residual gain values (meas model)
def residual_z(a, b):
    y = a - b

    y[2] = normalize_angle(y[2])

    return y


# Fourth order Runge-Kutta for 2-dimensions (faster by factor of 3)
# hs = time-step
# (http://www.codeproject.com/Tips/792927/Fourth-Order-Runge-Kutta-Method-in-Python)

def rK2(a, b, fa, fb, hs):

    a1 = fa(a, b)*hs
    b1 = fb(a, b)*hs

    ak = a + a1*0.5
    bk = b + b1*0.5

    a2 = fa(ak, bk)*hs
    b2 = fb(ak, bk)*hs

    ak = a + a2*0.5
    bk = b + b2*0.5

    a3 = fa(ak, bk)*hs
    b3 = fb(ak, bk)*hs

    ak = a + a3
    bk = b + b3

    a4 = fa(ak, bk)*hs
    b4 = fb(ak, bk)*hs

    a = a + (a1 + 2*(a2+a3) + a4)/6
    b = b + (b1 + 2*(b2+b3) + b4)/6

    return [a, b]

# Fourth order Runge-Kutta for 7-dimensions (faster by factor of 3)
# hs = time-step
# (http://www.codeproject.com/Tips/792927/Fourth-Order-Runge-Kutta-Method-in-Python)

def rK7(a, b, c, d, e, f, g, fa, fb, fc, fd, fe, ff, fg, hs):

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


def rKN(x, fx, n, hs):
    k1 = []
    k2 = []
    k3 = []
    k4 = []
    xk = []
    for i in range(n):
        k1.append(fx[i](x)*hs)
    for i in range(n):
        xk.append(x[i] + k1[i]*0.5)
    for i in range(n):
        k2.append(fx[i](xk)*hs)
    for i in range(n):
        xk[i] = x[i] + k2[i]*0.5
    for i in range(n):
        k3.append(fx[i](xk)*hs)
    for i in range(n):
        xk[i] = x[i] + k3[i]
    for i in range(n):
        k4.append(fx[i](xk)*hs)
    for i in range(n):
        x[i] = x[i] + (k1[i] + 2*(k2[i] + k3[i]) + k4[i])/6
    return x


# Scaled Sigma Point algorithm by Julier
class JulierSigmaPoints(object):

    def __init__(self,n,  kappa, sqrt_method=None, subtract=None):
        """ Generates sigma points and weights according to Simon J. Julier
        and Jeffery K. Uhlmann's original paper []. It parametizes the sigma
        points using kappa.
        Parameters
        ----------
        n : int
            Dimensionality of the state. 2n+1 weights will be generated.
        kappa : float, default=0.
            Scaling factor that can reduce high order errors. kappa=0 gives
            the standard unscented filter. According to [Julier], if you set
            kappa to 3-dim_x for a Gaussian x you will minimize the fourth
            order errors in x and P.
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
        References
        ----------
        .. [1] Julier, Simon J.; Uhlmann, Jeffrey "A New Extension of the Kalman
            Filter to Nonlinear Systems". Proc. SPIE 3068, Signal Processing,
            Sensor Fusion, and Target Recognition VI, 182 (July 28, 1997)
       """

        self.n = n
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
        r""" Computes the sigma points for an unscented Kalman filter
        given the mean (x) and covariance(P) of the filter.
        kappa is an arbitrary constant. Returns sigma points.
        Works with both scalar and array inputs:
        sigma_points (5, 9, 2) # mean 5, covariance 9
        sigma_points ([5, 2], 9*eye(2), 2) # means 5 and 2, covariance 9I
        Parameters
        ----------
        X : array-like object of the means of length n
            Can be a scalar if 1D.
            examples: 1, [1,2], np.array([1,2])
        P : scalar, or np.array
           Covariance of the filter. If scalar, is treated as eye(n)*P.
        kappa : float
            Scaling factor.
        Returns
        -------
        sigmas : np.array, of size (n, 2n+1)
            2D array of sigma points :math:`\chi`. Each column contains all of
            the sigmas for one dimension in the problem space. They
            are ordered as:
            .. math::
                :nowrap:
                \begin{eqnarray}
                  \chi[0]    = &x \\
                  \chi[1..n] = &x + [\sqrt{(n+\kappa)P}]_k \\
                  \chi[n+1..2n] = &x - [\sqrt{(n+\kappa)P}]_k
                \end{eqnarray}
                
        """

        assert self.n == np.size(x)
        n = self.n

        if np.isscalar(x):
            x = np.asarray([x])

        n = np.size(x)  # dimension of problem

        if np.isscalar(P):
            P = np.eye(n)*P

        sigmas = np.zeros((2*n+1, n))

        # implements U'*U = (n+kappa)*P. Returns lower triangular matrix.
        # Take transpose so we can access with U[i]
        U = self.sqrt((n + self.kappa) * P)

        sigmas[0] = x
        for k in range(n):
            sigmas[k+1]   = self.subtract(x, -U[k])
            sigmas[n+k+1] = self.subtract(x, U[k])
        return sigmas


    def weights(self):
        """ Computes the weights for the unscented Kalman filter. In this
        formulatyion the weights for the mean and covariance are the same.
        Returns
        -------
        Wm : ndarray[2n+1]
            weights for mean
        Wc : ndarray[2n+1]
            weights for the covariances
        """
        
        n = self.n
        k = self.kappa

        W = np.full(2*n+1, .5 / (n + k))
        W[0] = k / (n+k)
        return W, W


class SimplexSigmaPoints(object):

    def __init__(self, n, alpha=1, sqrt_method=None, subtract=None):
        """ Generates sigma points and weights according to the simplex 
        method presented in [1] DOI: 10.1051/cocv/2010006
        Parameters
        ----------
        n : int
            Dimensionality of the state. n+1 weights will be generated.
        sqrt_method : function(ndarray), default=scipy.linalg.cholesky
            Defines how we compute the square root of a matrix, which has
            no unique answer. Cholesky is the default choice due to its
            speed. Typically your alternative choice will be
            scipy.linalg.sqrtm
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
        .. [1] Phillippe Moireau and Dominique Chapelle "Reduced-Order Unscented
        Kalman Filtering with Application to Parameter Identification in
        Large-Dimensional Systems"
        """

        self.n = n
        self.alpha = alpha
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
        return self.n + 1


    def sigma_points(self, x, P):
        """ Computes the implex sigma points for an unscented Kalman filter
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
        sigmas : np.array, of size (n, n+1)
            Two dimensional array of sigma points. Each column contains all of
            the sigmas for one dimension in the problem space.
            Ordered by Xi_0, Xi_{1..n}
        """

        assert self.n == np.size(x), "expected size {}, but size is {}".format(
            self.n, np.size(x))

        n = self.n

        if np.isscalar(x):
            x = np.asarray([x])
        x = x.reshape(-1, 1)
        if np.isscalar(P):
            P = np.eye(n)*P
        else:
            P = np.asarray(P)

        U = self.sqrt(P)

        lambda_ = n / (n + 1)
        Istar = np.array([[-1/np.sqrt(2*lambda_), 1/np.sqrt(2*lambda_)]])
        for d in range(2, n+1):
            row = np.ones((1, Istar.shape[1] + 1)) * 1. / np.sqrt(lambda_*d*(d + 1))
            row[0, -1] = -d / np.sqrt(lambda_ * d * (d + 1))
            Istar = np.r_[np.c_[Istar, np.zeros((Istar.shape[0]))], row]

        I = np.sqrt(n)*Istar
        scaled_unitary = U.dot(I)

        sigmas = self.subtract(x, -scaled_unitary)
        return sigmas.T


    def weights(self):
        """ Computes the weights for the scaled unscented Kalman filter.
        Returns
        -------
        Wm : ndarray[n+1]
            weights for mean
        Wc : ndarray[n+1]
            weights for the covariances
        """

        n = self.n
        c = 1. / (n + 1)
        W = np.full(n + 1, c)

        return W, W
        

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