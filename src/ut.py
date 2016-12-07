#!/usr/bin/env python

import numpy as np 

def unscented_transform(sigmas, Wm, Wc, noise_cov=None, mean_fn=None, residual_fn=None):

    kmax, n = sigmas.shape

    x = mean_fn(sigmas, Wm)

    P = np.zeros((n,n))
    for k in xrange(kmax):
        y = residual_fn(sigmas[k], x)
        P += Wc[k] * np.outer(y, y)

    if noise_cov is not None:
        P += noise_cov

    return (x, P)