import numpy as np 
from scipy.integrate import ode, odeint

class PF(object):

    def __init__(self, dim_x, dim_z, mu_initial, sigma_initial, num_particles, dt, consts, motion_consts, alpha_var):

        
        self.dim_x = dim_x
        self.dim_z = dim_z

        self.mu_x_ini = mu_initial
        self.sig_x_ini = sigma_initial
        self.NUM_PARTICLES = num_particles

        self.dt = dt

        self.dl = consts[0]
        self.df = consts[1]
        self.dc = consts[2]

        self.Iz = motion_consts[0]
        self.m = motion_consts[1]
        self.g = motion_consts[2]
        self.fric = motion_consts[3]
        fric_var = motion_consts[4]

        self.frichat = self.fric + self.sample(fric_var)

        self.ep = motion_consts[5]
        self.d = motion_consts[6]
        self.L = motion_consts[7]
        self.Rr = motion_consts[8]
        self.s = motion_consts[9]

        self.a1 = alpha_var[0]
        self.a2 = alpha_var[1]

        # self.weights = np.zeros((self.dim_x, self.NUM_PARTICLES))


    def generate_particles(self):
        
        self.Xt = np.random.multivariate_normal(self.mu_x_ini, self.sig_x_ini, size=self.NUM_PARTICLES)
        # print self.Xt

    def predict(self):

        self.solve_motion_model(self.Xt, self.dt)



    def update(self, mu_z, sig_z):

        zt = self.generate_measurement(mu_z, sig_z)

        temp = np.array([self.prob_zt_given_xt(zt,x[2:5],sig_z) for x in self.Xt])
        self.weights = np.reshape(np.array([np.append(np.append([1., 1.], arr), [1.,1.]) for arr in temp]), (self.dim_x, self.NUM_PARTICLES))
        # print self.weights


    def resample(self):
        size = self.NUM_PARTICLES
        self.Xt = np.reshape(self.Xt, (self.dim_x, self.NUM_PARTICLES))
        ranges = np.array([np.cumsum(w/np.sum(w)) for w in self.weights]) # normalize
        self.Xt = np.reshape(np.array([np.array(p)[np.digitize(np.random.random_sample(size), r)] for p,r in zip(self.Xt,ranges)]), (self.NUM_PARTICLES, self.dim_x))



    def prob_zt_given_xt(self, zt, xt, sig_z):

        sm = np.diag(sig_z)

        return np.array([1./(s*np.sqrt(2.*np.pi))*np.exp(-(z-x)**2./(2.*s**2.)) for s,z,x in zip(sm,zt,xt)])


    def generate_measurement(self, mu_z, sig_z):

        return np.random.multivariate_normal(mu_z, sig_z)


    def solve_motion_model(self, X, dt):

        self.Xt = np.array([self.ode2(x, dt, i) for x,i in zip(X,xrange(self.NUM_PARTICLES))])



    def ode2(self, x0, dt, i):

        x0 = np.array(x0)

        a, b, c, d, e, f, g = x0.tolist()
        # self._i = i

        self._omega1 = self.omegas(self.delta(f),self.delta(g), self.frichat, i)[0]
        self._omega2 = self.omegas(self.delta(f),self.delta(g), self.frichat, i)[1]
        self._omega3 = self.omegas(self.delta(f),self.delta(g), self.frichat, i)[2]

        def fa(a, b, c, d, e, f, g):
            # omega3 = self.omegas(self.delta(f),self.delta(g), self._i)[2]
            return self._omega3/self.Iz

        def fb(a, b, c, d, e, f, g):
            # omega1 = self.omegas(self.delta(f),self.delta(g), self._i)[0]
            # omega2 = self.omegas(self.delta(f),self.delta(g), self._i)[1]
            return ((-self._omega1*np.sin(e) + self._omega2*np.cos(e))/self.m)

        def fc(a, b, c, d, e, f, g):
            return b*np.sin(e)

        def fd(a, b, c, d, e, f, g):
            return -b*np.cos(e)

        def fe(a, b, c, d, e, f, g):
            return a

        def ff(a, b, c, d, e, f, g):
            return (a*(self.dl*np.cos(f) - (self.df*np.sin(f)/2) - self.dc)/self.dc) - (b*np.sin(f)/self.dc)

        def fg(a, b, c, d, e, f, g):
            return (a*(self.dl*np.cos(g) + (self.df*np.sin(g)/2) - self.dc)/self.dc) - (b*np.sin(g)/self.dc)

        return np.array(self.rK7(a, b, c, d, e, f, g, fa, fb, fc, fd, fe, ff, fg, dt))



    def omegas(self, delta1, delta2, frichat, i):

        N = self.m*self.g

        F1u = frichat*self.ep*N/2.
        F1w = 0.0      
        F2u = frichat*self.ep*N/2.
        F2w = 0.0
        F3u = frichat*(1-self.ep)*N/2.
        F3w = 0.0
        F4u = frichat*(1-self.ep)*N/2.
        F4w = 0.0

        omega1 = (F3u*np.cos(delta1)) + (F3w*np.sin(delta1)) + F1u + F2u + (F4u*np.cos(delta2)) + (F4w*np.sin(delta2))
        omega2 = F1w - (F3u*np.sin(delta1)) + (F3w*np.cos(delta1)) - (F4u*np.sin(delta2)) + (F4w*np.cos(delta2)) + F2w
        omega3 = (F2u*(self.Rr/2.-self.s))-(F1u*(self.Rr/2.-self.s))-((F2w+F1w)*self.d)+((F4u*np.cos(delta2)+F4w*np.sin(delta2))*(self.Rr/2.-self.s))-((F3u*np.cos(delta1)-F3w*np.sin(delta1))*(self.Rr/2.+self.s))+((F4w*np.cos(delta2)-F4u*np.sin(delta2)+F3w*np.cos(delta1)-F3u*np.sin(delta1))*(self.L-self.d))

        return [omega1[i], omega2[i], omega3[i]]

    def delta(self, alpha):
        return -alpha

    def sample(self, variance):
        return np.random.normal(0.0, np.sqrt(variance), size=self.NUM_PARTICLES)

    def rK7(self, a, b, c, d, e, f, g, fa, fb, fc, fd, fe, ff, fg, hs):

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



# ==================================================================================================================

# Slow method using odeint / ode from scipy
# Might be more accurate though

    # def ode_int(self, x0, dt, i):

    #     a_t = np.arange(0.0, dt, 0.001)
    #     sol = odeint(self.solvr, x0, a_t, args=(i,))
    #     return sol[-1]


    # def solvr(self, x, t, i):

    #     alpha1hat = x[5] + self.sample(self.a1*x[5]**2.)
    #     alpha2hat = x[6] + self.sample(self.a2*x[6]**2.)

    #     x[5] = alpha1hat[i]
    #     x[5] = alpha2hat[i]

    #     omega1 = self.omegas(self.delta(x[5]),self.delta(x[6]))[0]
    #     omega2 = self.omegas(self.delta(x[5]),self.delta(x[6]))[1]
    #     omega3 = self.omegas(self.delta(x[5]),self.delta(x[6]))[2]

    #     omega1, omega2, omega3 = omega1[i], omega2[i], omega3[i]



    #     # Assume v_w = 0  ==>  ignore lateral movement of wheelchair
    #     # ==>  remove function/equation involving v_w from the model
    #     eq1 = omega3/self.Iz
    #     eq2 = ((-omega1*np.sin(x[4]) + omega2*np.cos(x[4]))/self.m)
    #     eq3 = ((-omega1*np.cos(x[4]) - omega2*np.sin(x[4]))/self.m) + x[0]*x[1]
    #     eq4 = x[1]*np.sin(x[4])
    #     eq5 = -x[1]*np.cos(x[4]) 
    #     eq6 = x[0]
    #     eq7 = (x[0]*(self.dl*np.cos(x[5]) - (self.df*np.sin(x[5])/2) - self.dc)/self.dc) - (x[1]*np.sin(x[5])/self.dc)
    #     eq8 = (x[0]*(self.dl*np.cos(x[6]) + (self.df*np.sin(x[6])/2) - self.dc)/self.dc) - (x[1]*np.sin(x[6])/self.dc)

    #     return [eq1, eq2, eq4, eq5, eq6, eq7, eq8]