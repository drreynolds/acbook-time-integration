#!/usr/bin/env python
#
# Driver to verify implementations of IVP method implementations.
#
# This uses a nonlinear version of the "Prothero--Robinson" problem:
#
#  u'(t) = lambda_f * A(t,ut)
#          + (1-epsilon)/alpha * (lambda_f - lambda_s)* B(t,vt)
#          - omega*sin(omega*t)/(2u)
#  v'(t) = -alpha*epsilon*(u - lambda_s)*A(t,u) + lambda_s*B(t,v)
#          - sin(t)/(2*v)
#
# where  A(t,u) = (-3 + u*u - cos(omega*t))/(2u)
#        B(t,v) = (-2 + v*v - cos(t))/(2v)
#
# over the time interval [0, 5*pi/2], with parameters alpha = 1,
# omega = 20, epsilon = 0.1, lambda_f = -10, lambda_s = -1, and
# C = [ [lambda_f, ((1-epsilon)/alpha)*(lambda_f - lambda_s)]
#       [-alpha*epsilon*(lambda_f - lambda_s), lambda_s] ]
#
# This problem has analytical solution u(t) = sqrt(3 + cos(omega*t))
# and v(t) = sqrt(2 + cos(t)).
#
#
# Daniel R. Reynolds
# Department of Mathematics
# Southern Methodist University

import numpy as np
from forward_euler import *
from scipy.integrate import solve_ivp

# problem time interval and parameters
t0 = 0.0
tf = 2.5*np.pi
alpha = 1.0
omega = 20.0
epsilon = 0.1
lambda_f = -10.0
lambda_s = -1.0
C = np.array( [ [lambda_f, ((1.0-epsilon)/alpha)*(lambda_f - lambda_s)],
                [-alpha*epsilon*(lambda_f - lambda_s), lambda_s] ], dtype=float )

# problem-defining functions
def ytrue(t):
    """
    Generates a numpy array containing the true solution to the IVP at a given input t.
    """
    return np.array([np.sqrt(3.0 + np.cos(omega*t)), np.sqrt(2.0 + np.cos(t))], dtype=float)
def f(t,y):
    """
    Right-hand side function, f(t,y), for the IVP
    """
    AB = np.array( [(-3.0 + y[0]**2 - np.cos(omega*t))/(2.0*y[0]),
                    (-2.0 + y[1]**2 - np.cos(t))/(2.0*y[1])], dtype=float)
    Udot = C@AB - np.array([omega*np.sin(omega*t)/(2.0*y[0]), np.sin(t)/(2.0*y[1])], dtype=float)
    return Udot
def J_dense(t,y):
    """
    Jacobian (in dense matrix format) of the right-hand side function, J(t,y) = df/dy
    """
    dA_dU1 = 1.0 - (-3.0 + y[0]**2 - np.cos(omega*t))/(2*y[0]**2)
    dB_dU2 = 1.0 - (-2.0 + y[1]**2 - np.cos(t))/(2.0*y[1]**2)
    return np.array( [ [lambda_f*dA_dU1 + omega*np.sin(omega*t)/(2.0*y[0]**2),
                        (1.0-epsilon)*(lambda_f - lambda_s)/alpha*dB_dU2],
                       [-alpha*epsilon*(lambda_f - lambda_s)*dA_dU1,
                        lambda_s*dB_dU2 + np.sin(t)/(2.0*y[1]**2)] ], dtype=float)
def J_sparse(t,y):
    """
    Jacobian (in sparse matrix format) of the right-hand side function, J(t,y)
    """
    return csc_array(J_dense(t,y))
def J_matvec(t,y,v):
    """
    Jacobian-vector product function, J@v
    """
    return (J_dense(t,y)@v)

# shared testing data
Nout = 20
tspan = np.linspace(t0, tf, Nout+1)
yref = np.zeros((Nout+1, 2), dtype=float)
for i in range(Nout+1):
    yref[i,:] = ytrue(tspan[i])
y0 = ytrue(t0)

# run tests if this file is called as main
if __name__ == "__main__":

    # reference solution test
    ivpsol = solve_ivp(f, (t0,tf), y0, t_eval=tspan, rtol=1e-8)
    print("ivp_sol error =", np.linalg.norm(np.transpose(ivpsol.y)-yref,1))

    # forward Euler tests
    print("forward Euler tests:")
    hvals = (tf-t0)/Nout/np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0], dtype=float)
    errs = np.ones(hvals.size)
    for ih in range(hvals.size):
        t, y, success = forward_euler(f, tspan, y0, hvals[ih])
        if (not success):
            print("  failure with h =", hvals[ih])
        else:
            errs[ih] = np.linalg.norm(y-yref,1)
            if (ih == 0):
                print("  h =", hvals[ih], "  error =", errs[ih])
            else:
                print("  h =", hvals[ih], "  error =", errs[ih], "  rate =", np.log(errs[ih]/errs[ih-1])/np.log(hvals[ih]/hvals[ih-1]))

#
