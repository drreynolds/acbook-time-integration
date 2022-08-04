#!/usr/bin/env python3
"""
Script to test newton implementations.
"""
import numpy as np
import time
from newton import *
from scipy.sparse import csc_array

# general parameters for all tests
maxit = 20
atols = np.array([1e-8, 1e-10, 1e-12])
rtols = np.array([1e-2, 1e-4, 1e-6])
ntols = 3
lags = [0,1,2]

# set the residual function and various flavors of the Jacobian
def f(x):              # nonlinear residual
    f = np.array([x[0] + 0.004*x[0] - 1e3*x[1]*x[2] - 1.0,
                  x[1] - 0.004*x[0] + 1e3*x[1]*x[2] + 30.0*x[1]*x[1],
                  x[2] - 30.0*x[1]*x[1]])
    return f
def Jdense(x):         # dense Jacobian
    J = np.array([[1.004, -1e3*x[2], -1e3*x[1]],
                  [-0.004, 1 + 1e3*x[2] + 60.0*x[1], 1e3*x[1]],
                  [0.0, -60.0*x[1], 1.0]])
    return J
def Jsparse(x):        # sparse Jacobian
    return csc_array(Jdense(x))
def Jtimes(x,v):       # Jacobian-vector product routine
    J = Jdense(x)
    return (J @ v)
def Psolve(x,v):       # preconditioner solver, using diagonal of J
    return np.array([v[0]/1.004, v[1]/(1 + 1e3*x[2] + 60.0*x[1]), v[2]])

# set the intial guess for all tests
x0 = np.array([0.95, 0.0, 0.01])

# call solvers for each of three initial guesses
for i in range(ntols):
    print("\nCalling solvers with tolerances: rtol =", rtols[i], ", atol = ", atols[i])

    for l in lags:
        sol = newton(f, Jdense, x0, rtol=rtols[i], atol=atols[i], lag=l, lsolve='dense')
        print("  dense,  lag=", l,": success =", sol['success'], ", iters =", sol['iters'])

        sol = newton(f, Jsparse, x0, rtol=rtols[i], atol=atols[i], lag=l, lsolve='sparse')
        print("  sparse, lag=", l,": success =", sol['success'], ", iters =", sol['iters'])

    sol = newton(f, Jtimes, x0, rtol=rtols[i], atol=atols[i], lsolve='iterative', Pfcn=Psolve)
    print("  iterative: success =", sol['success'], ", iters =", sol['iters'])

    vatol = atols[i]*np.ones(x0.shape)
    sol = newton(f, Jdense, x0, rtol=rtols[i], atol=vatol, lag=0, lsolve='dense')
    print("  dense, lag=0, vector atol: success =", sol['success'], ", iters =", sol['iters'])
