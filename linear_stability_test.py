#!/usr/bin/env python
#
# Driver to examing performance of forward and backward Euler on the Dahlquist test problem.
#
# Daniel R. Reynolds
# Department of Mathematics
# Southern Methodist University

import numpy as np
from implicit_solver import *
from forward_euler import *
from backward_euler import *

# problem time interval and Dahlquist parameter
t0 = 0.0
tf = 0.4
lam = -100.0

# problem-defining functions
def ytrue(t):
    """
    Generates a numpy array containing the true solution to the IVP at a given input t.
    """
    return np.array([np.exp(lam*t)], dtype=float)
def f(t,y):
    """
    Right-hand side function, f(t,y), for the Dahlquist IVP
    """
    return (lam*y)
def J(t,y):
    """
    Jacobian (in dense matrix format) of the right-hand side function, J(t,y) = df/dy
    """
    return np.array( [ [lam] ], dtype=float)

# shared testing data
Nout = 10
tspan = np.linspace(t0, tf, Nout+1)
hvals = np.array([0.005, 0.01, 0.02, 0.04], dtype=float)

# run tests if this file is called as main
if __name__ == "__main__":

    # construct implicit solvers
    implicit_solver = ImplicitSolver(J, solver_type='dense', maxiter=20, rtol=1e-9, atol=1e-12, Jfreq=1)

    # forward Euler tests
    print("forward Euler tests:")
    for ih in range(hvals.size):
        y0 = ytrue(t0)
        t, y, success = forward_euler(f, tspan, y0, hvals[ih])
        err = np.abs(np.transpose(y)-ytrue(t))
        print(" ")
        print("h = ", hvals[ih])
        print("   t      y          error")
        for it in range(t.size):
            print("  %4.2f  %9.2e  %9.2e" % (t[it], y[it,:], err[:,it]))

    # backward Euler tests
    print(" ")
    print("backward Euler tests:")
    for ih in range(hvals.size):
        y0 = ytrue(t0)
        t, y, success = backward_euler(f, tspan, y0, hvals[ih], implicit_solver)
        err = np.abs(np.transpose(y)-ytrue(t))
        print(" ")
        print("h = ", hvals[ih])
        print("   t      y          error")
        for it in range(t.size):
            print("  %4.2f  %9.2e  %9.2e" % (t[it], y[it,:], err[:,it]))
