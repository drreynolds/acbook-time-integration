#!/usr/bin/env python3
#
# Script that runs tests un-preconditioned and preconditioned GMRES solvers
# on the Oregonator example (using backward Euler time-stepping).
#
# Daniel R. Reynolds
# Department of Mathematics
# Southern Methodist University

import numpy as np
import matplotlib.pyplot as plt
from implicit_solver import *
from backward_euler import *
from scipy.sparse.linalg import aslinearoperator
from scipy.integrate import solve_ivp
import example2 as ex2

print("Oregonator preconditioning tests")

# preconditioning matrix
def PMatrix(t, y, gamma):
    """
    Preconditioning matrix that approximates the inverse of the nonlinear solver Jacobian,
    I - gamma*Jf, where Jf(t,y) is the Jacobian of the IVP right-hand side function.
    """
    Jf = np.array([[ -ex2.k1*y[1] - 2.0*ex2.k3*y[0], -ex2.k1*y[0], 0 ],
                     [ -ex2.k1*y[1], -ex2.k1*y[0], 0 ],
                     [ 0, 0, 0.0 ]], dtype=float)
    return np.linalg.inv((np.identity(3) - gamma*Jf))

# preconditioner solver constructor
def PrecSetup(t, y, gamma, rtol, abstol):
    P = PMatrix(t, y, gamma)
    return aslinearoperator(P)

# implicit solver matrix
def JMatrix(t, y, gamma):
    """
    Nonlinear solver Jacobian matrix, I - gamma*Jf, where Jf(t,y) is the
    Jacobian of the IVP right-hand side function.
    """
    return (np.identity(3) - gamma*ex2.J_dense(t,y))

# preconditioned implicit solver matrix
def JPMatrix(t, y, gamma):
    """
    Preconditioned nonlinear solver Jacobian matrix, JP, at a given (t,y) input.
    """
    return (JMatrix(t,y,gamma) @ PMatrix(t,y,gamma))

# check maximum eigenvalue spread for preconditioned and non-preconditioned
# implicit solver matrices along the solution trajectory
def compare_stiffness(N, gamma):
    """
    Utility routine to approximate the maximum eigenvalue spread for both
    the preconditioned and non-preconditioned implicit solver matrices over N
    evenly-spaced time points along the solution trajectory.
    """
    import numpy.linalg as la
    tspan,yref = ex2.reference_solution(N)
    Jeigs = la.eigvals(JMatrix(tspan[0], yref[:,0], gamma))
    Jstiffness = np.max(np.abs(Jeigs))/np.min(np.abs(Jeigs))
    JPeigs = la.eigvals(JPMatrix(tspan[0], yref[:,0], gamma))
    JPstiffness = np.max(np.abs(JPeigs))/np.min(np.abs(JPeigs))
    for i in range(1,np.size(tspan)):
        Jeigs = la.eigvals(JMatrix(tspan[i], yref[:,i], gamma))
        Jstiffness = max(Jstiffness, np.max(np.abs(Jeigs))/np.min(np.abs(Jeigs)))
        JPeigs = la.eigvals(JPMatrix(tspan[i], yref[:,i], gamma))
        JPstiffness = max(JPstiffness, np.max(np.abs(JPeigs))/np.min(np.abs(JPeigs)))
    return (Jstiffness, JPstiffness)

# compute/output maximum approximated stiffness for preconditioned and
# non-preconditioned implicit solver matrices
Nout = 20
interval = ex2.tf - ex2.t0
gammas = interval/Nout/np.array([100, 200, 300, 400, 500, 600], dtype=float)
print("Maximum approxmated stiffness:")
print("     gamma     non-prec    precond")
for gamma in gammas:
    Jstiff, JPstiff = compare_stiffness(501, gamma)
    print("    %.2e   %.2e   %.2e" % (gamma, Jstiff, JPstiff))
