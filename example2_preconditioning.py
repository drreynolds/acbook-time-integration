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
def PMatrix(t, y, h):
    """
    Preconditioning matrix that approximates the inverse of the nonlinear solver Jacobian,
    I - h*Jf, where Jf(t,y) is the Jacobian of the IVP right-hand side function.
    """
    Jf = np.array([[ -ex2.k1*y[1] - 2.0*ex2.k3*y[0], -ex2.k1*y[0], 0 ],
                     [ -ex2.k1*y[1], -ex2.k1*y[0], 0 ],
                     [ 0, 0, 0.0 ]], dtype=float)
    return np.linalg.inv((np.identity(3) - h*Jf))

# preconditioner solver constructor
def PrecSetup(t, y, h, rtol, abstol):
    P = PMatrix(t, y, h)
    return aslinearoperator(P)

# implicit solver matrix
def JMatrix(t, y, h):
    """
    Nonlinear solver Jacobian matrix, I - h*Jf, where Jf(t,y) is the
    Jacobian of the IVP right-hand side function.
    """
    return (np.identity(3) - h*ex2.J_dense(t,y))

# preconditioned implicit solver matrix
def JPMatrix(t, y, h):
    """
    Preconditioned nonlinear solver Jacobian matrix, JP, at a given (t,y) input.
    """
    return (JMatrix(t,y,h) @ PMatrix(t,y,h))

# check maximum eigenvalue spread for preconditioned and non-preconditioned
# implicit solver matrices along the solution trajectory
def compare_stiffness(N, h):
    """
    Utility routine to approximate the maximum eigenvalue spread for both
    the preconditioned and non-preconditioned implicit solver matrices over N
    evenly-spaced time points along the solution trajectory.
    """
    import numpy.linalg as la
    tspan,yref = ex2.reference_solution(N)
    Jeigs = la.eigvals(JMatrix(tspan[0], yref[:,0], h))
    Jstiffness = np.max(np.abs(Jeigs))/np.min(np.abs(Jeigs))
    JPeigs = la.eigvals(JPMatrix(tspan[0], yref[:,0], h))
    JPstiffness = np.max(np.abs(JPeigs))/np.min(np.abs(JPeigs))
    for i in range(1,np.size(tspan)):
        Jeigs = la.eigvals(JMatrix(tspan[i], yref[:,i], h))
        Jstiffness = max(Jstiffness, np.max(np.abs(Jeigs))/np.min(np.abs(Jeigs)))
        JPeigs = la.eigvals(JPMatrix(tspan[i], yref[:,i], h))
        JPstiffness = max(JPstiffness, np.max(np.abs(JPeigs))/np.min(np.abs(JPeigs)))
    return (Jstiffness, JPstiffness)

# compute/output maximum approximated stiffness for preconditioned and
# non-preconditioned implicit solver matrices
Nout = 20
interval = ex2.tf - ex2.t0
hvals = interval/Nout/np.array([100, 200, 300, 400, 500, 600], dtype=float)
print("Maximum approxmated stiffness:")
print("      h        non-prec    precond")
for h in hvals:
    Jstiff, JPstiff = compare_stiffness(501, h)
    print("    %.2e   %.2e   %.2e" % (h, Jstiff, JPstiff))
