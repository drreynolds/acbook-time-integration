#!/usr/bin/env python3
"""
Script to test newton implementations.
"""
import numpy as np
import time
from newton import *
from scipy.linalg import lu_factor
from scipy.linalg import lu_solve
from scipy.sparse import csc_array
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import aslinearoperator
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import factorized

# general parameters for all tests
maxit = 20
atols = np.array([1e-8, 1e-10, 1e-12])
rtols = np.array([1e-2, 1e-4, 1e-6])
ntols = 3
lags = [1,2,3]

# set the residual function and various flavors of the Jacobian
def f(x):              # nonlinear residual
    f = np.array([x[0] + 0.004*x[0] - 1e3*x[1]*x[2] - 1.0,
                  x[1] - 0.004*x[0] + 1e3*x[1]*x[2] + 30.0*x[1]*x[1],
                  x[2] - 30.0*x[1]*x[1]])
    return f
def Jac(x):         # dense Jacobian
    J = np.array([[1.004, -1e3*x[2], -1e3*x[1]],
                  [-0.004, 1 + 1e3*x[2] + 60.0*x[1], 1e3*x[1]],
                  [0.0, -60.0*x[1], 1.0]])
    return J

# set up Jacobian solvers
def Jsolver_dense(x,rtol,abstol):
    J = Jac(x)
    try:
        lu, piv = lu_factor(J)
    except:
        raise RuntimeError("Dense Jacobian factorization failure")
    Jsolve = lambda b: lu_solve((lu, piv), b)
    return LinearOperator((x.size,x.size), matvec=Jsolve)

def Jsolver_sparse(x,rtol,abstol):
    J = csc_array(Jac(x))
    try:
        Jfactored = factorized(J)
    except:
        raise RuntimeError("Sparse Jacobian factorization failure")
    Jsolve = lambda b: Jfactored(b)
    return LinearOperator((x.size,x.size), matvec=Jsolve)

def Jsolver_gmres(x,rtol,abstol):
    Jv = lambda v: Jac(x) @ v
    J = LinearOperator((x.size,x.size), matvec=Jv)
    Pinv = np.diag([1.0/1.004, 1.0/(1 + 1e3*x[2] + 60.0*x[1]), 1.0])
    P = aslinearoperator(Pinv)
    Jsolve = lambda b: gmres(J, b, tol=rtol, atol=abstol, M=P)[0]
    return LinearOperator((x.size,x.size), matvec=Jsolve)

# set the intial guess for all tests
x0 = np.array([0.95, 0.0, 0.01])

# call solvers for each of three initial guesses
for i in range(ntols):
    print("\nCalling solvers with tolerances: rtol =", rtols[i], ", atol = ", atols[i])

    for l in lags:
        sol = newton(f, Jsolver_dense, x0, rtol=rtols[i], atol=atols[i], Jfreq=l)
        print("  dense,  Jfreq=", l,": success =", sol['success'], ", iters =", sol['iters'])

        sol = newton(f, Jsolver_sparse, x0, rtol=rtols[i], atol=atols[i], Jfreq=l)
        print("  sparse, Jfreq=", l,": success =", sol['success'], ", iters =", sol['iters'])

    sol = newton(f, Jsolver_gmres, x0, rtol=rtols[i], atol=atols[i])
    print("  iterative: success =", sol['success'], ", iters =", sol['iters'])

    vatol = atols[i]*np.ones(x0.shape)
    sol = newton(f, Jsolver_dense, x0, rtol=rtols[i], atol=vatol, Jfreq=1)
    print("  dense, Jfreq=1, vector atol: success =", sol['success'], ", iters =", sol['iters'])
