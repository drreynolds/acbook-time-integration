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
from implicit_solver import *
from forward_euler import *
from backward_euler import *
from erk import *
from explicit_lmm import *
from implicit_lmm import *
from dirk import *
from scipy.integrate import solve_ivp
from scipy.sparse import csc_array

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

# flags to enable/disable classes of tests
test_fwd_euler = True
test_bwd_euler = True
test_erk = True
test_explicit_lmm = True
test_implicit_lmm = True
test_dirk = True
test_dense = True
test_sparse = True
test_gmres = True

# run tests if this file is called as main
if __name__ == "__main__":

    # reference solution test
    y0 = ytrue(t0)
    ivpsol = solve_ivp(f, (t0,tf), y0, t_eval=tspan, rtol=1e-8)
    print("ivp_sol error =", np.linalg.norm(np.transpose(ivpsol.y)-yref,1))

    # construct implicit solvers
    if (test_dense):
        dense_solver  = ImplicitSolver(J_dense,  solver_type='dense',  maxiter=20, rtol=1e-9, atol=1e-12, Jfreq=3)
    if (test_sparse):
        sparse_solver = ImplicitSolver(J_sparse, solver_type='sparse', maxiter=20, rtol=1e-9, atol=1e-12, Jfreq=3)
    if (test_gmres):
        gmres_solver  = ImplicitSolver(J_matvec, solver_type='gmres',  maxiter=20, rtol=1e-9, atol=1e-12, Jfreq=1)

    if (test_fwd_euler):
        # forward Euler tests
        print("forward Euler tests:")
        hvals = (tf-t0)/Nout/np.array([16, 32, 64, 128, 256, 512], dtype=float)
        errs = np.ones(hvals.size)
        for ih in range(hvals.size):
            y0 = ytrue(t0)
            t, y, success = forward_euler(f, tspan, y0, hvals[ih])
            if (not success):
                print("  failure with h =", hvals[ih])
            else:
                errs[ih] = np.linalg.norm(y-yref,1)
                if (ih == 0):
                    print("  h = %.12e,  error = %.12e" % (hvals[ih], errs[ih]))
                else:
                    print("  h = %.12e,  error = %.12e,  rate = %.3f" %
                          (hvals[ih], errs[ih], np.log(errs[ih]/errs[ih-1])/np.log(hvals[ih]/hvals[ih-1])))

    if (test_bwd_euler):
        # backward Euler tests
        hvals = (tf-t0)/Nout/np.array([4, 8, 16, 32, 64, 128], dtype=float)
        if (test_dense):
            print("backward Euler dense tests:")
            errs = np.ones(hvals.size)
            for ih in range(hvals.size):
                y0 = ytrue(t0)
                t, y, success = backward_euler(f, tspan, y0, hvals[ih], dense_solver)
                if (not success):
                    print("  failure with h =", hvals[ih])
                else:
                    errs[ih] = np.linalg.norm(y-yref,1)
                    if (ih == 0):
                        print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i" % (hvals[ih], errs[ih],
                                                                                          dense_solver.get_total_iters(),
                                                                                          dense_solver.get_total_setups()))
                    else:
                        print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i,  rate = %.3f" %
                              (hvals[ih], errs[ih], dense_solver.get_total_iters(), dense_solver.get_total_setups(),
                               np.log(errs[ih]/errs[ih-1])/np.log(hvals[ih]/hvals[ih-1])))
                dense_solver.reset()

        if (test_sparse):
            print("backward Euler sparse tests:")
            errs = np.ones(hvals.size)
            for ih in range(hvals.size):
                y0 = ytrue(t0)
                t, y, success = backward_euler(f, tspan, y0, hvals[ih], sparse_solver)
                if (not success):
                    print("  failure with h =", hvals[ih])
                else:
                    errs[ih] = np.linalg.norm(y-yref,1)
                    if (ih == 0):
                        print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i" % (hvals[ih], errs[ih],
                                                                                          sparse_solver.get_total_iters(),
                                                                                          sparse_solver.get_total_setups()))
                    else:
                        print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i,  rate = %.3f" %
                              (hvals[ih], errs[ih], sparse_solver.get_total_iters(), sparse_solver.get_total_setups(),
                               np.log(errs[ih]/errs[ih-1])/np.log(hvals[ih]/hvals[ih-1])))
                sparse_solver.reset()

        if (test_gmres):
            print("backward Euler gmres tests:")
            errs = np.ones(hvals.size)
            for ih in range(hvals.size):
                y0 = ytrue(t0)
                t, y, success = backward_euler(f, tspan, y0, hvals[ih], gmres_solver)
                if (not success):
                    print("  failure with h =", hvals[ih])
                else:
                    errs[ih] = np.linalg.norm(y-yref,1)
                    if (ih == 0):
                        print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i" % (hvals[ih], errs[ih],
                                                                                          gmres_solver.get_total_iters(),
                                                                                          gmres_solver.get_total_setups()))
                    else:
                        print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i,  rate = %.3f" %
                              (hvals[ih], errs[ih], gmres_solver.get_total_iters(), gmres_solver.get_total_setups(),
                               np.log(errs[ih]/errs[ih-1])/np.log(hvals[ih]/hvals[ih-1])))
                gmres_solver.reset()

    if (test_erk):
        # ERK tests
        print("ERK-2 tests:")
        kappa = 0.5
        A = np.array([[0, 0], [0.5/kappa, 0]], dtype=float)
        b = np.array([1-kappa, kappa], dtype=float)
        c = np.array([0, 0.5/kappa], dtype=float)
        hvals = (tf-t0)/Nout/np.array([8, 16, 32, 64, 128, 256], dtype=float)
        errs = np.ones(hvals.size)
        for ih in range(hvals.size):
            y0 = ytrue(t0)
            t, y, success = erk(f, tspan, y0, hvals[ih], A, b, c)
            if (not success):
                print("  failure with h =", hvals[ih])
            else:
                errs[ih] = np.linalg.norm(y-yref,1)
                if (ih == 0):
                    print("  h = %.12e,  error = %.12e" % (hvals[ih], errs[ih]))
                else:
                    print("  h = %.12e,  error = %.12e,  rate = %.3f" %
                          (hvals[ih], errs[ih], np.log(errs[ih]/errs[ih-1])/np.log(hvals[ih]/hvals[ih-1])))

        print("ERK-3 tests:")
        delta = 0.5
        A = np.array([[0, 0, 0], [2.0/3.0, 0, 0], [2.0/3.0-0.25/delta, 0.25/delta, 0]], dtype=float)
        b = np.array([0.25, 0.75-delta, delta], dtype=float)
        c = np.array([0, 2.0/3.0, 2.0/3.0], dtype=float)
        hvals = (tf-t0)/Nout/np.array([4, 8, 16, 32, 64, 128], dtype=float)
        errs = np.ones(hvals.size)
        for ih in range(hvals.size):
            y0 = ytrue(t0)
            t, y, success = erk(f, tspan, y0, hvals[ih], A, b, c)
            if (not success):
                print("  failure with h =", hvals[ih])
            else:
                errs[ih] = np.linalg.norm(y-yref,1)
                if (ih == 0):
                    print("  h = %.12e,  error = %.12e" % (hvals[ih], errs[ih]))
                else:
                    print("  h = %.12e,  error = %.12e,  rate = %.3f" %
                          (hvals[ih], errs[ih], np.log(errs[ih]/errs[ih-1])/np.log(hvals[ih]/hvals[ih-1])))

        print("ERK-4 tests:")
        A = np.array([[0, 0, 0, 0], [0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1, 0]], dtype=float)
        b = np.array([1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0], dtype=float)
        c = np.array([0, 0.5, 0.5, 1], dtype=float)
        hvals = (tf-t0)/Nout/np.array([4, 8, 16, 32, 64, 128], dtype=float)
        errs = np.ones(hvals.size)
        for ih in range(hvals.size):
            y0 = ytrue(t0)
            t, y, success = erk(f, tspan, y0, hvals[ih], A, b, c)
            if (not success):
                print("  failure with h =", hvals[ih])
            else:
                errs[ih] = np.linalg.norm(y-yref,1)
                if (ih == 0):
                    print("  h = %.12e,  error = %.12e" % (hvals[ih], errs[ih]))
                else:
                    print("  h = %.12e,  error = %.12e,  rate = %.3f" %
                          (hvals[ih], errs[ih], np.log(errs[ih]/errs[ih-1])/np.log(hvals[ih]/hvals[ih-1])))

    if (test_explicit_lmm):
        # Explicit LMM tests
        print("AB-1 tests:")
        alphas = np.array([1, -1], dtype=float)
        betas = np.array([0, 1], dtype=float)
        hvals = (tf-t0)/Nout/np.array([16, 32, 64, 128, 256, 512], dtype=float)
        errs = np.ones(hvals.size)
        for ih in range(hvals.size):
            y0 = np.array([ytrue(t0)])
            t, y, success = explicit_lmm(f, tspan, y0, hvals[ih], alphas, betas)
            if (not success):
                print("  failure with h =", hvals[ih])
            else:
                errs[ih] = np.linalg.norm(y-yref,1)
                if (ih == 0):
                    print("  h = %.12e,  error = %.12e" % (hvals[ih], errs[ih]))
                else:
                    print("  h = %.12e,  error = %.12e,  rate = %.3f" %
                          (hvals[ih], errs[ih], np.log(errs[ih]/errs[ih-1])/np.log(hvals[ih]/hvals[ih-1])))

        print("AB-2 tests:")
        alphas = np.array([1, -1, 0], dtype=float)
        betas = np.array([0, 1.5, -0.5], dtype=float)
        hvals = (tf-t0)/Nout/np.array([8, 16, 32, 64, 128, 256], dtype=float)
        errs = np.ones(hvals.size)
        for ih in range(hvals.size):
            y0 = np.array([ytrue(t0-hvals[ih]), ytrue(t0)])
            t, y, success = explicit_lmm(f, tspan, y0, hvals[ih], alphas, betas)
            if (not success):
                print("  failure with h =", hvals[ih])
            else:
                errs[ih] = np.linalg.norm(y-yref,1)
                if (ih == 0):
                    print("  h = %.12e,  error = %.12e" % (hvals[ih], errs[ih]))
                else:
                    print("  h = %.12e,  error = %.12e,  rate = %.3f" %
                          (hvals[ih], errs[ih], np.log(errs[ih]/errs[ih-1])/np.log(hvals[ih]/hvals[ih-1])))

        print("AB-3 tests:")
        alphas = np.array([1, -1, 0, 0], dtype=float)
        betas = np.array([0, 23.0/12.0, -16.0/12.0, 5.0/12.0], dtype=float)
        hvals = (tf-t0)/Nout/np.array([8, 16, 32, 64, 128, 256], dtype=float)
        errs = np.ones(hvals.size)
        for ih in range(hvals.size):
            y0 = np.array([ytrue(t0-2*hvals[ih]), ytrue(t0-hvals[ih]), ytrue(t0)])
            t, y, success = explicit_lmm(f, tspan, y0, hvals[ih], alphas, betas)
            if (not success):
                print("  failure with h =", hvals[ih])
            else:
                errs[ih] = np.linalg.norm(y-yref,1)
                if (ih == 0):
                    print("  h = %.12e,  error = %.12e" % (hvals[ih], errs[ih]))
                else:
                    print("  h = %.12e,  error = %.12e,  rate = %.3f" %
                          (hvals[ih], errs[ih], np.log(errs[ih]/errs[ih-1])/np.log(hvals[ih]/hvals[ih-1])))


    if (test_implicit_lmm):
        # Implicit LMM tests
        alphas = np.array([1, -1], dtype=float)
        betas = np.array([1, 0], dtype=float)
        hvals = (tf-t0)/Nout/np.array([4, 8, 16, 32, 64, 128], dtype=float)
        if (test_dense):
            print("BDF-1 dense tests:")
            errs = np.ones(hvals.size)
            for ih in range(hvals.size):
                y0 = np.array([ytrue(t0)])
                t, y, success = implicit_lmm(f, tspan, y0, hvals[ih], alphas, betas, dense_solver)
                if (not success):
                    print("  failure with h =", hvals[ih])
                else:
                    errs[ih] = np.linalg.norm(y-yref,1)
                    if (ih == 0):
                        print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i" % (hvals[ih], errs[ih],
                                                                                          dense_solver.get_total_iters(),
                                                                                          dense_solver.get_total_setups()))
                    else:
                        print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i,  rate = %.3f" %
                              (hvals[ih], errs[ih], dense_solver.get_total_iters(), dense_solver.get_total_setups(),
                               np.log(errs[ih]/errs[ih-1])/np.log(hvals[ih]/hvals[ih-1])))
                dense_solver.reset()

        if (test_sparse):
            print("BDF-1 sparse tests:")
            errs = np.ones(hvals.size)
            for ih in range(hvals.size):
                y0 = np.array([ytrue(t0)])
                t, y, success = implicit_lmm(f, tspan, y0, hvals[ih], alphas, betas, sparse_solver)
                if (not success):
                    print("  failure with h =", hvals[ih])
                else:
                    errs[ih] = np.linalg.norm(y-yref,1)
                    if (ih == 0):
                        print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i" % (hvals[ih], errs[ih],
                                                                                          sparse_solver.get_total_iters(),
                                                                                          sparse_solver.get_total_setups()))
                    else:
                        print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i,  rate = %.3f" %
                              (hvals[ih], errs[ih], sparse_solver.get_total_iters(), sparse_solver.get_total_setups(),
                               np.log(errs[ih]/errs[ih-1])/np.log(hvals[ih]/hvals[ih-1])))
                sparse_solver.reset()

        if (test_gmres):
            print("BDF-1 gmres tests:")
            errs = np.ones(hvals.size)
            for ih in range(hvals.size):
                y0 = np.array([ytrue(t0)])
                t, y, success = implicit_lmm(f, tspan, y0, hvals[ih], alphas, betas, gmres_solver)
                if (not success):
                    print("  failure with h =", hvals[ih])
                else:
                    errs[ih] = np.linalg.norm(y-yref,1)
                    if (ih == 0):
                        print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i" % (hvals[ih], errs[ih],
                                                                                          gmres_solver.get_total_iters(),
                                                                                          gmres_solver.get_total_setups()))
                    else:
                        print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i,  rate = %.3f" %
                              (hvals[ih], errs[ih], gmres_solver.get_total_iters(), gmres_solver.get_total_setups(),
                               np.log(errs[ih]/errs[ih-1])/np.log(hvals[ih]/hvals[ih-1])))
                gmres_solver.reset()

        alphas = np.array([1, -4.0/3.0, 1.0/3.0], dtype=float)
        betas = np.array([2.0/3.0, 0, 0], dtype=float)
        hvals = (tf-t0)/Nout/np.array([4, 8, 16, 32, 64, 128], dtype=float)
        if (test_dense):
            print("BDF-2 dense tests:")
            errs = np.ones(hvals.size)
            for ih in range(hvals.size):
                y0 = np.array([ytrue(t0-hvals[ih]), ytrue(t0)])
                t, y, success = implicit_lmm(f, tspan, y0, hvals[ih], alphas, betas, dense_solver)
                if (not success):
                    print("  failure with h =", hvals[ih])
                else:
                    errs[ih] = np.linalg.norm(y-yref,1)
                    if (ih == 0):
                        print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i" % (hvals[ih], errs[ih],
                                                                                          dense_solver.get_total_iters(),
                                                                                          dense_solver.get_total_setups()))
                    else:
                        print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i,  rate = %.3f" %
                              (hvals[ih], errs[ih], dense_solver.get_total_iters(), dense_solver.get_total_setups(),
                               np.log(errs[ih]/errs[ih-1])/np.log(hvals[ih]/hvals[ih-1])))
                dense_solver.reset()

        if (test_sparse):
            print("BDF-2 sparse tests:")
            errs = np.ones(hvals.size)
            for ih in range(hvals.size):
                y0 = np.array([ytrue(t0-hvals[ih]), ytrue(t0)])
                t, y, success = implicit_lmm(f, tspan, y0, hvals[ih], alphas, betas, sparse_solver)
                if (not success):
                    print("  failure with h =", hvals[ih])
                else:
                    errs[ih] = np.linalg.norm(y-yref,1)
                    if (ih == 0):
                        print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i" % (hvals[ih], errs[ih],
                                                                                          sparse_solver.get_total_iters(),
                                                                                          sparse_solver.get_total_setups()))
                    else:
                        print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i,  rate = %.3f" %
                              (hvals[ih], errs[ih], sparse_solver.get_total_iters(), sparse_solver.get_total_setups(),
                               np.log(errs[ih]/errs[ih-1])/np.log(hvals[ih]/hvals[ih-1])))
                sparse_solver.reset()

        if (test_gmres):
            print("BDF-2 gmres tests:")
            errs = np.ones(hvals.size)
            for ih in range(hvals.size):
                y0 = np.array([ytrue(t0-hvals[ih]), ytrue(t0)])
                t, y, success = implicit_lmm(f, tspan, y0, hvals[ih], alphas, betas, gmres_solver)
                if (not success):
                    print("  failure with h =", hvals[ih])
                else:
                    errs[ih] = np.linalg.norm(y-yref,1)
                    if (ih == 0):
                        print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i" % (hvals[ih], errs[ih],
                                                                                          gmres_solver.get_total_iters(),
                                                                                          gmres_solver.get_total_setups()))
                    else:
                        print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i,  rate = %.3f" %
                              (hvals[ih], errs[ih], gmres_solver.get_total_iters(), gmres_solver.get_total_setups(),
                               np.log(errs[ih]/errs[ih-1])/np.log(hvals[ih]/hvals[ih-1])))
                gmres_solver.reset()

        alphas = np.array([1, -18.0/11.0, 9.0/11.0, -2.0/11.0], dtype=float)
        betas = np.array([6.0/11.0, 0, 0, 0], dtype=float)
        hvals = (tf-t0)/Nout/np.array([4, 8, 16, 32, 64, 128], dtype=float)
        if (test_dense):
            print("BDF-3 dense tests:")
            errs = np.ones(hvals.size)
            for ih in range(hvals.size):
                y0 = np.array([ytrue(t0-2*hvals[ih]), ytrue(t0-hvals[ih]), ytrue(t0)])
                t, y, success = implicit_lmm(f, tspan, y0, hvals[ih], alphas, betas, dense_solver)
                if (not success):
                    print("  failure with h =", hvals[ih])
                else:
                    errs[ih] = np.linalg.norm(y-yref,1)
                    if (ih == 0):
                        print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i" % (hvals[ih], errs[ih],
                                                                                          dense_solver.get_total_iters(),
                                                                                          dense_solver.get_total_setups()))
                    else:
                        print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i,  rate = %.3f" %
                              (hvals[ih], errs[ih], dense_solver.get_total_iters(), dense_solver.get_total_setups(),
                               np.log(errs[ih]/errs[ih-1])/np.log(hvals[ih]/hvals[ih-1])))
                dense_solver.reset()

        if (test_sparse):
            print("BDF-3 sparse tests:")
            errs = np.ones(hvals.size)
            for ih in range(hvals.size):
                y0 = np.array([ytrue(t0-2*hvals[ih]), ytrue(t0-hvals[ih]), ytrue(t0)])
                t, y, success = implicit_lmm(f, tspan, y0, hvals[ih], alphas, betas, sparse_solver)
                if (not success):
                    print("  failure with h =", hvals[ih])
                else:
                    errs[ih] = np.linalg.norm(y-yref,1)
                    if (ih == 0):
                        print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i" % (hvals[ih], errs[ih],
                                                                                          sparse_solver.get_total_iters(),
                                                                                          sparse_solver.get_total_setups()))
                    else:
                        print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i,  rate = %.3f" %
                              (hvals[ih], errs[ih], sparse_solver.get_total_iters(), sparse_solver.get_total_setups(),
                               np.log(errs[ih]/errs[ih-1])/np.log(hvals[ih]/hvals[ih-1])))
                sparse_solver.reset()

        if (test_gmres):
            print("BDF-3 gmres tests:")
            errs = np.ones(hvals.size)
            for ih in range(hvals.size):
                y0 = np.array([ytrue(t0-2*hvals[ih]), ytrue(t0-hvals[ih]), ytrue(t0)])
                t, y, success = implicit_lmm(f, tspan, y0, hvals[ih], alphas, betas, gmres_solver)
                if (not success):
                    print("  failure with h =", hvals[ih])
                else:
                    errs[ih] = np.linalg.norm(y-yref,1)
                    if (ih == 0):
                        print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i" % (hvals[ih], errs[ih],
                                                                                          gmres_solver.get_total_iters(),
                                                                                          gmres_solver.get_total_setups()))
                    else:
                        print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i,  rate = %.3f" %
                              (hvals[ih], errs[ih], gmres_solver.get_total_iters(), gmres_solver.get_total_setups(),
                               np.log(errs[ih]/errs[ih-1])/np.log(hvals[ih]/hvals[ih-1])))
                gmres_solver.reset()

    if (test_dirk):
        # DIRK tests
        gamma = 0.43586652150845906
        tau2 = 0.5*(1.0+gamma)
        A = np.array([[gamma, 0, 0],
                      [tau2-gamma, gamma, 0],
                      [-0.25*(6*gamma**2 - 16*gamma + 1), 0.25*(6*gamma**2 - 20*gamma + 5), gamma]], dtype=float)
        b = np.array([-0.25*(6*gamma**2 - 16*gamma + 1), 0.25*(6*gamma**2 - 20*gamma + 5), gamma], dtype=float)
        c = np.array([gamma, tau2, 1], dtype=float)
        hvals = (tf-t0)/Nout/np.array([4, 8, 16, 32, 64, 128], dtype=float)
        if (test_dense):
            print("DIRK-3 dense tests:")
            errs = np.ones(hvals.size)
            for ih in range(hvals.size):
                y0 = ytrue(t0)
                t, y, success = dirk(f, tspan, y0, hvals[ih], A, b, c, dense_solver)
                if (not success):
                    print("  failure with h =", hvals[ih])
                else:
                    errs[ih] = np.linalg.norm(y-yref,1)
                    if (ih == 0):
                        print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i" % (hvals[ih], errs[ih],
                                                                                          dense_solver.get_total_iters(),
                                                                                          dense_solver.get_total_setups()))
                    else:
                        print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i,  rate = %.3f" %
                              (hvals[ih], errs[ih], dense_solver.get_total_iters(), dense_solver.get_total_setups(),
                               np.log(errs[ih]/errs[ih-1])/np.log(hvals[ih]/hvals[ih-1])))
                dense_solver.reset()

        if (test_sparse):
            print("DIRK-3 sparse tests:")
            errs = np.ones(hvals.size)
            for ih in range(hvals.size):
                y0 = ytrue(t0)
                t, y, success = dirk(f, tspan, y0, hvals[ih], A, b, c, sparse_solver)
                if (not success):
                    print("  failure with h =", hvals[ih])
                else:
                    errs[ih] = np.linalg.norm(y-yref,1)
                    if (ih == 0):
                        print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i" % (hvals[ih], errs[ih],
                                                                                          sparse_solver.get_total_iters(),
                                                                                          sparse_solver.get_total_setups()))
                    else:
                        print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i,  rate = %.3f" %
                              (hvals[ih], errs[ih], sparse_solver.get_total_iters(), sparse_solver.get_total_setups(),
                               np.log(errs[ih]/errs[ih-1])/np.log(hvals[ih]/hvals[ih-1])))
                sparse_solver.reset()

        if (test_gmres):
            print("DIRK-3 gmres tests:")
            errs = np.ones(hvals.size)
            for ih in range(hvals.size):
                y0 = ytrue(t0)
                t, y, success = dirk(f, tspan, y0, hvals[ih], A, b, c, gmres_solver)
                if (not success):
                    print("  failure with h =", hvals[ih])
                else:
                    errs[ih] = np.linalg.norm(y-yref,1)
                    if (ih == 0):
                        print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i" % (hvals[ih], errs[ih],
                                                                                          gmres_solver.get_total_iters(),
                                                                                          gmres_solver.get_total_setups()))
                    else:
                        print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i,  rate = %.3f" %
                              (hvals[ih], errs[ih], gmres_solver.get_total_iters(), gmres_solver.get_total_setups(),
                               np.log(errs[ih]/errs[ih-1])/np.log(hvals[ih]/hvals[ih-1])))
                gmres_solver.reset()


        gamma = np.cos(np.pi/18.0)/np.sqrt(3) + 0.5
        delta = 1.0/(6.0*(2.0*gamma-1)**2)
        A = np.array([[gamma, 0, 0],
                      [0.5-gamma, gamma, 0],
                      [2*gamma, 1.0-4*gamma, gamma]], dtype=float)
        b = np.array([delta, 1.0-2*delta, delta], dtype=float)
        c = np.array([gamma, 0.5, 1-gamma], dtype=float)
        hvals = (tf-t0)/Nout/np.array([4, 8, 16, 32, 64, 128], dtype=float)
        if (test_dense):
            print("DIRK-4 dense tests:")
            errs = np.ones(hvals.size)
            for ih in range(hvals.size):
                y0 = ytrue(t0)
                t, y, success = dirk(f, tspan, y0, hvals[ih], A, b, c, dense_solver)
                if (not success):
                    print("  failure with h =", hvals[ih])
                else:
                    errs[ih] = np.linalg.norm(y-yref,1)
                    if (ih == 0):
                        print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i" % (hvals[ih], errs[ih],
                                                                                          dense_solver.get_total_iters(),
                                                                                          dense_solver.get_total_setups()))
                    else:
                        print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i,  rate = %.3f" %
                              (hvals[ih], errs[ih], dense_solver.get_total_iters(), dense_solver.get_total_setups(),
                               np.log(errs[ih]/errs[ih-1])/np.log(hvals[ih]/hvals[ih-1])))
                dense_solver.reset()

        if (test_sparse):
            print("DIRK-4 sparse tests:")
            errs = np.ones(hvals.size)
            for ih in range(hvals.size):
                y0 = ytrue(t0)
                t, y, success = dirk(f, tspan, y0, hvals[ih], A, b, c, sparse_solver)
                if (not success):
                    print("  failure with h =", hvals[ih])
                else:
                    errs[ih] = np.linalg.norm(y-yref,1)
                    if (ih == 0):
                        print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i" % (hvals[ih], errs[ih],
                                                                                          sparse_solver.get_total_iters(),
                                                                                          sparse_solver.get_total_setups()))
                    else:
                        print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i,  rate = %.3f" %
                              (hvals[ih], errs[ih], sparse_solver.get_total_iters(), sparse_solver.get_total_setups(),
                               np.log(errs[ih]/errs[ih-1])/np.log(hvals[ih]/hvals[ih-1])))
                sparse_solver.reset()

        if (test_gmres):
            print("DIRK-4 gmres tests:")
            errs = np.ones(hvals.size)
            for ih in range(hvals.size):
                y0 = ytrue(t0)
                t, y, success = dirk(f, tspan, y0, hvals[ih], A, b, c, gmres_solver)
                if (not success):
                    print("  failure with h =", hvals[ih])
                else:
                    errs[ih] = np.linalg.norm(y-yref,1)
                    if (ih == 0):
                        print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i" % (hvals[ih], errs[ih],
                                                                                          gmres_solver.get_total_iters(),
                                                                                          gmres_solver.get_total_setups()))
                    else:
                        print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i,  rate = %.3f" %
                              (hvals[ih], errs[ih], gmres_solver.get_total_iters(), gmres_solver.get_total_setups(),
                               np.log(errs[ih]/errs[ih-1])/np.log(hvals[ih]/hvals[ih-1])))
                gmres_solver.reset()


        A = np.array([[4024571134387.0/14474071345096, 0, 0, 0, 0],
                      [9365021263232.0/12572342979331, 4024571134387.0/14474071345096, 0, 0, 0],
                      [2144716224527.0/9320917548702, -397905335951.0/4008788611757, 4024571134387.0/14474071345096, 0, 0],
                      [-291541413000.0/6267936762551, 226761949132.0/4473940808273, -1282248297070.0/9697416712681, 4024571134387.0/14474071345096, 0],
                      [-2481679516057.0/4626464057815, -197112422687.0/6604378783090, 3952887910906.0/9713059315593, 4906835613583.0/8134926921134, 4024571134387.0/14474071345096]], dtype=float)
        b = np.array([-2522702558582.0/12162329469185, 1018267903655.0/12907234417901, 4542392826351.0/13702606430957, 5001116467727.0/12224457745473, 1509636094297.0/3891594770934], dtype=float)
        c = np.array([4024571134387.0/14474071345096, 5555633399575.0/5431021154178, 5255299487392.0/12852514622453, 3.0/20, 10449500210709.0/14474071345096], dtype=float)
        hvals = (tf-t0)/Nout/np.array([4, 8, 16, 32, 64, 128], dtype=float)
        if (test_dense):
            print("DIRK-5 dense tests:")
            errs = np.ones(hvals.size)
            for ih in range(hvals.size):
                y0 = ytrue(t0)
                t, y, success = dirk(f, tspan, y0, hvals[ih], A, b, c, dense_solver)
                if (not success):
                    print("  failure with h =", hvals[ih])
                else:
                    errs[ih] = np.linalg.norm(y-yref,1)
                    if (ih == 0):
                        print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i" % (hvals[ih], errs[ih],
                                                                                          dense_solver.get_total_iters(),
                                                                                          dense_solver.get_total_setups()))
                    else:
                        print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i,  rate = %.3f" %
                              (hvals[ih], errs[ih], dense_solver.get_total_iters(), dense_solver.get_total_setups(),
                               np.log(errs[ih]/errs[ih-1])/np.log(hvals[ih]/hvals[ih-1])))
                dense_solver.reset()

        if (test_sparse):
            print("DIRK-5 sparse tests:")
            errs = np.ones(hvals.size)
            for ih in range(hvals.size):
                y0 = ytrue(t0)
                t, y, success = dirk(f, tspan, y0, hvals[ih], A, b, c, sparse_solver)
                if (not success):
                    print("  failure with h =", hvals[ih])
                else:
                    errs[ih] = np.linalg.norm(y-yref,1)
                    if (ih == 0):
                        print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i" % (hvals[ih], errs[ih],
                                                                                          sparse_solver.get_total_iters(),
                                                                                          sparse_solver.get_total_setups()))
                    else:
                        print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i,  rate = %.3f" %
                              (hvals[ih], errs[ih], sparse_solver.get_total_iters(), sparse_solver.get_total_setups(),
                               np.log(errs[ih]/errs[ih-1])/np.log(hvals[ih]/hvals[ih-1])))
                sparse_solver.reset()

        if (test_gmres):
            print("DIRK-5 gmres tests:")
            errs = np.ones(hvals.size)
            for ih in range(hvals.size):
                y0 = ytrue(t0)
                t, y, success = dirk(f, tspan, y0, hvals[ih], A, b, c, gmres_solver)
                if (not success):
                    print("  failure with h =", hvals[ih])
                else:
                    errs[ih] = np.linalg.norm(y-yref,1)
                    if (ih == 0):
                        print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i" % (hvals[ih], errs[ih],
                                                                                          gmres_solver.get_total_iters(),
                                                                                          gmres_solver.get_total_setups()))
                    else:
                        print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i,  rate = %.3f" %
                              (hvals[ih], errs[ih], gmres_solver.get_total_iters(), gmres_solver.get_total_setups(),
                               np.log(errs[ih]/errs[ih-1])/np.log(hvals[ih]/hvals[ih-1])))
                gmres_solver.reset()
