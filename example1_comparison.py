#!/usr/bin/env python3
#
# Script that runs various explicit methods on the Brusslator example,
# including built-in adaptive solvers.
#
# Daniel R. Reynolds
# Department of Mathematics
# Southern Methodist University

import numpy as np
import matplotlib.pyplot as plt
from forward_euler import *
from erk import *
from explicit_lmm import *
from scipy.integrate import solve_ivp
import example1 as ex1

print("Brusselator IVP Tests")

# generate solution plot
tspan,yref = ex1.reference_solution(501)
fig = plt.figure(figsize=(8,6), dpi=200)
plt.plot(tspan, yref[0,:], label='n(X)')
plt.plot(tspan, yref[1,:], label='n(Y)')
plt.xlabel('time')
plt.legend()
plt.savefig('Brusselator_solution.pdf')
plt.show()

# compute/output maximum approximated stiffness
print("Maximum approximate stiffness = %.2e" % (ex1.maximum_stiffness(501)))

# shared testing data
interval = ex1.tf - ex1.t0
Nout = 20
tspan,yref = ex1.reference_solution(Nout+1, reltol=1e-12)
hvals = interval/Nout/np.array([10, 20, 30, 40, 50, 60], dtype=float)

# flags to enable/disable classes of tests
test_fwd_euler = True
test_erk = True
test_explicit_lmm = False

# forward Euler tests
if (test_fwd_euler):
    print("forward Euler tests:")
    errs = np.ones(hvals.size)
    for ih in range(hvals.size):
        y0 = np.copy(ex1.y0)
        t, y, success = forward_euler(ex1.f, tspan, y0, hvals[ih])
        if (not success):
            print("  failure with h =", hvals[ih])
        else:
            errs[ih] = np.linalg.norm(y-np.transpose(yref),1)
            if (ih == 0):
                print("  h = %.12e,  error = %.12e" % (hvals[ih], errs[ih]))
            else:
                print("  h = %.12e,  error = %.12e,  rate = %.3f" %
                      (hvals[ih], errs[ih], np.log(errs[ih]/errs[ih-1])/np.log(hvals[ih]/hvals[ih-1])))

# ERK tests
if (test_erk):
    print("ERK-2 tests:")
    kappa = 0.5
    A = np.array([[0, 0], [0.5/kappa, 0]], dtype=float)
    b = np.array([1-kappa, kappa], dtype=float)
    c = np.array([0, 0.5/kappa], dtype=float)
    errs = np.ones(hvals.size)
    for ih in range(hvals.size):
        y0 = np.copy(ex1.y0)
        t, y, success = erk(ex1.f, tspan, y0, hvals[ih], A, b, c)
        if (not success):
            print("  failure with h =", hvals[ih])
        else:
            errs[ih] = np.linalg.norm(y-np.transpose(yref),1)
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
    errs = np.ones(hvals.size)
    for ih in range(hvals.size):
        y0 = np.copy(ex1.y0)
        t, y, success = erk(ex1.f, tspan, y0, hvals[ih], A, b, c)
        if (not success):
            print("  failure with h =", hvals[ih])
        else:
            errs[ih] = np.linalg.norm(y-np.transpose(yref),1)
            if (ih == 0):
                print("  h = %.12e,  error = %.12e" % (hvals[ih], errs[ih]))
            else:
                print("  h = %.12e,  error = %.12e,  rate = %.3f" %
                      (hvals[ih], errs[ih], np.log(errs[ih]/errs[ih-1])/np.log(hvals[ih]/hvals[ih-1])))

    print("ERK-4 tests:")
    A = np.array([[0, 0, 0, 0], [0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1, 0]], dtype=float)
    b = np.array([1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0], dtype=float)
    c = np.array([0, 0.5, 0.5, 1], dtype=float)
    errs = np.ones(hvals.size)
    for ih in range(hvals.size):
        y0 = np.copy(ex1.y0)
        t, y, success = erk(ex1.f, tspan, y0, hvals[ih], A, b, c)
        if (not success):
            print("  failure with h =", hvals[ih])
        else:
            errs[ih] = np.linalg.norm(y-np.transpose(yref),1)
            if (ih == 0):
                print("  h = %.12e,  error = %.12e" % (hvals[ih], errs[ih]))
            else:
                print("  h = %.12e,  error = %.12e,  rate = %.3f" %
                      (hvals[ih], errs[ih], np.log(errs[ih]/errs[ih-1])/np.log(hvals[ih]/hvals[ih-1])))

# Explicit LMM tests
if (test_explicit_lmm):
    print("AB-1 tests:")
    alphas = np.array([1, -1], dtype=float)
    betas = np.array([0, 1], dtype=float)
    errs = np.ones(hvals.size)
    for ih in range(hvals.size):
        y0 = np.array([ex1.y0])
        t, y, success = explicit_lmm(ex1.f, tspan, y0, hvals[ih], alphas, betas)
        if (not success):
            print("  failure with h =", hvals[ih])
        else:
            errs[ih] = np.linalg.norm(y-np.transpose(yref),1)
            if (ih == 0):
                print("  h = %.12e,  error = %.12e" % (hvals[ih], errs[ih]))
            else:
                print("  h = %.12e,  error = %.12e,  rate = %.3f" %
                      (hvals[ih], errs[ih], np.log(errs[ih]/errs[ih-1])/np.log(hvals[ih]/hvals[ih-1])))

    print("AB-2 tests:")
    alphas = np.array([1, -1, 0], dtype=float)
    betas = np.array([0, 1.5, -0.5], dtype=float)
    errs = np.ones(hvals.size)
    for ih in range(hvals.size):
        y0 = np.array([ytrue(t0-hvals[ih]), ytrue(t0)])
        t, y, success = explicit_lmm(ex1.f, tspan, y0, hvals[ih], alphas, betas)
        if (not success):
            print("  failure with h =", hvals[ih])
        else:
            errs[ih] = np.linalg.norm(y-np.transpose(yref),1)
            if (ih == 0):
                print("  h = %.12e,  error = %.12e" % (hvals[ih], errs[ih]))
            else:
                print("  h = %.12e,  error = %.12e,  rate = %.3f" %
                      (hvals[ih], errs[ih], np.log(errs[ih]/errs[ih-1])/np.log(hvals[ih]/hvals[ih-1])))

    print("AB-3 tests:")
    alphas = np.array([1, -1, 0, 0], dtype=float)
    betas = np.array([0, 23.0/12.0, -16.0/12.0, 5.0/12.0], dtype=float)
    errs = np.ones(hvals.size)
    for ih in range(hvals.size):
        y0 = np.array([ytrue(t0-2*hvals[ih]), ytrue(t0-hvals[ih]), ytrue(t0)])
        t, y, success = explicit_lmm(ex1.f, tspan, y0, hvals[ih], alphas, betas)
        if (not success):
            print("  failure with h =", hvals[ih])
        else:
            errs[ih] = np.linalg.norm(y-np.transpose(yref),1)
            if (ih == 0):
                print("  h = %.12e,  error = %.12e" % (hvals[ih], errs[ih]))
            else:
                print("  h = %.12e,  error = %.12e,  rate = %.3f" %
                      (hvals[ih], errs[ih], np.log(errs[ih]/errs[ih-1])/np.log(hvals[ih]/hvals[ih-1])))
