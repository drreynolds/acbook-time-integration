#!/usr/bin/env python3
#
# Script that runs various implicit methods on the Oregonator example,
# including built-in adaptive solvers.
#
# Daniel R. Reynolds
# Department of Mathematics
# Southern Methodist University

import numpy as np
import matplotlib.pyplot as plt
from implicit_solver import *
from backward_euler import *
from dirk import *
from implicit_lmm import *
from scipy.integrate import solve_ivp
import example2 as ex2

print("Oregonator IVP Tests")

# generate solution plot
tspan,yref = ex2.reference_solution(501)
fig = plt.figure(figsize=(8,6), dpi=100)
ax = fig.add_subplot(311)
ax.plot(tspan, yref[0,:])
ax.set_xlabel('time')
ax.set_ylabel('$n_1$')
ax.set_ylim([0, 6e-6])
ax.set_xlim([0, 400])
ax.ticklabel_format(axis='y', style='sci', scilimits=(-6,-6), useMathText=True)
ax = fig.add_subplot(312)
ax.semilogy(tspan, yref[1,:])
ax.set_xlabel('time')
ax.set_ylabel('$n_2$')
ax.set_ylim([1e-10, 1e-2])
ax.set_xlim([0, 400])
ax = fig.add_subplot(313)
ax.plot(tspan, yref[2,:])
ax.set_xlabel('time')
ax.set_ylabel('$n_3$')
ax.set_ylim([0, 8e-4])
ax.set_xlim([0, 400])
plt.savefig('Oregonator_solution.pdf')
plt.show()

# compute/output maximum approximated stiffness
print("Maximum approximate stiffness = %.2e" % (ex2.maximum_stiffness(501)))

# shared testing data
interval = ex2.tf - ex2.t0
Nout = 20
tspan,yref = ex2.reference_solution(Nout+1, reltol=1e-12)
hvals = interval/Nout/np.array([2000, 3000, 4000, 5000], dtype=float)
solver = ImplicitSolver(ex2.J_dense, solver_type='dense', maxiter=20,
                        rtol=1e-9, atol=1e-12, Jfreq=3)

# flags to enable/disable classes of tests
test_bwd_euler = False
test_dirk = False
test_implicit_lmm = True

# backward Euler tests
if (test_bwd_euler):

    print("backward Euler tests:")
    errs = np.ones(hvals.size)
    for ih in range(hvals.size):
        y0 = np.copy(ex2.y0)
        t, y, success = backward_euler(ex2.f, tspan, y0, hvals[ih], solver)
        if (not success):
            print("  failure with h =", hvals[ih])
        else:
            errs[ih] = np.linalg.norm(y-np.transpose(yref),1)
            if (ih == 0):
                print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i" %
                      (hvals[ih], errs[ih], solver.get_total_iters(),
                       solver.get_total_setups()))
            else:
                print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i,  rate = %.3f" %
                      (hvals[ih], errs[ih], solver.get_total_iters(),
                       solver.get_total_setups(),
                       np.log(errs[ih]/errs[ih-1])/np.log(hvals[ih]/hvals[ih-1])))
        solver.reset()

# DIRK tests
if (test_dirk):

    print("DIRK-3 tests:")
    gamma = 0.43586652150845906
    tau2 = 0.5*(1.0+gamma)
    A = np.array([[gamma, 0, 0],
                  [tau2-gamma, gamma, 0],
                  [-0.25*(6*gamma**2 - 16*gamma + 1), 0.25*(6*gamma**2 - 20*gamma + 5), gamma]], dtype=float)
    b = np.array([-0.25*(6*gamma**2 - 16*gamma + 1), 0.25*(6*gamma**2 - 20*gamma + 5), gamma], dtype=float)
    c = np.array([gamma, tau2, 1], dtype=float)
    errs = np.ones(hvals.size)
    for ih in range(hvals.size):
        y0 = np.copy(ex2.y0)
        t, y, success = dirk(ex2.f, tspan, y0, hvals[ih], A, b, c, solver)
        if (not success):
            print("  failure with h =", hvals[ih])
        else:
            errs[ih] = np.linalg.norm(y-np.transpose(yref),1)
            if (ih == 0):
                print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i" %
                      (hvals[ih], errs[ih], solver.get_total_iters(),
                       solver.get_total_setups()))
            else:
                print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i,  rate = %.3f" %
                      (hvals[ih], errs[ih], solver.get_total_iters(),
                       solver.get_total_setups(),
                       np.log(errs[ih]/errs[ih-1])/np.log(hvals[ih]/hvals[ih-1])))
        solver.reset()

    print("DIRK-4 tests:")
    gamma = np.cos(np.pi/18.0)/np.sqrt(3) + 0.5
    delta = 1.0/(6.0*(2.0*gamma-1)**2)
    A = np.array([[gamma, 0, 0],
                  [0.5-gamma, gamma, 0],
                  [2*gamma, 1.0-4*gamma, gamma]], dtype=float)
    b = np.array([delta, 1.0-2*delta, delta], dtype=float)
    c = np.array([gamma, 0.5, 1-gamma], dtype=float)
    errs = np.ones(hvals.size)
    for ih in range(hvals.size):
        y0 = np.copy(ex2.y0)
        t, y, success = dirk(ex2.f, tspan, y0, hvals[ih], A, b, c, solver)
        if (not success):
            print("  failure with h =", hvals[ih])
        else:
            errs[ih] = np.linalg.norm(y-np.transpose(yref),1)
            if (ih == 0):
                print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i" %
                      (hvals[ih], errs[ih], solver.get_total_iters(),
                       solver.get_total_setups()))
            else:
                print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i,  rate = %.3f" %
                      (hvals[ih], errs[ih], solver.get_total_iters(),
                       solver.get_total_setups(),
                       np.log(errs[ih]/errs[ih-1])/np.log(hvals[ih]/hvals[ih-1])))
        solver.reset()

    print("DIRK-5 tests:")
    A = np.array([[4024571134387.0/14474071345096, 0, 0, 0, 0],
                  [9365021263232.0/12572342979331, 4024571134387.0/14474071345096, 0, 0, 0],
                  [2144716224527.0/9320917548702, -397905335951.0/4008788611757, 4024571134387.0/14474071345096, 0, 0],
                  [-291541413000.0/6267936762551, 226761949132.0/4473940808273, -1282248297070.0/9697416712681, 4024571134387.0/14474071345096, 0],
                  [-2481679516057.0/4626464057815, -197112422687.0/6604378783090, 3952887910906.0/9713059315593, 4906835613583.0/8134926921134, 4024571134387.0/14474071345096]], dtype=float)
    b = np.array([-2522702558582.0/12162329469185, 1018267903655.0/12907234417901, 4542392826351.0/13702606430957, 5001116467727.0/12224457745473, 1509636094297.0/3891594770934], dtype=float)
    c = np.array([4024571134387.0/14474071345096, 5555633399575.0/5431021154178, 5255299487392.0/12852514622453, 3.0/20, 10449500210709.0/14474071345096], dtype=float)
    errs = np.ones(hvals.size)
    for ih in range(hvals.size):
        y0 = np.copy(ex2.y0)
        t, y, success = dirk(ex2.f, tspan, y0, hvals[ih], A, b, c, solver)
        if (not success):
            print("  failure with h =", hvals[ih])
        else:
            errs[ih] = np.linalg.norm(y-np.transpose(yref),1)
            if (ih == 0):
                print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i" %
                      (hvals[ih], errs[ih], solver.get_total_iters(),
                       solver.get_total_setups()))
            else:
                print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i,  rate = %.3f" %
                      (hvals[ih], errs[ih], solver.get_total_iters(),
                       solver.get_total_setups(),
                       np.log(errs[ih]/errs[ih-1])/np.log(hvals[ih]/hvals[ih-1])))
        solver.reset()


# Implicit LMM tests
if (test_implicit_lmm):

    print("BDF-1 tests:")
    alphas = np.array([1, -1], dtype=float)
    betas = np.array([1, 0], dtype=float)
    errs = np.ones(hvals.size)
    for ih in range(hvals.size):
        y0 = np.array([ex2.y0])
        t, y, success = implicit_lmm(ex2.f, tspan, y0, hvals[ih], alphas, betas, solver)
        if (not success):
            print("  failure with h =", hvals[ih])
        else:
            errs[ih] = np.linalg.norm(y-np.transpose(yref),1)
            if (ih == 0):
                print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i" %
                      (hvals[ih], errs[ih], solver.get_total_iters(),
                       solver.get_total_setups()))
            else:
                print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i,  rate = %.3f" %
                      (hvals[ih], errs[ih], solver.get_total_iters(),
                       solver.get_total_setups(),
                       np.log(errs[ih]/errs[ih-1])/np.log(hvals[ih]/hvals[ih-1])))
        solver.reset()

    print("BDF-2 tests:")
    alphas = np.array([1, -4.0/3.0, 1.0/3.0], dtype=float)
    betas = np.array([2.0/3.0, 0, 0], dtype=float)
    errs = np.ones(hvals.size)
    for ih in range(hvals.size):
        tvals = np.array((ex2.t0, ex2.t0+hvals[ih]))
        initconds = solve_ivp(ex2.f, tvals, ex2.y0, method='BDF', jac=ex2.J_dense,
                              t_eval=tvals, rtol=1e-8, atol=[1e-16, 1e-20, 1e-18])
        y0 = np.copy(np.transpose(initconds.y))
        tspan2 = np.copy(tspan)
        tspan2[0] = ex2.t0+hvals[ih]
        yref2 = np.copy(yref)
        yref2[:,0] = (initconds.y)[:,-1]
        t, y, success = implicit_lmm(ex2.f, tspan2, y0, hvals[ih], alphas, betas, solver)
        if (not success):
            print("  failure with h =", hvals[ih])
        else:
            errs[ih] = np.linalg.norm(y-np.transpose(yref2),1)
            if (ih == 0):
                print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i" %
                      (hvals[ih], errs[ih], solver.get_total_iters(),
                       solver.get_total_setups()))
            else:
                print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i,  rate = %.3f" %
                      (hvals[ih], errs[ih], solver.get_total_iters(),
                       solver.get_total_setups(),
                       np.log(errs[ih]/errs[ih-1])/np.log(hvals[ih]/hvals[ih-1])))
        solver.reset()

    print("BDF-3 tests:")
    alphas = np.array([1, -18.0/11.0, 9.0/11.0, -2.0/11.0], dtype=float)
    betas = np.array([6.0/11.0, 0, 0, 0], dtype=float)
    errs = np.ones(hvals.size)
    for ih in range(hvals.size):
        tvals = np.array((ex2.t0, ex2.t0+hvals[ih], ex2.t0+2*hvals[ih]))
        initconds = solve_ivp(ex2.f, np.array((ex2.t0, ex2.t0+2*hvals[ih])), ex2.y0, method='BDF',
                              jac=ex2.J_dense, t_eval=tvals, rtol=1e-8, atol=[1e-16, 1e-20, 1e-18])
        y0 = np.copy(np.transpose(initconds.y))
        tspan2 = np.copy(tspan)
        tspan2[0] = ex2.t0+2*hvals[ih]
        yref2 = np.copy(yref)
        yref2[:,0] = (initconds.y)[:,-1]
        t, y, success = implicit_lmm(ex2.f, tspan2, y0, hvals[ih], alphas, betas, solver)
        if (not success):
            print("  failure with h =", hvals[ih])
        else:
            errs[ih] = np.linalg.norm(y-np.transpose(yref2),1)
            if (ih == 0):
                print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i" %
                      (hvals[ih], errs[ih], solver.get_total_iters(),
                       solver.get_total_setups()))
            else:
                print("  h = %.12e,  error = %.12e,  iters = %i,  setups = %i,  rate = %.3f" %
                      (hvals[ih], errs[ih], solver.get_total_iters(),
                       solver.get_total_setups(),
                       np.log(errs[ih]/errs[ih-1])/np.log(hvals[ih]/hvals[ih-1])))
        solver.reset()
