#!/usr/bin/env python
#
# Driver to examine effect of "atol" input on performance solve_ivp for the
# Oregonator test problem.
#
# Daniel R. Reynolds
# Department of Mathematics
# Southern Methodist University

import numpy as np
import time
from scipy.integrate import solve_ivp
import numpy.linalg as la
from example2 import *

# time resolution for checking solution accuracy
N = 500

# utility routine to run Oregonator test for a given atol array and print
# solver statistics (always uses rtol = 1e-4)
def runtest(reltol,abstol):
    # run test with this abstol input
    tic = time.perf_counter()
    ivpsol = solve_ivp(f, (t0,tf), y0, method='BDF', jac=J_dense,
                       rtol=reltol, atol=abstol)
    toc = time.perf_counter()
    # generate reference solution at these same times
    refsol = solve_ivp(f, (t0,tf), y0, method='BDF', jac=J_dense,
                       t_eval=ivpsol.t, rtol=1e-8, atol=[1e-20, 1e-18, 1e-18])
    if (not ivpsol.success):
        print("   %.0e | %.0e %.0e %.0e | solver failure"
              % (reltol, abstol[0], abstol[1], abstol[2]))
    else:
        print("   %.0e | %.0e %.0e %.0e | %5i | %7.4f | %.6f"
              % (reltol, abstol[0], abstol[1], abstol[2],
                 np.size(ivpsol.t),
                 la.norm(refsol.y-ivpsol.y, np.inf)/la.norm(refsol.y, np.inf)*100,
                 toc-tic ))
    return ivpsol

# print header
print("\nTolerance performance tests for ivp_sol on Oregonator problem:\n")
print("   rtol  |        atol       | steps | % error | runtime")
print("  ========================================================")

# run tests for a variety of atol inputs
runtest(1e-4, [1e-4, 1e-4, 1e-4])
runtest(1e-4, [1e-5, 1e-5, 1e-5])
runtest(1e-4, [1e-6, 1e-6, 1e-6])
runtest(1e-4, [1e-7, 1e-7, 1e-7])
runtest(1e-4, [1e-8, 1e-8, 1e-8])
runtest(1e-4, [1e-9, 1e-9, 1e-9])
runtest(1e-4, [1e-10, 1e-10, 1e-10])
runtest(1e-4, [1e-11, 1e-11, 1e-11])
runtest(1e-4, [1e-12, 1e-12, 1e-12])
runtest(1e-4, [1e-13, 1e-13, 1e-13])
runtest(1e-4, [1e-13, 1e-12, 1e-12])
runtest(1e-4, [1e-13, 1e-11, 1e-11])
ivpsol = runtest(1e-4, [1e-13, 1e-10, 1e-10])

# print footer
print("  ========================================================\n")

# output some solution statistics
n1 = np.abs((ivpsol.y)[0,:])
n2 = np.abs((ivpsol.y)[1,:])
n3 = np.abs((ivpsol.y)[2,:])
print("\nSolution value statistics over time interval:")
print("   n1:  |max| = %.2e,  |min| = %.2e,  |avg| = %.2e" %
      (np.max(n1), np.min(n1), np.sum(n1)/np.size(n1)))
print("   n2:  |max| = %.2e,  |min| = %.2e,  |avg| = %.2e" %
      (np.max(n2), np.min(n2), np.sum(n2)/np.size(n2)))
print("   n3:  |max| = %.2e,  |min| = %.2e,  |avg| = %.2e\n" %
      (np.max(n3), np.min(n3), np.sum(n3)/np.size(n3)))
