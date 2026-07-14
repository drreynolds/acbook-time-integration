#!/usr/bin/env python3
#
# Script that runs solve_ivp with various methods for the homework.
#
# Daniel R. Reynolds
# Department of Mathematics & Statistics
# University of Maryland Baltimore County

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import homework as hw
import time

print("Homework Tests")

# use solve_ivp with BDF method
print(" ")
print("Running solve_ivp with BDF method...")
tstart = time.time()
ivpsol = solve_ivp(hw.f, (hw.t0, hw.tf), hw.y0, method='BDF', jac=hw.J_dense,
                   rtol=1e-5, atol=1e-10)
tend = time.time()
print("Execution time: %.4f seconds" % (tend - tstart))
print("Number of time steps: %d" % len(ivpsol.t))
print("Number of function evaluations: %d" % ivpsol.nfev)
print("Number of Jacobian evaluations: %d" % ivpsol.njev)
print("Number of LU decompositions: %d" % ivpsol.nlu)

# use solve_ivp with RK45 method
print(" ")
print("Running solve_ivp with RK45 method...")
tstart = time.time()
ivpsol = solve_ivp(hw.f, (hw.t0, hw.tf), hw.y0, method='RK45', 
                   rtol=1e-5, atol=1e-10)
tend = time.time()
print("Execution time: %.4f seconds" % (tend - tstart))
print("Number of time steps: %d" % len(ivpsol.t))
print("Number of function evaluations: %d" % ivpsol.nfev)

# plot the solution
plt.figure(figsize=(8, 6), dpi=100)
plt.subplot(421)
plt.plot(ivpsol.t, ivpsol.y[0, :])
plt.xlabel('t')
plt.ylabel('$n_1$')
plt.subplot(422)
plt.semilogy(ivpsol.t, ivpsol.y[1, :])
plt.xlabel('t')
plt.ylabel('$n_2$')
plt.subplot(423)
plt.plot(ivpsol.t, ivpsol.y[2, :])
plt.xlabel('t')
plt.ylabel('$n_3$')
plt.subplot(424)
plt.semilogy(ivpsol.t, ivpsol.y[3, :])
plt.xlabel('t')
plt.ylabel('$n_4$')
plt.subplot(425)
plt.semilogy(ivpsol.t, ivpsol.y[4, :])
plt.xlabel('t')
plt.ylabel('$n_5$')
plt.subplot(426)
plt.semilogy(ivpsol.t, ivpsol.y[5, :])
plt.xlabel('t')
plt.ylabel('$n_6$')
plt.subplot(427)
plt.semilogy(ivpsol.t, ivpsol.y[6, :])
plt.xlabel('t')
plt.ylabel('$n_7$')
plt.subplot(428)
plt.semilogy(ivpsol.t, ivpsol.y[7, :])
plt.xlabel('t')
plt.ylabel('$n_8$')
plt.tight_layout()
plt.savefig('Homework_solution.pdf')
plt.show()

