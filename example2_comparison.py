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
from scipy.integrate import solve_ivp
import example2 as ex2

print("Oregonator IVP Tests")

# generate solution plot
tspan,yref = ex2.reference_solution(501)
fig = plt.figure(figsize=(8,6), dpi=200)
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
hvals = interval/Nout/np.array([10, 20, 30, 40, 50, 60], dtype=float)
