# example2_comparison.py
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
from example2 import *

# generate solution plot using built-in solver
tspan = np.linspace(t0, tf, 501)
ivpsol = solve_ivp(f, (t0,tf), y0, method='BDF', jac=J_dense, t_eval=tspan, rtol=1e-8, atol=[1e-16, 1e-20, 1e-18])
fig = plt.figure(figsize=(8,6), dpi=200)
ax = fig.add_subplot(311)
ax.plot(tspan, (ivpsol.y)[0,:])
ax.set_xlabel('time')
ax.set_ylabel('$n_1$')
ax.set_ylim([0, 6e-6])
ax.set_xlim([0, 400])
ax.ticklabel_format(axis='y', style='sci', scilimits=(-6,-6), useMathText=True)
ax = fig.add_subplot(312)
ax.semilogy(tspan, (ivpsol.y)[1,:])
ax.set_xlabel('time')
ax.set_ylabel('$n_2$')
ax.set_ylim([1e-10, 1e-2])
ax.set_xlim([0, 400])
ax = fig.add_subplot(313)
ax.plot(tspan, (ivpsol.y)[2,:])
ax.set_xlabel('time')
ax.set_ylabel('$n_3$')
ax.set_ylim([0, 8e-4])
ax.set_xlim([0, 400])
plt.savefig('Oregonator_solution.pdf')
plt.show()
