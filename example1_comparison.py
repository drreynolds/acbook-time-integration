# example1_comparison.py
#
# Script that runs various explicit methods on the Brusslator example,
# including built-in adaptive solvers.
#
# Daniel R. Reynolds
# Department of Mathematics
# Southern Methodist University

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from example1 import *

# generate solution plot
tspan = np.linspace(t0, tf, 501)
yref = reference_solution(tspan)
fig = plt.figure(figsize=(8,6), dpi=200)
plt.plot(tspan, yref[0,:], label='n(X)')
plt.plot(tspan, yref[1,:], label='n(Y)')
plt.xlabel('time')
plt.legend()
plt.savefig('Brusselator_solution.pdf')
plt.show()
