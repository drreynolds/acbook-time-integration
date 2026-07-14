# homework.py
#
# Hands-on exercise.
#
# Daniel R. Reynolds
# Department of Mathematics & Statistics
# University of Maryland Baltimore County

import numpy as np

# module-level global variables
y0 = np.array([1, 0, 0, 0, 0, 0, 0, 0.0057], dtype=float)
t0 = 0.0
tf = 320.0
k1 = 1.71
k2 = 0.43
k3 = 8.32
k4 = 8.75
k5 = 10.03
k6 = 0.035
k7 = 1.12
k8 = 1.745
k9 = 0.43
k10 = 280
k11 = 0.69
k12 = 1.81

# module functions
def f(t, y):
    """
    Right-hand side function, f(t,y), for the IVP.
    """
    return np.array([-k1*y[0] + k2*y[1] + k3*y[2] + 0.0007,
                     k1*y[0]-k4*y[1],
                     -k5*y[2] + k2*y[3] + k6*y[4],
                     k3*y[1] + k1*y[2] - k7*y[3],
                     -k8*y[4] + k9*y[5] + k9*y[6],
                     -k10*y[5]*y[7] + k11*y[3] + k1*y[4] - k9*y[5] + k11*y[6],
                     k10*y[5]*y[7] - k12*y[6],
                     -k10*y[5]*y[7] + k12*y[6]], dtype=float)

def J_dense(t, y):
    """
    Jacobian (in dense matrix format) of the right-hand side
    function, J(t,y) = df/dy, for the IVP.
    """
    return np.array([[-k1, k2, k3, 0, 0, 0, 0, 0],
                     [k1, -k4, 0, 0, 0, 0, 0, 0],
                     [0, 0, -k5, k2, k6, 0, 0, 0],
                     [0, k3, k1, -k7, 0, 0, 0, 0],
                     [0, 0, 0, 0, -k8, k9, k9, 0],
                     [0, 0, 0, k11, k1, -k10*y[7]-k9, k11, -k10*y[5]],
                     [0, 0, 0, 0, 0, k10*y[7], -k12, k10*y[5]],
                     [0, 0, 0, 0, 0, -k10*y[7], k12, -k10*y[5]]], dtype=float)


def reference_solution(N, reltol=1e-8, abstol=None):
    """
    Function that returns a high-accuracy reference solution to
    the IVP over a specified number of time outputs -- both the
    array of these time outputs and the solution at these outputs
    are returned.
    """
    from scipy.integrate import solve_ivp

    if not abstol:
        abstol = [1e-20, 1e-18, 1e-18]

    tvals = np.linspace(t0, tf, N)
    ivpsol = solve_ivp(f, (t0, tf), y0, method='BDF', jac=J_dense,
                       t_eval=tvals, rtol=reltol, atol=abstol)
    if not ivpsol.success:
        raise Exception("Failed to generate reference solution")
    return tvals, ivpsol.y
