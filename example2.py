# example2.py
#
# Module containing functions related to the "stiff" running example
# (Oregonator).
#
# Daniel R. Reynolds
# Department of Mathematics
# Southern Methodist University

import numpy as np

# module-level global variables
y0 = np.array([5.025e-11, 6e-7, 7.236e-8], dtype=float)
t0 = 0.0
tf = 360.0
k1 = 2.57555802e8
k2 = 7.72667406e1
k3 = 1.28777901e7
k4 = 1.29421790e-2
k5 = 1.60972376e-1

# module functions
def f(t,y):
    """
    Right-hand side function, f(t,y), for the IVP.
    """
    return np.array([ -k1*y[0]*y[1] + k2*y[0] - k3*y[0]*y[0] + k4*y[1],
                      -k1*y[0]*y[1] - k4*y[1] + k5*y[2],
                      k2*y[0] - k5*y[2] ], dtype=float)

def J_dense(t,y):
    """
    Jacobian (in dense matrix format) of the right-hand side
    function, J(t,y) = df/dy, for the IVP.
    """
    return np.array([[ -k1*y[1] + k2 - 2.0*k3*y[0], -k1*y[0] + k4, 0 ],
                     [ -k1*y[1], -k1*y[0] - k4, k5 ],
                     [ k2, 0, -k5 ]], dtype=float)

def J_sparse(t,y):
    """
    Jacobian (in sparse matrix format) of the right-hand side
    function, J(t,y) = df/dy, for the IVP.
    """
    return csc_array(J_dense(t,y))

def J_matvec(t,y,v):
    """
    Jacobian-vector product function, J@v
    """
    return (J_dense(t,y)@v)

def reference_solution(N, reltol=1e-8, abstol=[1e-20, 1e-18, 1e-18]):
    """
    Function that returns a high-accuracy reference solution to
    the IVP over a specified number of time outputs -- both the
    array of these time outputs and the solution at these outputs
    are returned.
    """
    from scipy.integrate import solve_ivp
    tvals = np.linspace(t0, tf, N)
    ivpsol = solve_ivp(f, (t0,tf), y0, method='BDF', jac=J_dense,
                       t_eval=tvals, rtol=reltol, atol=abstol)
    if (not ivpsol.success):
        raise Exception("Failed to generate reference solution")
    return (tvals, ivpsol.y)

def Jacobian_eigenvalues(t,y):
    """
    Function that returns the eigenvalues of the Jacobian at a
    specific set of input values.
    """
    import numpy.linalg as la
    return la.eigvals(J_dense(t,y))

def maximum_stiffness(N):
    """
    Function to approximate the maximum eigenvalue spread over N
    evenly-spaced time points along the solution trajectory.
    """
    tspan,yref = reference_solution(N)
    evals = Jacobian_eigenvalues(tspan[0], yref[:,0])
    stiffness = np.max(np.abs(evals))/np.min(np.abs(evals))
    for i in range(1,np.size(tspan)):
        evals = Jacobian_eigenvalues(tspan[i], yref[:,i])
        stiffness = max(stiffness, np.max(np.abs(evals))/np.min(np.abs(evals)))
    return stiffness
