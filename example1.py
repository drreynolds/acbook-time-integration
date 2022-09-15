# example1.py
#
# Module containing functions related to the "non-stiff" running
# example (Brusselator).
#
# Daniel R. Reynolds
# Department of Mathematics
# Southern Methodist University

import numpy as np

# module-level global variables
y0 = np.array([1.0, 1.0], dtype=float)
t0 = 0.0
tf = 20.0

# problem-specific parameters
nA = 1.0
nB = 3.0
k1 = 1.0
k2 = 1.0
k3 = 1.0
k4 = 1.0

# module functions
def f(t,y):
    """
    Right-hand side function, f(t,y), for the IVP.
    """
    return np.array([ k1*nA - k2*nB*y[0] + k3*y[0]*y[0]*y[1] - k4*y[0],
                      k2*nB*y[0] - k3*y[0]*y[0]*y[1] ], dtype=float)

def J_dense(t,y):
    """
    Jacobian (in dense matrix format) of the right-hand side
    function, J(t,y) = df/dy.
    """
    return np.array([[ -k2*nB + 2.0*k3*y[0]*y[1] - k4, k3*y[0]*y[0] ],
                     [ k2*nB - 2.0*k3*y[0]*y[1], -k3*y[0]*y[0] ]], dtype=float)

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

def reference_solution(N, reltol=1e-8):
    """
    Function that returns a high-accuracy reference solution to
    the IVP over a specified number of time outputs -- both the
    array of these time outputs and the solution at these outputs
    are returned.
    """
    from scipy.integrate import solve_ivp
    tvals = np.linspace(t0, tf, N)
    ivpsol = solve_ivp(f, (t0,tf), y0, t_eval=tvals, rtol=reltol)
    if (not ivpsol.success):
        raise Exception("Failed to generate reference solution")
    return [tvals, ivpsol.y]

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
