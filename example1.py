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

def reference_solution(tvals):
    """
    Function that returns a high-accuracy reference solution to
    the IVP over a specified set of time outputs -- both the
    array of these time outputs and the solution at these outputs
    are returned.
    """
    yref = np.zeros(np.size(y0),np.size(tvals))
    return yref

def Jacobian_eigenvalues(t,y):
    """
    Function that returns the eigenvalues of the Jacobian at a
    specific set of input values.
    """
    import numpy.linalg as la
    return la.eig(J_dense(t,y))
