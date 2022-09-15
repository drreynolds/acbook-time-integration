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
