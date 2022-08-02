# example1.py
#
# Module containing functions related to the "non-stiff" running example.
#
# Daniel R. Reynolds
# Department of Mathematics
# Southern Methodist University

import numpy as np

# module-level global variables
y0 = np.array([0.0, 0.0, 0.0])
t0 = 0.0
tf = 1.0
# add other parameters as relevant

# module functions
def f(t,y):
    """
    Right-hand side function, f(t,y), for the IVP
    y'(t) = f(t,y), t > t0, y(t0) = y0.
    """

    return np.zeros(np.shape(y))

def J_dense(t,y):
    """
    Jacobian (in dense matrix format) of the right-hand side
    function, J(t,y) = df/dy, for the IVP
    y'(t) = f(t,y), t > t0, y(t0) = y0.
    """

    return np.zeros(np.size(y),np.size(y))

def J_sparse(t,y):
    """
    Jacobian (in sparse matrix format) of the right-hand side
    function, J(t,y) = df/dy, for the IVP
    y'(t) = f(t,y), t > t0, y(t0) = y0.
    """

def J_matvec(t,y,v):
    """
    Jacobian-vector product function, J@v, for the Jacobian of
    the right-hand side function, J(t,y) = df/dy, for the IVP
    y'(t) = f(t,y), t > t0, y(t0) = y0.
    """

    return np.zeros(np.shape(y))

def reference_solution():
    """
    Function that returns a high-accuracy reference solution to
    the IVP over a specified set of time outputs -- both the
    array of these time outputs and the solution at these outputs
    are returned.
    """
    tvals = np.linspace(t0,tf,101)
    yref = np.zeros(np.size(y0),101)
    return [tvals, yref]

def Jacobian_eigenvalues(t,y):
    """
    Function that returns the eigenvalues of the Jacobian at a
    specific set of input values.
    """
    import numpy.linalg as la
    return la.eig(J_dense(t,y))
