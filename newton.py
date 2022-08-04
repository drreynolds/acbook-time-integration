# newton.py
#
# Module containing functions for performing a modified or inexact Newton nonlinear solver.
#
# Daniel R. Reynolds
# Department of Mathematics
# Southern Methodist University

def newton_dense(Ffcn, Jfcn, x0, maxiter=10, rtol=1e-3, atol=0.0, lag=0):
    """
    Usage: sol = newton_dense(F, J, x0, maxiter=10, rtol=1e-3, atol=1e-6, lag=0)

    This routine uses a modified Newton method to approximate a root of
    the nonlinear system of equations F(x)=0.  Here x is an array with n entries, and
    F(x) is a function that outputs an array with n entries.  The iteration ceases
    when the following condition is met:

       || (xnew - xold) / (atol + rtol*|xnew|) ||_RMS < 1

    Required inputs:
       * F -- nonlinear residual function, F(x0) should return a numpy array n entries
       * J -- Jacobian function, J(x0) should return a n*n numpy 2D array
       * x0 -- initial guess at solution (numpy array with n entries)

    Optional inputs:
       * maxiter -- maximum allowed number of iterations (integer >0)
       * rtol -- relative solution tolerance (float, >= 1e-15)
       * atol -- absolute solution tolerance (float or numpy array with n entries, all >=0)
       * lag -- # iterations to skip before recomputing Jacobian (integer, >=0); note that
              this is only relevant for dense and sparse linear solvers

    Output: solution dictionary with members
       * sol['x'] -- the approximate solution (numpy array with n entries)
       * sol['iters'] -- the number of iterations performed
       * sol['success'] -- True if iteration converged; False otherwise
    """

    # imports
    import numpy as np
    from scipy.linalg import lu_factor
    from scipy.linalg import lu_solve

    # initialize output structure
    sol = {'x': x0, 'iters' : 0, 'success' : False }

    # store nonlinear system size
    n = x0.size

    # evaluate initial residual
    F = Ffcn(sol['x'])

    # set up and factor initial Jacobian; return on LU factorization failure
    J = Jfcn(sol['x'])
    try:
        lu, piv = lu_factor(J)
    except:
        return sol

    # perform iteration
    for its in range(1,maxiter+1):

        # increment iteration counter
        sol['iters'] += 1

        # solve Newton linear system; return on LU solver failure
        try:
            h = lu_solve((lu, piv), F)
        except:
            return sol

        # compute Newton update, new guess at solution, new residual
        sol['x'] = sol['x'] - h

        # check for convergence
        if (np.linalg.norm(h / (atol + rtol*np.abs(sol['x'])))/np.sqrt(n) < 1):
            sol['success'] = True
            return sol

        # update nonlinear residual
        F = Ffcn(sol['x'])

        # update Jacobian every "lag" iterations; return on LU factorization failure
        if (its % (lag+1) == 0):
            J = Jfcn(sol['x'])
            try:
                lu, piv = lu_factor(J)
            except:
                return sol

    # if we made it here, return with current solution (note that sol['success'] is still False)
    return sol


def newton_sparse(Ffcn, Jfcn, x0, maxiter=10, rtol=1e-3, atol=0.0, lag=0):
    """
    Usage: sol = newton_sparse(F, J, x0, maxiter=10, rtol=1e-3, atol=1e-6, lag=0)

    This routine uses a modified Newton method to approximate a root of
    the nonlinear system of equations F(x)=0.  Here x is an array with n entries, and
    F(x) is a function that outputs an array with n entries.  The iteration ceases
    when the following condition is met:

      || (xnew - xold) / (atol + rtol*|xnew|) ||_RMS < 1

    Required inputs:
       * F -- nonlinear residual function, F(x0) should return a numpy array n entries
       * J -- Jacobian function, J(x0) should return a n*n scipy.sparse matrix
       * x0 -- initial guess at solution (numpy array with n entries)

    Optional inputs:
       * maxiter -- maximum allowed number of iterations (integer >0)
       * rtol -- relative solution tolerance (float, >= 1e-15)
       * atol -- absolute solution tolerance (float or numpy array with n entries, all >=0)
       * lag -- # iterations to skip before recomputing Jacobian (integer, >=0); note that
              this is only relevant for dense and sparse linear solvers

    Output: solution structure with members
       * sol['x'] -- the approximate solution (numpy array with n entries)
       * sol['iters'] -- the number of iterations performed
       * sol['success'] -- True if iteration converged; False otherwise
    """

    # imports
    import numpy as np
    from scipy.sparse.linalg import factorized

    # initialize output structure
    sol = {'x': x0, 'iters' : 0, 'success' : False }

    # store nonlinear system size
    n = x0.size

    # evaluate initial residual
    F = Ffcn(sol['x'])

    # set up and factor initial Jacobian; return on LU factorization failure
    J = Jfcn(sol['x'])
    try:
        sparse_solver = factorized(J)
    except:
        return sol

    # perform iteration
    for its in range(1,maxiter+1):

        # increment iteration counter
        sol['iters'] += 1

        # solve Newton linear system; return on linear solver failure
        try:
            h = sparse_solver(F)
        except:
            return sol

        # compute Newton update, new guess at solution, new residual
        sol['x'] = sol['x'] - h

        # check for convergence
        if (np.linalg.norm(h / (atol + rtol*np.abs(sol['x'])))/np.sqrt(n) < 1):
            sol['success'] = True
            return sol

        # update nonlinear residual
        F = Ffcn(sol['x'])

        # update Jacobian every "lag" iterations
        if (its % (lag+1) == 0):
            J = Jfcn(sol['x'])
            try:
                sparse_solver = factorized(J)
            except:
                return sol

    # if we made it here, return with current solution (note that sol['success'] is still False)
    return sol



def newton_gmres(Ffcn, Jfcn, x0, maxiter=10, rtol=1e-3, atol=0.0, Pfcn=0):
    """
    Usage: sol = newton_gmres(F, J, x0, maxiter=10, rtol=1e-3, atol=1e-6, P=pcond)

    This routine uses an inexact Newton's method to approximate a root of
    the nonlinear system of equations F(x)=0.  Here x is an array with n entries, and
    F(x) is a function that outputs an array with n entries.  The iteration ceases
    when the following condition is met:

       || (xnew - xold) / (atol + rtol*|xnew|) ||_RMS < 1

    Required inputs:
       * F -- nonlinear residual function, F(x0) should return a numpy array n entries
       * J -- Jacobian-vector product function, J(x0,v) should compute the product, J(x0)@v
       * x0 -- initial guess at solution (numpy array with n entries)

    Optional inputs:
       * maxiter -- maximum allowed number of iterations (integer >0)
       * rtol -- relative solution tolerance (float, >= 1e-15)
       * atol -- absolute solution tolerance (float or numpy array with n entries, all >=0)
       * P -- if supplied, P(x0,v) should solve the linear system, P(x0)@z = v, where P(x0)
              is an approximation of the Jacobian

    Output: solution structure with members
       * sol['x'] -- the approximate solution (numpy array with n entries)
       * sol['iters'] -- the number of iterations performed
       * sol['success'] -- True if iteration converged; False otherwise
    """

    # imports
    import numpy as np
    from scipy.sparse.linalg import LinearOperator
    from scipy.sparse.linalg import gmres

    # initialize output structure
    sol = {'x': x0, 'iters' : 0, 'success' : False }

    # store nonlinear system size
    n = x0.size

    # evaluate initial residual
    F = Ffcn(sol['x'])

    # set up initial Jacobian-vector product, and [optionally] preconditioning operator
    Jv = lambda v: Jfcn(sol['x'],v)
    J = LinearOperator((n,n), Jv)
    if (Pfcn != 0):
        Prec = lambda v: Pfcn(sol['x'],v)
        P = LinearOperator((n,n), Prec)

    # set scalar-valued absolute tolerance for GMRES solver
    if (np.isscalar(atol)):
        abstol = atol
    else:
        abstol = np.average(atol)

    # perform iteration
    for its in range(1,maxiter+1):

        # increment iteration counter
        sol['iters'] += 1

        # solve Newton linear system; return with failure on linear solver failure
        if (Pfcn != 0):
            h, exitCode = gmres(J, F, tol=rtol, atol=abstol, M=P)
        else:
            h, exitCode = gmres(J, F, tol=rtol, atol=abstol)
        if (exitCode < 0):
            return sol

        # compute Newton update, new guess at solution, new residual
        sol['x'] = sol['x'] - h

        # check for convergence
        if (np.linalg.norm(h / (atol + rtol*np.abs(sol['x'])))/np.sqrt(n) < 1):
            sol['success'] = True
            return sol

        # update nonlinear residual
        F = Ffcn(sol['x'])

    # if we made it here, return with current solution (note that sol['success'] is still False)
    return sol



def newton(Ffcn, Jfcn, x0, maxiter=10, rtol=1e-3, atol=0.0, lag=0, lsolve='dense', Pfcn=0):
    """
    Usage: sol = newton(F, J, x0, maxiter=10, rtol=1e-3, atol=1e-6, lag=0, lsolve='dense', P=pcond)

    This is a wrapper routine to linear-algebra-specific implementations of the modified Newton
    method for dense/sparse matrices, or the inexact Newton method when using the GMRES linear solver.

    We approximate a root of the nonlinear system of equations F(x)=0.  Here x is an array with n
    entries, and F(x) is a function that outputs an array with n entries.  The iteration ceases
    when the following condition is met:

       || (xnew - xold) / (atol + rtol*|xnew|) ||_RMS < 1

    Required inputs:
       * F -- nonlinear residual function, F(x0) should return a numpy array n entries
       * J -- Jacobian function, with inputs/output dependent on "lsolve":
              'dense' -- J(x0) should return a n*n numpy 2D array
              'sparse' -- J(x0) should return a n*n scipy.sparse matrix
              'iterative' -- J(x0,v) should perform the Jacobian-vector product, J(x0)@v
       * x0 -- initial guess at solution (numpy array with n entries)

    Optional inputs:
       * maxiter -- maximum allowed number of iterations (integer >0)
       * rtol -- relative solution tolerance (float, >= 1e-15)
       * atol -- absolute solution tolerance (float or numpy array with n entries, all >=0)
       * lag -- # iterations to skip before recomputing Jacobian (integer, >=0); note that
              this is only relevant for dense and sparse linear solvers
       * lsolve -- type of linear algebra to use ('dense', 'sparse', 'iterative')
       * P -- if supplied, P(x0,v) should solve the linear system, P(x0)@z = v, where P(x0)
              is an approximation of the Jacobian

    Output: solution structure with members
       * sol['x'] -- the approximate solution (numpy array with n entries)
       * sol['iters'] -- the number of iterations performed
       * sol['success'] -- True if iteration converged; False otherwise
    """

    # imports
    import numpy as np

    # overwrite illegal input arguments with default values
    if (int(maxiter) < 1):
        maxit = 10
    if (rtol < 1e-15):
        rtol = 1e-3
    if (np.isscalar(atol)):
        if (atol < 0):
            atol = 0
    else:
        if (np.min(atol) < 0):
            atol = np.abs(atol)
    if (lag < 0):
        lag = 0

    # return an error if x0 isn't a numpy array
    if (not isinstance(x0, np.ndarray)):
        raise ValueError('newton error: initial guess x0 must be a numpy array')

    # call the appropriate implementation of Newton's method based on 'lsolve'
    if (lsolve == 'dense'):
        return newton_dense(Ffcn, Jfcn, x0, maxiter, rtol, atol, lag)
    elif (lsolve == 'sparse'):
        return newton_sparse(Ffcn, Jfcn, x0, maxiter, rtol, atol, lag)
    elif (lsolve == 'iterative'):
        return newton_gmres(Ffcn, Jfcn, x0, maxiter, rtol, atol, Pfcn)
    else:
        raise ValueError('newton error: lsolve must be one of dense/sparse/iterative')
