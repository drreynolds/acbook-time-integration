# newton.py
#
# Daniel R. Reynolds
# Department of Mathematics
# Southern Methodist University

def newton(Ffcn, Jfcn, x0, maxiter=10, rtol=1e-3, atol=0.0, Jfreq=1):
    """
    Usage: sol = newton(F, J, x0, maxiter=10, rtol=1e-3, atol=1e-6, Jfreq=1)

    This implements a modified Newton method for approximating a root of the nonlinear
    system of equations F(x)=0.  Here x is a numpy array with n entries, and F(x) is a
    function that outputs an array with n entries.  The iteration ceases when

       || (xnew - xold) / (atol + rtol*|xnew|) ||_RMS < 1

    Required inputs:
       * F -- nonlinear residual function, F(x0) should return a numpy array n entries
       * J -- function to construct a scipy.sparse.linalg.LinearOperator object.  This
              should be callable as  Jsolver = J(x,rtol,abstol), where Jsolver is a
              scipy.sparse.linalg.LinearOperator object that solves the Newton system
              dF(x)/dx @ z = r for the numpy array z.  Thus the function J should:
                1. construct the Jacobian approximation, (dF(x)/dx), for the input x
                2. set up a linear system solver that takes as input the vector r
                3. wrap this solver as a scipy.sparse.linalg.LinearOperator object
              If the solver is iterative, then it should solve the linear system such that
                   norm(residual) <= max(rtol*norm(r), abstol),
              where both rtol and abstol are scalar 'float' values.
       * x0 -- initial guess at solution (numpy array with n entries)

    Optional inputs:
       * maxiter -- maximum allowed number of iterations (integer >0)
       * rtol -- relative solution tolerance (float, >= 1e-15)
       * atol -- absolute solution tolerance (float or numpy array with n entries, all >=0)
       * Jfreq -- frequency for calling J (i.e., J is called every Jfreq iterations)

    Output: solution structure with members
       * sol['x'] -- the approximate solution (numpy array with n entries)
       * sol['iters'] -- the number of iterations performed
       * sol['success'] -- True if iteration converged; False otherwise
    """

    # imports
    import numpy as np

    # set scalar-valued absolute tolerance for linear solver
    if (np.isscalar(atol)):
        abstol = atol
    else:
        abstol = np.average(atol)

    # initialize output structure
    sol = {'x': x0, 'iters' : 0, 'success' : False }

    # store nonlinear system size
    n = x0.size

    # evaluate initial residual
    F = Ffcn(sol['x'])

    # set up initial Jacobian solver
    Jsolver = Jfcn(sol['x'], rtol, abstol)

    # perform iteration
    for its in range(1,maxiter+1):

        # increment iteration counter
        sol['iters'] += 1

        # solve Newton linear system
        h = Jsolver.matvec(F)

        # compute Newton update, new guess at solution, new residual
        sol['x'] = sol['x'] - h

        # check for convergence
        if (np.linalg.norm(h / (atol + rtol*np.abs(sol['x'])))/np.sqrt(n) < 1):
            sol['success'] = True
            return sol

        # update nonlinear residual
        F = Ffcn(sol['x'])

        # update Jacobian every "Jfreq" iterations
        if (its % Jfreq == 0):
            Jsolver = Jfcn(sol['x'], rtol, abstol)

    # if we made it here, return with current solution (note that sol['success'] is still False)
    return sol
