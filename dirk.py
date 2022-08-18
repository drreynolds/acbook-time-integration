# dirk.py
#
# Daniel R. Reynolds
# Department of Mathematics
# Southern Methodist University
from implicit_solver import ImplicitSolver

def dirk(f, tspan, ycur, h, A, b, c, solver):
    """
    Usage: t, y, success = dirk(f, tspan, y0, h, A, b, c, solver)

    Time-stepping function with syntax similar to
    scipy.integrate.solve_ivp, and that uses a diagonally-implicit
    Runge--Kutta method for computing each internal step, for an
    ODE IVP of the form
       y' = f(t,y), t in tspan,
       y(t0) = y0.

    Inputs:  f      = function for ODE right-hand side, f(t,y)
             tspan  = times at which to store the computed solution
                      (must be sorted, and each must occur naturally in
                      the set tspan[0], tspan[0]+h, tspan[0]+2h , ...
                      [nd-array, shape(n_out)]
             y0     = initial condition [nd-array, shape(n)]
             h      = internal time step to use [float]
             A      = Butcher table coefficients [nd-array, shape(s,s)]
             b      = Butcher table coefficients [nd-array, shape(s)]
             c      = Butcher table coefficients [nd-array, shape(s)]
             solver = algebraic solver object to use
                      [ImplicitSolver, see implicit_solver.py]

    Outputs: t       = tspan [nd-array, shape(n_out)]
             y       = values of the solution at t [nd-arary, shape(n, n_out)]
             success = True if the solver traversed the interval,
                       false if an integration step failed [bool]
    """
    import numpy as np

    # verify that tspan values are separated by multiples of h
    for n in range(tspan.size-1):
        hn = tspan[n+1]-tspan[n]
        if (abs(round(hn/h) - (hn/h)) > 100*np.sqrt(np.finfo(h).eps)*abs(h)):
            raise ValueError("input values in tspan (%e,%e) are not separated by a multiple of h = %e" % (tspan[n],tspan[n+1],h))

    # verify that Butcher table has appropriate structure
    if (np.linalg.norm(np.triu(A,1)) > 10*np.finfo(float).eps):
        raise ValueError("input Butcher table must be lower-triangular, A =", A)
    if (np.shape(A)[0] != np.shape(A)[1]):
        raise ValueError("input Butcher table must be square, A =", A)
    if (np.shape(A)[0] != b.size):
        raise ValueError("incompatible Butcher table inputs, A =", A, ", b =", b)
    if (b.size != c.size):
        raise ValueError("incompatible Butcher table inputs, b =", b, ", c =", c)

    # initialize outputs, and set first entry corresponding to initial condition
    t = np.zeros(tspan.size)
    y = np.zeros((tspan.size,ycur.size))
    t[0] = tspan[0]
    y[0,:] = ycur

    # initialize internal data
    k = np.zeros((b.size,ycur.size))

    # loop over desired output times
    for iout in range(1,tspan.size):

        # determine how many internal steps are required
        N = round((tspan[iout]-tspan[iout-1])/h)

        # reset "current" t that will be evolved internally
        tcur = tspan[iout-1]

        # iterate over internal time steps to reach next output
        for n in range(N):

            # loop over stages
            for istage in range(c.size):

                # construct "data" for this stage solve
                data = np.copy(ycur)
                for jstage in range(istage):
                    data += h*A[istage,jstage]*k[jstage,:]

                # construct implicit residual and Jacobian solver for this stage
                tstage = tcur+h*c[istage]
                F = lambda zcur: zcur - data - h*A[istage,istage]*f(tstage,zcur)
                solver.setup_linear_solver(tstage, -h*A[istage,istage])

                # perform implicit solve, and return on solver failure
                z, iters, success = solver.solve(F, ycur)
                if (not success):
                    return [t, y, False]

                # store RHS at this stage
                k[istage,:] = f(tstage, z)

            # update time step solution and tcur
            for istage in range(b.size):
                ycur += h*b[istage]*k[istage,:]
            tcur += h

        # store current results in output arrays
        t[iout] = tcur
        y[iout,:] = ycur

    # return with "success" flag
    return [t, y, True]
