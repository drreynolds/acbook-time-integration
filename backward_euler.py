# backward_euler.py
#
# Daniel R. Reynolds
# Department of Mathematics
# Southern Methodist University
from implicit_solver import ImplicitSolver

def backward_euler(f, tspan, ycur, h, solver):
    """
    Usage: t, y, success = backward_euler(f, tspan, y0, h, solver)

    Time-stepping function with syntax similar to
    scipy.integrate.solve_ivp, and that uses the backward Euler
    (a.k.a., implicit Euler) method for computing each internal
    step, for an ODE IVP of the form
       y' = f(t,y), t in tspan,
       y(t0) = y0.

    Inputs:  f      = function for ODE right-hand side, f(t,y)
             tspan  = times at which to store the computed solution
                      (must be sorted, and each must occur naturally in
                      the set tspan[0], tspan[0]+h, tspan[0]+2h , ...
                      [nd-array, shape(n_out)]
             y0     = initial condition [nd-array, shape(n)]
             h      = internal time step to use [float]
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

    # initialize outputs, and set first entry corresponding to initial condition
    t = np.zeros(tspan.size)
    y = np.zeros((tspan.size,ycur.size))
    t[0] = tspan[0]
    y[0,:] = ycur

    # loop over desired output times
    for iout in range(1,tspan.size):

        # determine how many internal steps are required
        N = round((tspan[iout]-tspan[iout-1])/h)

        # reset "current" t that will be evolved internally
        tcur = tspan[iout-1]

        # iterate over internal time steps to reach next output
        for n in range(N):

            # update tcur for this step
            tcur += h

            # create implicit residual and Jacobian solver for this step
            F = lambda ynew: ynew - ycur - h*f(tcur,ynew)
            solver.setup_linear_solver(tcur, -h)

            # perform implicit solve, and return on solver failure
            ycur, iters, success = solver.solve(F, ycur)
            if (not success):
                print("backward_euler warning: solver failure in computing output %i, at step %i, t = %f (iters = %i)" %
                      (iout, n, tcur, iters))
                return [t, y, False]

        # store current results in output arrays
        t[iout] = tcur
        y[iout,:] = ycur

    # return with "success" flag
    return [t, y, True]
