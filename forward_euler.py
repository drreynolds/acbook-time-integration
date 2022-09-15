# forward_euler.py
#
# Daniel R. Reynolds
# Department of Mathematics
# Southern Methodist University

def forward_euler(f, tspan, ycur, h):
    """
    Usage: t, y, success = forward_euler(f, tspan, y0, h)

    Time-stepping function with syntax similar to
    scipy.integrate.solve_ivp, and that uses the forward Euler
    (a.k.a., explicit Euler) method for computing each internal
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
        N = int(round((tspan[iout]-tspan[iout-1])/h))

        # reset "current" t that will be evolved internally
        tcur = tspan[iout-1]

        # iterate over internal time steps to reach next output
        for n in range(N):

            # perform forward Euler update, and update tcur
            ycur += h*f(tcur,ycur)
            tcur += h

        # store current results in output arrays
        t[iout] = tcur
        y[iout,:] = ycur

    # return with "success" flag
    return [t, y, True]
