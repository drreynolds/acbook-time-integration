# implicit_lmm.py
#
# Daniel R. Reynolds
# Department of Mathematics
# Southern Methodist University
from implicit_solver import ImplicitSolver

def implicit_lmm_step(f, yarr, farr, t, h, alpha, beta, sol):
    """
    Usage: t, yarr, farr, sol, success = implicit_lmm_step(f, yarr, farr, t, h, alpha, beta, sol)

    Utility routine to take a single implicit LMM time step,
    where the inputs (t,yarr,farr,sol) are overwritten by the updated versions.
    If success==True then the step succeeded; otherwise it failed.
    """
    # create LMM residual and Jacobia solver for this step
    t += h
    data = (h*beta[1]/alpha[0])*farr[-1] - (alpha[1]/alpha[0])*yarr[-1]
    for i in range(2,alpha.size):
        data += (h*beta[i]/alpha[0])*farr[-i] - (alpha[i]/alpha[0])*yarr[-i]

    # create implicit residual and Jacobian solver for this step
    F = lambda ynew: ynew - data - (h*beta[0]/alpha[0])*f(t,ynew)
    sol.setup_linear_solver(t, -h*beta[0]/alpha[0])

    # perform implicit solve, and return on solver failure
    y, iters, success = sol.solve(F, yarr[-1])
    if (not success):
        return (t, yarr, farr, sol, False)

    # add current solution and RHS to queue, and remove oldest solution and RHS
    yarr.pop(0)
    yarr.append(y)
    farr.pop(0)
    farr.append(f(t,y))
    return (t, yarr, farr, sol, True)


def implicit_lmm(f, tspan, y0, h, alpha, beta, solver):
    """
    Usage: t, y, success = implicit_lmm(f, tspan, y0, h, alpha, beta, solver)

    Time-stepping function with syntax similar to
    scipy.integrate.solve_ivp, and that uses an implicit linear
    multistep method of the form
       \sum_{j=0}^{k-1} \alpha_j y_{n+1-j} = h\sum_{j=0}^{k-1} \beta_j f_{n+1-j},
    for computing each internal step, for an ODE IVP of the form
       y' = f(t,y), t in tspan,
       y(t0) = y0.

    Note: this requires that y0 has separate columns containing
    sufficiently accurate "initial" values for all previous LMM steps.

    Inputs:  f     = function for ODE right-hand side, f(t,y)
             tspan = times at which to store the computed solution
                     (must be sorted, and each must occur naturally in
                     the set tspan[0], tspan[0]+h, tspan[0]+2h , ...
                     [nd-array, shape(n_out)]
             y0    = initial condition [nd-array, shape(k-1,n)], sorted as
                     [y0(t0-(k-2)*h), ... y0(t0-h), y0(t0)]
             h     = internal time step to use [float]
             alpha = linear multistep coefficients, sorted as
                     [\alpha_0, \alpha_1, ... \alpha_{k-1}] [nd-array, shape(k)]
             beta  = linear multistep coefficients, sorted as
                     [\beta_0, \beta_1, ... \beta_{k-1}]  [nd-array, shape(k)]
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
            raise ValueError("input values in tspan (%e,%e) are not separated by a multiple of h" % (tspan[n],tspan[n+1]))

    # verify that input LMM coefficients are valid
    k = alpha.size
    if (abs(alpha[0]) == 0):
        raise ValueError("LMM coefficient must have nonzero alpha[0], ", alpha[0], " was input")
    if (beta.size != k):
        raise ValueError("LMM coefficient arrays must be the same length,", beta.size, " != ", k)
    if (np.shape(y0)[0] < (k-1)):
        raise ValueError("insufficient initial conditions provided, ", np.shape(y0)[0], " < ", alpha.size-1)

    # initialize outputs, and set first entry corresponding to initial condition
    t = np.zeros(tspan.size)
    y = np.zeros((tspan.size,y0.shape[1]))
    t[0] = tspan[0]
    y[0,:] = y0[-1,:]

    # initialize internal data
    fprev = []
    yprev = []
    for i in range(k-1):
        yprev.append(y0[i,:])
        fprev.append(f(tspan[0]-(k-2-i)*h, y0[i,:]))

    # loop over desired output times
    for iout in range(1,tspan.size):

        # determine how many internal steps are required
        N = int(round((tspan[iout]-tspan[iout-1])/h))

        # reset "current" t that will be evolved internally
        tcur = tspan[iout-1]

        # iterate over internal time steps to reach next output
        for n in range(N):

            # perform LMM update
            tcur, yprev, fprev, solver, step_success = implicit_lmm_step(f, yprev, fprev,
                                                                         tcur, h, alpha,
                                                                         beta, solver)
            if (not step_success):
                print("implicit_lmm error in time step at t =", tcur)
                return (t, y, False)

        # store current results in output arrays
        t[iout] = tcur
        y[iout,:] = yprev[-1]

    # return with "success" flag
    return (t, y, True)
