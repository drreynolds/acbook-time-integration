# acbook-time-integration

Code examples for time integration chapter of book, "Astrochemical modelling: practical aspects of microphysics in numerical simulations," edited by Stefano Bovino and Tommaso Grassi.

**Note: the repository currently contains only placeholder files.  When all codes are ready this text will be removed.**

These codes are grouped into the following categories:

* Running examples: each module should include functions that implement the IVP right-hand side, the RHS Jacobian (in dense, sparse, and Jacobian-vector product formats), the reference solution, and a function to return the eigenvalues of the Jacobian.  The module should also specify the initial condition, time interval, and any relevant parameters as module-level global variables

  * example1.py : functions for the "non-stiff" running example.
  * example2.py : functions for the "stiff" running example

* Shared utilities:

  * implicit_solver.py : module that defines an "ImplicitSolver" class, whose objects implement a modified Newton iteration with either dense, sparse, GMRES, or preconditioned GMRES inner linear solver.

* Simple time-steppers:

  * forward_euler.py : basic forward Euler time stepper
  * backward_euler.py : basic backward Euler time stepper
  * erk.py : basic explicit Runge--Kutta time stepper, that accepts Butcher table as input
  * dirk.py : basic diagonaly-implicit Runge--Kutta time stepper, that accepts Butcher table as input
  * explicit_lmm.py : basic explicit linear multistep time stepper, that accepts LMM coefficients as input
  * implicit_lmm.py : basic implicit linear multistep time stepper, that accepts LMM coefficients as input

* Advanced drivers:

  * example1_comparison.py : script that runs various explicit methods on example 1, including built-in adaptive solvers
  * example2_comparison.py : script that runs various implicit methods on example 2, including built-in adaptive solvers
