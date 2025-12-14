import numpy as np


def create_space_grid(x_start: float, x_end: float, dx: float) -> np.ndarray:
    """
    We create a 1D spatial grid from x_start to x_end (inclusive) with spacing dx.

    We also add dx/2 to be sure that endpoint is included despite floating-point rounding.
    """
    return np.arange(x_start, x_end + dx / 2.0, dx)


def create_time_grid(t_start: float, t_end: float, dt: float) -> np.ndarray:
    """
    In this part we create a 1D time grid from t_start to t_end inclusive with timestep dt.

    Also we add dt/2 to ensure the endpoint is included despite floating-point rounding.
    """
    return np.arange(t_start, t_end + dt / 2.0, dt)


def advect_1d_backward(
    C0: np.ndarray,
    U: float,
    dx: float,
    dt: float,
    nt: int,
    inlet_func=None,
    decay_rate: float = 0.0,
) -> np.ndarray:
    """
    This is backward 1D advection solver with optional first-order decay.

    PDE being solved (1D advection with optional first order decay):
    dC/dt = -U * dC/dx - decay_rate * C

    Discretisation (backward Euler in time, backward difference in space):
    (C_i^n - C_i^{n-1})/dt = -U_i * (C_i^n - C_{i-1}^n)/dx - decay_rate * C_i^n

    Eventually this produces an implicit *lower - triangular* system for C^n which can be solved 
    efficiently by forward substitution (loop from i=1 to nx-1).
    """
    # ensuring initial condition is a float numpy array
    C0 = np.asarray(C0, dtype=float)
    nx = C0.size
    
    # converting U into an array U_arr of length nx
    # this is result lets the solver treat scalar and spatially varying U in the same way
    if np.isscalar(U):
        U_arr = np.full(nx, float(U))
    else:
        U_arr = np.asarray(U, dtype=float)
        if U_arr.size != nx:
            raise ValueError("If U is an array it must have same length as C0")

    # allocating solution array: rows = time, columns = space 
    C = np.zeros((nt, nx), dtype=float)
    C[0, :] = C0.copy() # this is initial condition at time step n = 0

    # precomputing constants to avoid repeated division in the time loop 
    inv_dt = 1.0 / dt
    inv_dx = 1.0 / dx

    # this is the main time-stepping loop
    for n in range(1, nt):
        # new solution at time step n
        C_new = np.zeros(nx, dtype=float)

        # inlet boundary condition at x = 0
        # if inlet_func is not provided, we say keep inlet concentration fixed at initial value
        if inlet_func is None:
            C_new[0] = C[0, 0]  
        else:
            # this part evaluates boundary function at current physical time t = n*dt
            C_new[0] = float(inlet_func(n * dt))

        # this is interior velocities 
        # we solve for C_new[1:], while C_new[0] is already known from boundary condition.
        U_int = U_arr[1:]  

        # coefficients for the implicit formula at interior nodes
        # a_i * C_i^n = f_i + b_i * C_{i-1}^n
        # where:
        #  a_i = 1/dt + U_i/dx + decay_rate 
        #  b_i = U_i/dx
        #  f_i = (1/dt) * C_i^{n-1} 
        a = inv_dt + U_int * inv_dx + decay_rate      
        b = U_int * inv_dx                            
        f = inv_dt * C[n - 1, 1:]                    

        # here is forward substitution (lower triangular solve)
        # and since backward differencing uses C_{i-1}^n, we can compute left to right
        for i in range(1, nx):
            idx = i - 1 

            # right hand side includes known value from the node to the left (already computed)
            if i == 1:
                rhs = f[idx] + b[idx] * C_new[0]
            else:
                rhs = f[idx] + b[idx] * C_new[i - 1]

            # solves directly for C_new[i] since rhs only depends on already computed values (forward substitution) 
            C_new[i] = rhs / a[idx]

        # stores the completed time level
        C[n, :] = C_new

    return C


def run_base_case():
    """
    this is simple demo run which is useful for quick cheking 
    this creates domain, sets an initial condition and runs the solver
    """
    # this is spatial grid
    L = 20.0
    dx = 0.2
    x = create_space_grid(0.0, L, dx)
    nx = x.size

    # this is time grid
    t_end = 300.0
    dt = 10.0
    t = create_time_grid(0.0, t_end, dt)
    nt = t.size

    # velocity which is constant
    U = 0.1

    # initial condition 
    C0 = np.zeros(nx)
    C0[0] = 250.0

    # this is the boundary condition function at inlet (constant in time here)
    def inlet(t_n):
        return 250.0

    C = advect_1d_backward(C0, U, dx, dt, nt, inlet_func=inlet, decay_rate=0.0)
    return x, t, C
    
def advect_1d_backward_variable_u(
    C0: np.ndarray,
    U_series: np.ndarray,  
    dx: float,
    dt: float,
    nt: int,
    inlet_func=None,
    decay_rate: float = 0.0,
) -> np.ndarray:
    """ 
    this is the backward 1D advection solver but here velocity varies in time.

    here U_series is a time series of length nt
    and at each time step n we use U = U_series[n-1] to advance from n-1 to n.

    it is important to note that:
    this assumes spatially uniform velocity at each time step (same U across x)
    """
    # ensuring initial condition is float numpy array 
    C0 = np.asarray(C0, dtype=float)
    nx = C0.size

    # validates velocity time series
    U_series = np.asarray(U_series, dtype=float)
    if U_series.size != nt:
        raise ValueError("U_series must have length nt")

    # allocating solution array
    C = np.zeros((nt, nx), dtype=float)
    C[0, :] = C0.copy()

    #preocmputing constants 
    inv_dt = 1.0 / dt
    inv_dx = 1.0 / dx

    # time stepping
    for n in range(1, nt):
        U_n = float(U_series[n-1])   
        U_arr = np.full(nx, U_n)    # uniform in space at this time level

        # new time level
        C_new = np.zeros(nx, dtype=float)

        
        if inlet_func is None:
            C_new[0] = C[0, 0]
        else:
            C_new[0] = float(inlet_func(n * dt))

        # same structure as constant U solver)
        U_int = U_arr[1:]
        a = inv_dt + U_int * inv_dx + decay_rate   
        b = U_int * inv_dx                         
        f = inv_dt * C[n - 1, 1:]                  

        #forward substitution solve
        for i in range(1, nx):
            idx = i - 1
            if i == 1:
                rhs = f[idx] + b[idx] * C_new[0]
            else:
                rhs = f[idx] + b[idx] * C_new[i - 1]
            C_new[i] = rhs / a[idx]

        C[n, :] = C_new

    return C
