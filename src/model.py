import numpy as np


def create_space_grid(x_start: float, x_end: float, dx: float) -> np.ndarray:
    return np.arange(x_start, x_end + dx / 2.0, dx)


def create_time_grid(t_start: float, t_end: float, dt: float) -> np.ndarray:
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

    C0 = np.asarray(C0, dtype=float)
    nx = C0.size

    if np.isscalar(U):
        U_arr = np.full(nx, float(U))
    else:
        U_arr = np.asarray(U, dtype=float)
        if U_arr.size != nx:
            raise ValueError("If U is an array it must have same length as C0")

    C = np.zeros((nt, nx), dtype=float)
    C[0, :] = C0.copy()

    inv_dt = 1.0 / dt
    inv_dx = 1.0 / dx

    for n in range(1, nt):
        C_new = np.zeros(nx, dtype=float)
        if inlet_func is None:
            C_new[0] = C[0, 0]  
        else:
            C_new[0] = float(inlet_func(n * dt))

       
        U_int = U_arr[1:]  
        a = inv_dt + U_int * inv_dx + decay_rate      
        b = U_int * inv_dx                            
        f = inv_dt * C[n - 1, 1:]                    

        
        for i in range(1, nx):
            idx = i - 1  
            if i == 1:
                rhs = f[idx] + b[idx] * C_new[0]
            else:
                rhs = f[idx] + b[idx] * C_new[i - 1]
            C_new[i] = rhs / a[idx]

        C[n, :] = C_new

    return C


def run_base_case():
   
    L = 20.0
    dx = 0.2
    x = create_space_grid(0.0, L, dx)
    nx = x.size

    t_end = 300.0
    dt = 10.0
    t = create_time_grid(0.0, t_end, dt)
    nt = t.size

    U = 0.1

    C0 = np.zeros(nx)
    C0[0] = 250.0

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

    C0 = np.asarray(C0, dtype=float)
    nx = C0.size

    U_series = np.asarray(U_series, dtype=float)
    if U_series.size != nt:
        raise ValueError("U_series must have length nt")

    C = np.zeros((nt, nx), dtype=float)
    C[0, :] = C0.copy()

    inv_dt = 1.0 / dt
    inv_dx = 1.0 / dx

    for n in range(1, nt):
        U_n = float(U_series[n-1])   
        U_arr = np.full(nx, U_n)    

        C_new = np.zeros(nx, dtype=float)

        
        if inlet_func is None:
            C_new[0] = C[0, 0]
        else:
            C_new[0] = float(inlet_func(n * dt))

        
        U_int = U_arr[1:]
        a = inv_dt + U_int * inv_dx + decay_rate   
        b = U_int * inv_dx                         
        f = inv_dt * C[n - 1, 1:]                  

        
        for i in range(1, nx):
            idx = i - 1
            if i == 1:
                rhs = f[idx] + b[idx] * C_new[0]
            else:
                rhs = f[idx] + b[idx] * C_new[i - 1]
            C_new[i] = rhs / a[idx]

        C[n, :] = C_new

    return C
