import numpy as np

def solve_advection(C0, U, dx, dt, nt):
    """
    Solves the 1D advection equation using an upwind finite difference method.

    Parameters:
        C0 : array
            Initial concentration distribution (1D array).
        U : float
            Advection velocity (m/s).
        dx : float
            Spatial step (m).
        dt : float
            Time step (s).
        nt : int
            Number of time steps.

    Returns:
        numpy.ndarray:
            2D array of shape (nt, nx) giving concentration at each time.
    """
    C = C0.copy()
    nx = len(C)

    results = np.zeros((nt, nx))
    results[0, :] = C

    for n in range(1, nt):
        C_new = C.copy()
        for i in range(1, nx):
            C_new[i] = C[i] - U * (dt/dx) * (C[i] - C[i-1])
        C = C_new
        results[n] = C

    return results

