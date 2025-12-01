import numpy as np
from src.model import solve_advection

def test_solver_runs_with_simple_input():
    nx = 5
    C0 = np.zeros(nx)
    C0[0] = 1.0
    U = 0.1
    dx = 0.2
    dt = 0.1
    nt = 3

    C = solve_advection(C0, U, dx, dt, nt)

    assert C.shape == (nt, nx)

