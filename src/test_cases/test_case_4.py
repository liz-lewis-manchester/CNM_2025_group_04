import numpy as np
import os
import sys

THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.dirname(THIS_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from model import create_space_grid, create_time_grid, advect_1d_backward

def run_test_case_4(
    decay_rates=None,
    C_in0: float = 250.0,
    L: float = 20.0,
    dx: float = 0.2,
    t_end: float = 300.0,
    dt: float = 10.0,
    U: float = 0.1,
):

  if decay_rates is None:
        decay_rates = [0.0, 0.005, 0.01, 0.02]

  results = {}

  for lam in decay_rates:

        x = create_space_grid(0.0, L, dx)
        t = create_time_grid(0.0, t_end, dt)
  
        nx = x.size
        nt = t.size

        C0 = np.zeros(nx)
        C0[0] = C_in0

        def inlet(time):
          return C_in0 * np.exp(-lam * time)

        C = advect_1d_backward(
            C0,
            U=U,
            dx=dx,
            dt=dt,
            nt=nt,
            inlet_func=inlet,
        )


        results[lam] = {"x": x, "t": t, "C": C}

        print(f"λ={lam}, U={U}, dx={dx}, dt={dt}, nx={nx}, nt={nt}")

  return results

if __name__ == "__main__":
    results = run_test_case_4()
