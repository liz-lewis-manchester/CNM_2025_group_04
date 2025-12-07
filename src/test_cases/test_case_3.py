import numpy as np
import sys
import os

THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.dirname(THIS_DIR)
if SRC_DIR not in sys.path:
  sys.path.append(SRC_DIR)

from model import create_space_grid, create_time_grid, advect_1d_backward

def run_test_case_3(
  U_values=None,
  dx_values=None,
  dt_values=None,
  L: float = 20.0,
  t_end: float = 300.0,
):

  if U_values is None:
    U_values = [0.05, 0.1, 0.2]

  if dx_values is None:
    dx_values = [0.1, 0.2, 0.5]

  if dt_values is None:
    dt_values = [1.0, 5.0, 10.0]

  results = {}

  for U in U_values:
    for dx in dx_values:
      for dt in dt_values:

        
        x = create_space_grid(0.0, L, dx)
        t = create_time_grid(0.0, t_end, dt)

        nx = x.size
        nt = t.size
      
        C0 = np.zeros(nx)
        C0[0] = 250.0

        def inlet(time):
          return 250

        C = advect_1d_backward(
          C0, 
          U=U,
          dx=dx,
          dt=dt,
          nt=nt,
          inlet_func=None,
        )

        results[(U, dx, dt)] = {"x": x, "t" : t, "C" : C}

        print(f"U={U}, dx={dx}, dt={dt}, nx={nx}, nt={nt}")
        
  return results

if __name__ == "__main__":
  results = run_test_case_3()
