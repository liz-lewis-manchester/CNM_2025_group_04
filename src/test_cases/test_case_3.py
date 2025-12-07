import numpy as np

def sensitivity_tests(U_values, dx_values, dt_values, L = 20, t_end = 300):

  results = {}

  for U in U_values:
    for dx in dx_values:
      for dt in dt_values:

        
        x = create_space_grid(0, L, dx)
        t = create_time_grid(0, t_end, dt)

        nx = x.size
        nt = t.size
      
        C0 = np.zeros(nx)
        C0[0] = 250

        def inlet(time):
          return 250

        C = advect_1d_backward(C0, U, dx, dt, nt, inlet_func=inlet)

        results[(U, dx, dt)] = {'x': x, 't' : t, 'C' : C}

        print('U =', {U}, 'dx=', {dx}, 'dt=', {dt}, 'nx=', {nx}, 'nt=', {nt})
        
  return results

U_values  = [0.05, 0.1, 0.2]
dx_values = [0.1, 0.2, 0.5]
dt_values = [1.0, 5.0, 10.0]

results = sensitivity_tests(U_values, dx_values, dt_values)

