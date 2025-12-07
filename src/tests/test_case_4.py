import numpy as np

def exponential_inlet_test(decay_rates, C_in0=250.0, L=20.0, dx=0.2, t_end=300.0, dt=10.0, U=0.1,):

  results = {}

  for l in decay_rates:

        # Creates grids
        x = create_space_grid(0.0, L, dx)
        t = create_time_grid(0.0, t_end, dt)
  
        nx = x.size
        nt = t.size

        # initial conditions: pollutant present only at inlet initially
        C0 = np.zeros(nx)
        C0[0] = C_in0

        # inlet boundary: exponentially decaying
        def inlet(time):
          return C_in0 * np.exp(-l * time)

        C = advect_1d_backward(C0, U, dx, dt, nt, inlet_func=inlet)

        results[l] = {"x": x, "t": t, "C": C}

        print('lambda =', l)

        print('U =', {U}, 'dx=', {dx}, 'dt=', {dt}, 'nx=', {nx}, 'nt=', {nt})

  return results

# Imput decay rates to be tested here:
decay_rates = [0.0, 0.005, 0.01, 0.02]   # λ values to compare
results = exponential_inlet_test(decay_rates)

