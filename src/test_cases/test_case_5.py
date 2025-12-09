# Variable Velocity Test

import numpy as np

def test_case_5(
    U_base=0.1,
    perturbation_level=0.10,
    num_realizations=5,
    L=20.0,
    dx=0.2,
    t_end=300.0,
    dt=10.0,
    C_in0=250.0,
):
  
    # Creates Grids
    x = create_space_grid(0.0, L, dx)
    t = create_time_grid(0.0, t_end, dt)
    
    nx = x.size
    nt = t.size

    # Initial Conditions
    C0 = np.zeros(nx)
    C0[0] = C_in0

    # inlet function (constant source)
    def inlet(time):
      return C_in0
    
    # Baseline Run (Constant U)
    C_baseline = advect_1d_backward(C0, U_base, dx, dt, nt, inlet_func=inlet)

    results = {"baseline": C_baseline, "perturbed": []}

    # Petrubed Velocity Runs
    for k in range(num_realizations):

        # 10% random perturbation: U(x) = U_base * (1 + 0.1 * random_noise)
        random_noise = np.random.uniform(-1, 1, nx)
        U_perturbed = U_base * (1 + perturbation_level * random_noise)

        C_perturbed = advect_1d_backward(C0, U_perturbed, dx, dt, nt, inlet_func=inlet)

        results["perturbed"].append({'U': U_perturbed, 'C': C_perturbed,})

    results["x"] = x
    results["t"] = t

    return results
