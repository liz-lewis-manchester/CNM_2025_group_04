import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import IPython.display as ipd

# the path setup so this script can import modules from src/ 
THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.dirname(THIS_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
# imports the solver utilities and animation function 
from model import create_space_grid, create_time_grid, advect_1d_backward
from plots import plot_space_time_snapshots, animate_advection



def run_test_case_3(
    U_values=None,
    dx_values=None,
    dt_values=None,
    L: float = 20.0,
    t_end: float = 300.0,
):
    """ 
    this is the sensitivity analysis which examines the sensitivity of the model results to:
    the physical/model parameter U which is the advection velocity 
    numerical discretisation parameters dx and dt 

    it is worth mentioning also that a constant inlet concentration is imposed (C(0,t)=constant) 
    in order to isolate the effects of parameter changes on the spatial distribution and propogation of the
    concentration profile, rather than on its magnitude.
    """
   # default parameter ranges (used if none are provided)
    if U_values is None:
        U_values = [0.05, 0.1, 0.2]

    if dx_values is None:
        dx_values = [0.1, 0.2, 0.5]

    if dt_values is None:
        dt_values = [1.0, 5.0, 10.0]

    # stores results from all simulations
    results = []

    # this is the parameter sweep: run solver for all combinations of U, dx, dt
    for U in U_values:
        for dx in dx_values:
            for dt in dt_values:

                x = create_space_grid(0.0, L, dx) # creates spatial and temporal gridds
                t = create_time_grid(0.0, t_end, dt)

                nx = x.size
                nt = t.size

                C0 = np.zeros(nx) # initial condition: non-zero concentration at inlet only
                C0[0] = 250.0

                C = advect_1d_backward(
                    C0,
                    U=U,
                    dx=dx,
                    dt=dt,
                    nt=nt,
                    inlet_func=None,
                )

                # simple metrics for sensitivity comparison 
                max_C = C.max() # maximum concentration in entire domain

                # location of concentration peak at final time
                peak_idx = np.argmax(C[-1, :])
                peak_x = x[peak_idx]

                print(
                    f"U={U}, dx={dx}, dt={dt} -> maxC={max_C:.2f}, peak_x={peak_x:.2f}"
                )

                # stores full solution and metrics 
                results.append({
                    "U": U,
                    "dx": dx,
                    "dt": dt,
                    "x": x,
                    "t": t,
                    "C": C,
                    "max_C": max_C,
                    "peak_x": peak_x,
                })
    # sensitivity plot: effect of dx (U and dt held constant)
    fig, ax = plt.subplots(figsize=(8,4))
    target_U = 0.1
    target_dt = 5.0

    for entry in results:
        if entry["U"] == target_U and entry["dt"] == target_dt:
            x, C = entry["x"], entry["C"]
            label = f"dx={entry['dx']}"
            ax.plot(x, C[-1, :], label=label)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("C (µg/m³)")
    ax.set_title("Sensitivity to dx (U=0.1 m/s, dt=5 s)")
    ax.legend()
    plt.show()

    # sensitivity plot: effect of dt (U and dx held constant)
    fig, ax = plt.subplots(figsize=(8, 4))
    target_U = 0.1
    target_dx = 0.1

    for entry in results:
        if entry["U"] == target_U and abs(entry["dx"] - target_dx) < 1e-12:
            x = entry["x"]
            C = entry["C"]
            label = f"dt={entry['dt']}"
            ax.plot(x, C[-1, :], label=label)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("C (µg/m³)")
    ax.set_title("Sensitivity to dt (U=0.1 m/s, dx=0.1 m)")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # sensitivity plot: effect of U (dx and dt held constant)
    fig, ax = plt.subplots(figsize=(8, 4))
    target_dx = 0.1
    target_dt = 5.0

    for entry in results:
        if abs(entry["dx"] - target_dx) < 1e-12 and entry["dt"] == target_dt:
            x = entry["x"]
            C = entry["C"]
            label = f"U={entry['U']}"
            ax.plot(x, C[-1, :], label=label)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("C (µg/m³)")
    ax.set_title("Sensitivity to U (dx=0.1 m, dt=5 s)")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # example animation for one representative case 
    example_case = results[0]
    anim = animate_advection(
        example_case["x"],
        example_case["t"],
        example_case["C"],
        title="Test Case 3 – Animation Example",
    )
    ipd.display(ipd.HTML(anim.to_jshtml()))

    return None # results are used for plotting only 

# allows this test to be run as a standalone script 
if __name__ == "__main__":
    results = run_test_case_3()
