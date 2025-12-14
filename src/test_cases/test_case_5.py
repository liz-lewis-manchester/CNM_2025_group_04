import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# path setup so the script can import modules from src/
THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.dirname(THIS_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# import solver utilities 
from model import (
    create_space_grid,
    create_time_grid,
    advect_1d_backward,
    advect_1d_backward_variable_u,
)
from plots import plot_space_time_snapshots # imports plotting utility 


def run_test_case_5(
    U0: float = 0.1,     # mean velocity 
    dx: float = 0.2,     # spatial step size
    dt: float = 10.0,    # time step size
    L: float = 20.0,     # domain length
    t_end: float = 300.0 # final simulation time 
) -> None:
    """ 
    this test investigates how a variable stream velocity profile alters the model results
    and we will add a 10% random perturbation to a constant velocity profile.
    """
    # creating spatial and temporal grids
    x = create_space_grid(0.0, L, dx)
    t = create_time_grid(0.0, t_end, dt)
    nt = t.size
    nx = x.size

    # initial condition: non-zero concentration at inlet only 
    C0 = np.zeros(nx)
    C0[0] = 250.0

    # case 1: constant velocity U=U0
    C_const = advect_1d_backward(
        C0,
        U=U0,
        dx=dx,
        dt=dt,
        nt=nt,
        inlet_func=None, # inlet held constant at initial value 
        decay_rate=0.0,
    )

    # case 2: time varying velocity U(t)
    # velocity fluctuates randomly around U0
    rng = np.random.default_rng(seed=0) # fixed seed for reproducibility  
    noise = 0.1 * rng.standard_normal(nt)  # 10% Gaussian noise
    U_series = U0 * (1.0 + noise)

    # ensuring velocity remains physically meaningful (positive)
    U_series = np.clip(U_series, 0.01, None)

    # running the solver with time dependant velocity 
    C_var = advect_1d_backward_variable_u(
        C0,
        U_series=U_series,
        dx=dx,
        dt=dt,
        nt=nt,
        inlet_func=None,
        decay_rate=0.0,
    )

    # creating results directory 
    results_dir = os.path.join(SRC_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)


    # 1) Velocity vs time
    plt.figure(figsize=(8, 4))
    plt.plot(t, U_series, label="U(t) with 10% noise")
    plt.axhline(U0, color="k", linestyle="--", label="constant U")
    plt.xlabel("t (s)")
    plt.ylabel("Velocity U (m/s)")
    plt.title("Test Case 5 – Variable velocity profile")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    vel_path = os.path.join(results_dir, "test_case_5_velocity_profile.png")
    plt.savefig(vel_path, dpi=200)
    plt.show()
    print(f"Saved velocity plot to {vel_path}")

    # 2) Final concentration profiles: constant vs variable U
    plt.figure(figsize=(8, 4))
    plt.plot(x, C_const[-1, :], label="constant U")
    plt.plot(x, C_var[-1, :], label="variable U(t)")
    plt.xlabel("x (m)")
    plt.ylabel("C (µg/m³)")
    plt.title("Test Case 5 – Effect of variable velocity on final profile")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    prof_path = os.path.join(results_dir, "test_case_5_final_profiles.png")
    plt.savefig(prof_path, dpi=200)
    plt.show()
    print(f"Saved final profile plot to {prof_path}")

    # 3) Space–time plot for variable U(t)
    st_path = os.path.join(results_dir, "test_case_5_space_time.png")
    plot_space_time_snapshots(
        x,
        t,
        C_var,
        snapshots=None,
        title="Test Case 5 – Variable U(t) space–time plot",
        savepath=st_path,
        show=True,
    )
    print(f"Saved space–time plot to {st_path}")

    print("Test Case 5 complete.")

# allows this test to be run as a standalone script
if __name__ == "__main__":
    run_test_case_5()
