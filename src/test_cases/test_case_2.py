import os
import sys
import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.dirname(THIS_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from model import create_space_grid, create_time_grid, advect_1d_backward
from plots import plot_space_time_snapshots


def run_test_case_2():
    """
    Test Case 2:
    - Read initial condition from CSV
    - Interpolate to computational grid
    - Run backward 1D advection
    - Save arrays + plot using central plotting function
    """

    dx = 0.2
    dt = 10.0
    t_end = 300.0
    U = 0.1

    data_path = os.path.join(SRC_DIR, "data", "initial_conditions.csv")
    df = pd.read_csv(data_path, encoding="latin1")

    x_data = df.iloc[:, 0].to_numpy()
    C_data = df.iloc[:, 1].to_numpy()

    x_min, x_max = x_data.min(), x_data.max()
    x = create_space_grid(x_min, x_max, dx)
    t = create_time_grid(0.0, t_end, dt)
    nt = len(t)

    C0 = np.interp(x, x_data, C_data)

    C = advect_1d_backward(
        C0,
        U,
        dx,
        dt,
        nt,
        inlet_func=None,
        decay_rate=0.0,
    )

    results_dir = os.path.join(SRC_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    np.save(os.path.join(results_dir, "test_case_2_C.npy"), C)
    np.save(os.path.join(results_dir, "test_case_2_x.npy"), x)
    np.save(os.path.join(results_dir, "test_case_2_t.npy"), t)
    print("Saved numerical arrays to", results_dir)

    plot_path = os.path.join(results_dir, "test_case_2.png")

    plot_space_time_snapshots(
        x,
        t,
        C,
        snapshots=None,
        title="Test Case 2 – Advection of CSV-Based Initial Condition",
        savepath=plot_path,
        show=True,  # show graph as well
    )

    print(f"Saved plot to {plot_path}")
    print("Test Case 2 complete.")


if __name__ == "__main__":
    run_test_case_2()

