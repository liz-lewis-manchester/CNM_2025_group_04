import numpy as np
import sys
import os

THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.dirname(THIS_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from model import create_space_grid, create_time_grid, advect_1d_backward
from plots import plot_space_time_snapshots, animate_advection
import matplotlib.pyplot as plt


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


    for U in U_values:
        for dx in dx_values:
            for dt in dt_values:

                x = create_space_grid(0.0, L, dx)
                t = create_time_grid(0.0, t_end, dt)

                nx = x.size
                nt = t.size

                C0 = np.zeros(nx)
                C0[0] = 250.0

                C = advect_1d_backward(
                    C0,
                    U=U,
                    dx=dx,
                    dt=dt,
                    nt=nt,
                    inlet_func=None,
                )

                max_C = C.max()
                peak_idx = np.argmax(C[-1, :])
                peak_x = x[peak_idx]

                print(
                    f"U={U}, dx={dx}, dt={dt} -> maxC={max_C:.2f}, peak_x={peak_x:.2f}"
                )

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

    example_case = results[0]
    animate_advection(
        example_case["x"],
        example_case["t"],
        example_case["C"],
        title="Test Case 3 – Animation Example",
    )

    return results


if __name__ == "__main__":
    results = run_test_case_3()
