import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import IPython.display as ipd

THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.dirname(THIS_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from model import create_space_grid, create_time_grid, advect_1d_backward
from plots import plot_space_time_snapshots, animate_advection


def run_test_case_4(
    decay_rates=None,
    C_in0: float = 250.0,
    L: float = 20.0,
    dx: float = 0.2,
    t_end: float = 300.0,
    dt: float = 10.0,
    U: float = 0.1,
    example_lambda: float | None = None,
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

        max_C = C.max()
        print(f"λ={lam}, U={U}, dx={dx}, dt={dt}, nx={nx}, nt={nt}, maxC={max_C:.2f}")

    fig, ax = plt.subplots(figsize=(8, 4))

    for lam, res in results.items():
        x = res["x"]
        C = res["C"]
        ax.plot(x, C[-1, :], label=f"λ = {lam}")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("C (µg/m³)")
    ax.set_title("Effect of decay rate λ on final concentration profile")
    ax.legend()
    plt.tight_layout()
    plt.show()

    
    if example_lambda is not None and example_lambda in results:
        res = results[example_lambda]
        plot_space_time_snapshots(
            res["x"],
            res["t"],
            res["C"],
            snapshots=None,
            title=f"Test Case 4 – Space–time plot for λ = {example_lambda}",
            savepath=None,
            show=True,
        )

        anim = animate_advection(
            res["x"],
            res["t"],
            res["C"],
            title=f"Test Case 4 – Animation for λ = {example_lambda}",
            interval=150,
        )
        ipd.display(ipd.HTML(anim.to_jshtml()))

    print("Test Case 4 complete.")

    return None


if __name__ == "__main__":
    results = run_test_case_4()

