import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import IPython.display as ipd

# path setup so this script can import modules from src/
THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.dirname(THIS_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
# imports solver utilities and plotting/animation functions 
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
    """ 
    this test investigates how an exponentially decaying pollutant
    concentration in time alters the model results.  
    the purpose of this test is to compare C(x, t_end) for different labmda values and observe how faster decay 
    in the inlet signal reduces downstream concentration over time.
    """
    # default decay rates (used if caller does not supply values)
    if decay_rates is None:
        decay_rates = [0.0, 0.005, 0.01, 0.02]

    results = {} # stores results for each labmda value (keyed by 1am)
    # loops over decay rates and runs the solver for each case
    for lam in decay_rates:
        # creates computational grids
        x = create_space_grid(0.0, L, dx)
        t = create_time_grid(0.0, t_end, dt)

        nx = x.size
        nt = t.size

        # initial condition at t=0 (concentration profile along the domain)
        C0 = np.zeros(nx)
        C0[0] = C_in0

        # time dependant inlet boundary condition (exponential decay in time)
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

        # stores outputs for later comparison/plotting
        results[lam] = {"x": x, "t": t, "C": C}

        # simple diagnostic output 
        max_C = C.max()
        print(f"λ={lam}, U={U}, dx={dx}, dt={dt}, nx={nx}, nt={nt}, maxC={max_C:.2f}")

    # plot: compares final concentration profiles for each lambda 
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

    # this is optional which shows space-time snapshots and animation for one chosen lambda 
    if example_lambda is not None:
        if example_lambda not in results:
            x = create_space_grid(0.0, L, dx)
            t = create_space_grid(0.0, t_end, dt)

            nx = x.size
            nt = t.size

            C0 = np.zeros(nx)
            C0[0] = C_in0

            def inlet(time):
                return C_in0 * np.exp(-example_lambda * time)
            C = advect_1d_backward(
              C0,
              U=U,
              dx=dx,
              dt=dt,
              nt=nt,
              inlet_func=inlet,
            )

            results[example_lambda] = {"x": x, "t": t, "C": C}
            print(
              f"(extra run) λ={example_lambda} was not in decay_rates, "
              "so it was simulated for animation."
            )

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

        # animation for one selected case 
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

# allows this test to be run as a standalone script 
if __name__ == "__main__":
    run_test_case_4()

