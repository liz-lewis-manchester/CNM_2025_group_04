import os
import sys
import numpy as np

THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.dirname(THIS_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from model import run_base_case
from plots import plot_space_time_snapshots


def run_test_case_1():
    """
    Test Case 1: (explanation)
    - Uses the base case defined in model.py (run_base_case)
    - Plots several time snapshots
    - Saves plot to src/results/test_case_1.png
    """

    x, t, C = run_base_case() 

    results_dir = os.path.join(SRC_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    plot_path = os.path.join(results_dir, "test_case_1.png")

    plot_space_time_snapshots(
        x,
        t,
        C,
        snapshots=None,  # default: [0, nt/3, 2nt/3, nt-1]
        title="Test Case 1 – Advection of Point Source at x=0",
        savepath=plot_path,
        show=True,  # show graph as well
    )

    print("Test Case 1 complete.")
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    run_test_case_1()


