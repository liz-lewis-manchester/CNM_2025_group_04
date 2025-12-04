import numpy as np
from src.model import run_base_case
import os

def run_test_case_1():
    """
    Test Case 1: (explanation)
    - Uses the base case defined in model.py (backward advection, U = 0.1 m/s)
    - Runs simulation for t = 0–300 s
    - Saves results into results/test_case_1.npy
    """

    x, t, C = run_base_case()

    # By this we ensure results folder exists
    results_dir = "src/results"
    os.makedirs(results_dir, exist_ok=True)

    # Here we save the output
    np.save(os.path.join(results_dir, "test_case_1.npy"), C)

    print("Test Case 1 complete.")
    print(f"Grid size: nx={len(x)}, nt={len(t)}")
    print(f"Saved results to {results_dir}/test_case_1.npy")

if __name__ == "__main__":
    run_test_case_1()

