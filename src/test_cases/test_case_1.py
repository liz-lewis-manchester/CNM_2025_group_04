import os
import sys
import numpy as np

THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.dirname(THIS_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from model import run_base_case
from plots import plot_space_time_snapshots, animate_advection
from IPython.display import HTML, display


def run_test_case_1():
    x, t, C = run_base_case() 

    results_dir = os.path.join(SRC_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    plot_path = os.path.join(results_dir, "test_case_1.png")
    print("Saving to:", plot_path)

    plot_space_time_snapshots(
        x,
        t,
        C,
        snapshots=None,  # default: [0, nt/3, 2nt/3, nt-1]
        title="Test Case 1 – Advection of Point Source at x=0",
        savepath=plot_path,
        show=True, 
    )

    print(f"Saved plot to {plot_path}")

    
    print ("Generating animation...")
    anim = animate_advection(
      x,
      t,
      C,
      title = "Test Case 1 - Advection Animation"
    )
    display(HTML(anim.to_jshtml()))
    
    print("Test Case 1 complete.")


if __name__ == "__main__":
    run_test_case_1()
