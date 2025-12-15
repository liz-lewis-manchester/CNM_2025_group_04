import os
import sys
import numpy as np

# here we have the path setup so that the script can import modules from src/
# this also works when running test files directly, for example python test_case_1.py)
THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.dirname(THIS_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# here we import the "base case" scenario setup + plotting utilities 
from model import run_base_case
from plots import plot_space_time_snapshots, animate_advection
from IPython.display import HTML, display


def run_test_case_1():
    """ 
    here we run a simple base advection scenario deifend in model.run_base_case()
    generate a snapshot plot (space profiles at selected times)
    and generate animation of C(x,t)

    and the purpose is the sanity check that the the solver produces sensible transport behaviour 
    for a simple initial/boundary condition
    """
    # runs the base case scenario:
    # returns x-grid, time-grid and concentration array C[time, space]
    x, t, C = run_base_case() 

    # creates result folder if it doesn't already exist 
    # and we define where to save the output plot 
    results_dir = os.path.join(SRC_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    plot_path = os.path.join(results_dir, "test_case_1.png")
    print("Saving to:", plot_path)

    # here we plot concentration snapshot at selected times 
    # snpashots=None -> plotting function chooses default snapshot indices 
    plot_space_time_snapshots(
        x,
        t,
        C,
        snapshots=None,  # default: [0, nt/3, 2nt/3, nt-1]
        title="Test Case 1 – Advection of Point Source at x=0",
        savepath=plot_path,
        show=True,  # displays the figure when running interactively 
    )

    print(f"Saved plot to {plot_path}")

    # here we create and display an animation of the concentration field 
    print ("Generating animation...")
    anim = animate_advection(
      x,
      t,
      C,
      title = "Test Case 1 - Advection Animation"
    )
    display(HTML(anim.to_jshtml()))
    
    print("Test Case 1 complete.")

# this allows this test to be run as a standalone script 
if __name__ == "__main__":
    run_test_case_1()
