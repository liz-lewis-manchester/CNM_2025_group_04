import os
import sys
import numpy as np
import pandas as pd
import IPython.display as ipd

# this is the path setup so this script can import modules from src/
# also works when running this files directly 
THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.dirname(THIS_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# here we import solver utilities and plotting functions
from model import create_space_grid, create_time_grid, advect_1d_backward
from plots import plot_space_time_snapshots, animate_advection


def run_test_case_2():
    """
    this test reads the initial concentration profile from a CSV file 
    interpolates the CSV data onto the computational grid 
    runs the backward 1D advetion solver 
    saves a snapshot plot and displays an animation

    and the purpose is to validate that the model can ingest real initial condition data 
    rather than hand defined initial condition.
    """
    # here is the numerical parameters (grid + time stepping)
    dx = 0.2
    dt = 10.0
    t_end = 300.0
    U = 0.1 # constant advection velocity 

    # here we load initial condition data from CSV 
    # assumption: first column = x positions, second column = concentration C
    data_path = os.path.join(SRC_DIR, "data", "initial_conditions.csv")
    df = pd.read_csv(data_path, encoding="latin1")

    x_data = df.iloc[:, 0].to_numpy()
    C_data = df.iloc[:, 1].to_numpy()

    # creates computational grids 
    # use the data range [min(x_data), max(x_data)] as the model domain
    x_min, x_max = x_data.min(), x_data.max()
    x = create_space_grid(x_min, x_max, dx)
    t = create_time_grid(0.0, t_end, dt)
    nt = len(t)

    # interpolates the tabulated condition onto the model grid 
    # linear interpolation
    C0 = np.interp(x, x_data, C_data)

    # runs the backward advection solver 
    # inlet_func=None -> inlet boundary held constant at initial inlet value
    # decay_rate=0 -> no decay in this test case 
    C = advect_1d_backward(
        C0,
        U,
        dx,
        dt,
        nt,
        inlet_func=None,
        decay_rate=0.0,
    )

    #here we save the results to /results and generate the plots
    results_dir = os.path.join(SRC_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    plot_path = os.path.join(results_dir, "test_case_2.png")

    # here we plot concentration snapshots at selectd times (chosen internally if snapshots=None)
    plot_space_time_snapshots(
        x,
        t,
        C,
        snapshots=None,
        title="Test Case 2 – Advection of CSV-Based Initial Condition",
        savepath=plot_path,
        show=True,  # show graph as well during execution 
    )

    print(f"Saved plot to {plot_path}")

    # here we display the animation 
    print ("Generating animation...")
    anim = animate_advection(
        x,
        t,
        C,
        title="Test Case 2 – CSV Initial Condition Animation",
        interval=150,
    )
    ipd.display(ipd.HTML(anim.to_jshtml()))

    print("Test Case 2 complete.")

# allows this test to be run as a standalone script 
if __name__ == "__main__":
    run_test_case_2()
