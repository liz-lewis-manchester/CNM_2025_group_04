import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import IPython.display as ipd

THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.dirname(THIS_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from model import create_space_grid, create_time_grid, advect_1d_backward
from plots import plot_space_time_snapshots, animate_advection


def run_test_case_3(
    U_values=None,
    dx_values=None,
    dt_values=None,
    L: float = 20.0,
    t_end: float = 300.0,
):
    """Run sensitivity analysis for advection parameters (U, dx, dt)."""
    
    # Set default parameter values
    U_values = U_values or [0.05, 0.1, 0.2]
    dx_values = dx_values or [0.1, 0.2, 0.5]
    dt_values = dt_values or [1.0, 5.0, 10.0]
    
    results = []
    total_simulations = len(U_values) * len(dx_values) * len(dt_values)
    
    print(f"Running {total_simulations} simulations...")
    
    for U in U_values:
        for dx in dx_values:
            for dt in dt_values:
                # Create grid
                x = create_space_grid(0.0, L, dx)
                t = create_time_grid(0.0, t_end, dt)
                
                # Initial condition: point source at x=0
                C0 = np.zeros(len(x))
                C0[0] = 250.0
                
                # Run simulation
                C = advect_1d_backward(
                    C0, U=U, dx=dx, dt=dt, nt=len(t), inlet_func=None
                )
                
                # Calculate results
                max_C = C.max()
                peak_idx = np.argmax(C[-1, :])
                peak_x = x[peak_idx]
                
                print(f"U={U}, dx={dx}, dt={dt} → maxC={max_C:.2f}, peak_x={peak_x:.2f}")
                
                results.append({
                    "U": U, "dx": dx, "dt": dt,
                    "x": x, "t": t, "C": C,
                    "max_C": max_C, "peak_x": peak_x,
                })
    
    # Create sensitivity plots
    create_sensitivity_plots(results)
    
    # Generate animation
    if results:
        example = results[0]
        anim = animate_advection(
            example["x"], example["t"], example["C"],
            title="Test Case 3 – Animation Example"
        )
        ipd.display(ipd.HTML(anim.to_jshtml()))
    
    return results


def create_sensitivity_plots(results):
    """Create three sensitivity plots for dx, dt, and U."""
    
    # 1. Sensitivity to dx
    fig, ax = plt.subplots(figsize=(8, 4))
    target_U, target_dt = 0.1, 5.0
    
    for entry in results:
        if entry["U"] == target_U and entry["dt"] == target_dt:
            ax.plot(entry["x"], entry["C"][-1, :], 
                   label=f"dx={entry['dx']}")
    
    ax.set(xlabel="x (m)", ylabel="C (µg/m³)",
           title=f"Sensitivity to dx (U={target_U} m/s, dt={target_dt} s)")
    ax.legend()
    plt.show()
    
    # 2. Sensitivity to dt
    fig, ax = plt.subplots(figsize=(8, 4))
    target_U, target_dx = 0.1, 0.1
    
    for entry in results:
        if entry["U"] == target_U and abs(entry["dx"] - target_dx) < 1e-12:
            ax.plot(entry["x"], entry["C"][-1, :],
                   label=f"dt={entry['dt']}")
    
    ax.set(xlabel="x (m)", ylabel="C (µg/m³)",
           title=f"Sensitivity to dt (U={target_U} m/s, dx={target_dx} m)")
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    # 3. Sensitivity to U
    fig, ax = plt.subplots(figsize=(8, 4))
    target_dx, target_dt = 0.1, 5.0
    
    for entry in results:
        if abs(entry["dx"] - target_dx) < 1e-12 and entry["dt"] == target_dt:
            ax.plot(entry["x"], entry["C"][-1, :],
                   label=f"U={entry['U']}")
    
    ax.set(xlabel="x (m)", ylabel="C (µg/m³)",
           title=f"Sensitivity to U (dx={target_dx} m, dt={target_dt} s)")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_test_case_3()
