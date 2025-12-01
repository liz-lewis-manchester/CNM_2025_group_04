import matplotlib.pyplot as plt
import numpy as np

def plot_space_time(C, x, t, savepath=None):
    """
    Plots concentration vs distance for several time steps.

    C : 2D array of shape (nt, nx)
    x : 1D array of positions
    t : 1D array of times
    """
    plt.figure()
    for i, ti in enumerate(t):
        plt.plot(x, C[i, :], label=f"t={ti:.0f}s")
    plt.xlabel("Distance along river (m)")
    plt.ylabel("Concentration (µg/m³)")
    plt.legend()
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
    plt.close()

