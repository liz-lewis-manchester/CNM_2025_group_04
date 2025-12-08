import matplotlib.pyplot as plt
import numpy as np


def plot_space_time_snapshots(
    x: np.ndarray,
    t: np.ndarray,
    C: np.ndarray,
    snapshots=None,
    title: str = "",
    savepath: str | None = None,
    show: bool = False,
):

    nt, nx = C.shape

    if snapshots is None:
        snapshots = [0, nt // 3, 2 * nt // 3, nt - 1]

    plt.figure(figsize=(10, 6))

    for idx in snapshots:
        if 0 <= idx < nt:
            plt.plot(x, C[idx, :], label=f"t = {t[idx]:.0f} s")

    plt.xlabel("x (m)")
    plt.ylabel("Concentration C (µg/m³)")
    if title:
        plt.title(title)
    plt.legend()
    plt.grid(True)

    if savepath is not None:
        plt.savefig(savepath, dpi=200)

    if show:
        plt.show()
    else:
        plt.close()
