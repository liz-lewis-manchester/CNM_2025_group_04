import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def animate_advection(x, t, C, title="Advection Animation", interval=150):
    fig, ax = plt.subplots(figsize=(8, 4))
    line, = ax.plot([], [], lw=2)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(C.min(), C.max())
    ax.set_xlabel("x (m)")
    ax.set_ylabel("Concentration (µg/m³)")
    ax.set_title(title)
    
    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        line.set_data(x, C[frame, :])
        ax.set_title(f"{title}   (t = {t[frame]:.1f} s)")
        return line,

    anim = FuncAnimation(fig, update, frames=len(t), init_func=init,
                         blit=True, interval=interval)

    plt.show()
    return anim
    
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
