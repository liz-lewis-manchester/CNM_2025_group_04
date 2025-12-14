import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display
from matplotlib import rc
import numpy as np

# configuring Matplotlib to use Javascript-base animations
# that works well in Jupyter / Colab environments

rc("animation", html="jshtml")


def animate_advection(x, t, C, title="Advection Animation", interval=150):
    """ 
    this part creates the animation of the concentration profile C(x,t)
    """
    # creates figure and axis
    fig, ax = plt.subplots(figsize=(8, 4))

    # initialise an empty line object which is updated at each frame 
    line, = ax.plot([], [], lw=2)

    # fixes axis limits for consistent animation scaling 
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(C.min(), C.max())

    # axis labels and initial title 
    ax.set_xlabel("x (m)")
    ax.set_ylabel("Concentration (µg/m³)")
    ax.set_title(title)
    
    # initialisation function for the animation 
    def init():
        line.set_data([], [])
        return line,

    # updates function called for each animation frame
    def update(frame):
        # update concentration profile at time index 'frame'
        line.set_data(x, C[frame, :])
        ax.set_title(f"{title}   (t = {t[frame]:.1f} s)")
        return line,

    # creating the animation
    anim = FuncAnimation(fig, update, frames=len(t), init_func=init,
                         blit=True, interval=interval)
    
    plt.close(fig) # close the figure to avoid duplicate static plots
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
    """ 
    plots concentration profiles C(x) at selected time snapshots.
    """
    # number of time steps and spatial points
    nt, nx = C.shape

    # default snapshot indices if none are provided 
    if snapshots is None:
        snapshots = [0, nt // 3, 2 * nt // 3, nt - 1]

    plt.figure(figsize=(10, 6))

    # plot concentration profiles at selected times 
    for idx in snapshots:
        if 0 <= idx < nt:
            plt.plot(x, C[idx, :], label=f"t = {t[idx]:.0f} s")
    # axis labels and title
    plt.xlabel("x (m)")
    plt.ylabel("Concentration C (µg/m³)")
    if title:
        plt.title(title)

    # legend and grid for readability 
    plt.legend()
    plt.grid(True)

    # saves figure if requested
    if savepath is not None:
        plt.savefig(savepath, dpi=200)
    
    # shows or closes figure depending on context 
    if show:
        plt.show()
    else:
        plt.close()
