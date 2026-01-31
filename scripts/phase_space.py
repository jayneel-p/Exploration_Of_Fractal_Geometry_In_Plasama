#scripts/phase_space.py
import matplotlib.pyplot as plt
import numpy as np

from dynamical_system.vdp.simulate import simulate_vdp_pair
from dynamical_system.vdp.plotting import plot_phase_portrait, plot_xy, plot_time_series


def main():
    # ----------------------------
    # Parameters (change these)
    # ----------------------------
    a1, a2 = 1.0, 1.0
    a, b = 2.5, -0.5
    dt = 0.01
    t_max = 200.0
    t_transient = 50.0
    z0 = np.array([1.2, 0.0, 1.1, 0.0])

    # Plot controls
    stride = 2            # downsample for plotting; set to 1 for no downsampling
    t_plot_max = None     # set to a float to only plot early portion (e.g. 50.0)

    # ----------------------------
    # Simulate
    # ----------------------------
    t, x, vx, y, vy = simulate_vdp_pair(
        a1=a1,
        a2=a2,
        a=a,
        b=b,
        dt=dt,
        t_max=t_max,
        t_transient=t_transient,
        z0=z0,
    )

    # ----------------------------
    # Optional trimming + downsampling for cleaner plots
    # ----------------------------
    if t_plot_max is not None:
        mask = t <= t_plot_max
        t, x, vx, y, vy = t[mask], x[mask], vx[mask], y[mask], vy[mask]

    if stride is not None and stride > 1:
        t, x, vx, y, vy = t[::stride], x[::stride], vx[::stride], y[::stride], vy[::stride]

    # ----------------------------
    # Plot: time series + phase portraits
    # ----------------------------
    fig, axs = plt.subplots(2, 3, figsize=(13, 7))

    # Time series
    plot_time_series(t, x, ax=axs[0, 0], title="Time series: x(t)", xlabel="t", ylabel="x")
    plot_time_series(t, y, ax=axs[0, 1], title="Time series: y(t)", xlabel="t", ylabel="y")

    # Overlay for quick visual comparison
    axs[0, 2].plot(t, x, linewidth=0.8, label="x")
    axs[0, 2].plot(t, y, linewidth=0.8, label="y")
    axs[0, 2].set_title("Time series overlay")
    axs[0, 2].set_xlabel("t")
    axs[0, 2].set_ylabel("signal")
    axs[0, 2].grid(True, alpha=0.3)
    axs[0, 2].legend()

    # Phase portraits
    plot_phase_portrait(x, vx, ax=axs[1, 0], title="Phase portrait: (x, ẋ)", xlabel="x", ylabel="ẋ")
    plot_phase_portrait(y, vy, ax=axs[1, 1], title="Phase portrait: (y, ẏ)", xlabel="y", ylabel="ẏ")
    plot_xy(x, y, ax=axs[1, 2], title="Coupling portrait: (x, y)")

    fig.suptitle(f"Coupled VdP | a={a}, b={b}, a1={a1}, a2={a2}, dt={dt}", y=1.02)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
