# src/dynamical_system/vdp/plotting.py
import matplotlib.pyplot as plt


def plot_phase_portrait(u, du, *, ax=None, title=None, xlabel="u", ylabel="u_dot", lw=0.8):
    if ax is None:
        _, ax = plt.subplots()

    ax.plot(u, du, linewidth=lw)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax


def plot_xy(x, y, *, ax=None, title=None, lw=0.8):
    if ax is None:
        _, ax = plt.subplots()

    ax.plot(x, y, linewidth=lw)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax


def plot_time_series(t, s, *, ax=None, title=None, xlabel="t", ylabel="signal", lw=0.8):
    """
    Plot a single time series s(t).
    """
    if ax is None:
        _, ax = plt.subplots()

    ax.plot(t, s, linewidth=lw)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax
