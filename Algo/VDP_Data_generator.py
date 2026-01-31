'''
Generates coupled-VDP data. Saves generated VDP data as an array.
'''
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Dynamical system: coupled Van der Pol/Rayleigh oscillators
# ---------------------------------------------------------------------------

def vdp_coupled(
    t: float,
    z: np.ndarray,
    a1: float,
    a2: float,
    a: float,  # coupling x -> y
    b: float,  # coupling y -> x
) -> np.ndarray:
    """
    Right-hand side for the coupled coupled Van der Pol system.

    State:
        z = [x, vx, y, vy],
        where vx = dx/dt, vy = dy/dt.

    Equations:
        x'  = vx
        vx' = [a1 - (x + b*y)^2] * vx  - (x + b*y)

        y'  = vy
        vy' = [a2 - (y + a*x)^2] * vy  - (y + a*x)

    Parameters:
        a1, a2 : Rayleigh parameters of each oscillator (usually 1.0).
        a      : coupling strength from x → y.
        b      : coupling strength from y → x.

    Directionality:
        |a| > |b|  ⇒ x drives y
        |a| < |b|  ⇒ y drives x
        |a| = |b|  ⇒ symmetric coupling
    """
    x, vx, y, vy = z

    dx = vx
    dvx = (a1 - (x + b * y) ** 2) * vx - (x + b * y)

    dy = vy
    dvy = (a2 - (y + a * x) ** 2) * vy - (y + a * x)

    return np.array([dx, dvx, dy, dvy], dtype=float)


# ---------------------------------------------------------------------------
# Generic RK4 stepper
# ---------------------------------------------------------------------------

#Remark: maybe best to use higher dim- RK for chaotic (sensitive) systems.
#           I have seen 15 step algos used for three-body problems.

def rk4_step(
    f,
    t: float,
    z: np.ndarray,
    dt: float,
    *f_args,
) -> np.ndarray:
    """
    One explicit 4th-order Runge–Kutta step.

    Inputs
    ------
    f      : callable
             Right-hand side f(t, z, *f_args) → dz/dt (same shape as z).
    t      : float
             Current time.
    z      : np.ndarray
             Current state vector z(t).
    dt     : float
             Time step (assumed constant).
    f_args : tuple
             Extra parameters passed through to f.

    Returns
    -------
    z_next : np.ndarray
             Approximation to z(t + dt).
    """
    k1 = f(t,           z,               *f_args)
    k2 = f(t + 0.5*dt,  z + 0.5*dt*k1,   *f_args)
    k3 = f(t + 0.5*dt,  z + 0.5*dt*k2,   *f_args)
    k4 = f(t + dt,      z + dt*k3,       *f_args)

    return z + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)


# ---------------------------------------------------------------------------
# Simulation driver
# ---------------------------------------------------------------------------

def simulate_vdp_pair(
    a1: float,
    a2: float ,
    a: float ,
    b: float ,
    dt: float ,
    t_max: float ,
    t_transient: float ,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Integrate the coupled Van der Pol system using fixed-step RK4.

    Strategy
    --------
    - Integrate from t = 0 to t = t_max with constant step dt.
    - Store x(t), y(t) at every step.
    - After the integration, discard all samples with t < t_transient. (for chaotic systems)
      The recorded time axis is then shifted so that the first kept
      sample has t_rec = 0.

    Parameters
    ----------
    mu          : float
                  Rayleigh parameter for both oscillators (a1 = a2 = mu).
    a, b        : float
                  Coupling strengths (x → y and y → x).
    dt          : float
                  Time step.
    t_max       : float
                  Final time of the integration.
    t_transient : float
                  Length of transient to discard (in the same units as t).

    Returns
    -------
    t_rec : np.ndarray, shape (N_rec,)
        Recorded times, starting at 0.
    x_rec : np.ndarray, shape (N_rec,)
        Recorded x(t) samples.
    y_rec : np.ndarray, shape (N_rec,)
        Recorded y(t) samples.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if t_max <= 0.0:
        raise ValueError("t_max must be positive")
    if t_transient < 0.0:
        raise ValueError("t_transient must be non-negative")
    if t_transient >= t_max:
        raise ValueError("t_transient must be less than t_max")

    # Number of stored time points (include both endpoints)
    n_steps = int(np.floor(t_max / dt)) + 1

    t_all = np.empty(n_steps, dtype=float)
    x_all = np.empty(n_steps, dtype=float)
    y_all = np.empty(n_steps, dtype=float)

    # Initial condition
    t = 0.0
    z = np.array([1.2, 0.0, 1.1, 0.0])

    for n in range(n_steps):
        # Store current state
        t_all[n] = t
        x_all[n] = z[0]
        y_all[n] = z[2]

        # Advance to next time, except after the last stored point
        if n < n_steps - 1:
            z = rk4_step(vdp_coupled, t, z, dt, a1, a2, a, b)
            t += dt

    # Discard transient: keep t >= t_transient
    keep = t_all >= t_transient
    if not np.any(keep):
        raise RuntimeError("Transient cut removed all data; decrease t_transient or increase t_max")

    t_rec = t_all[keep] - t_all[keep][0]
    x_rec = x_all[keep]
    y_rec = y_all[keep]

    return t_rec, x_rec, y_rec


