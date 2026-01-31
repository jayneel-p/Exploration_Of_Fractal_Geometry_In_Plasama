'''
src/dynamical_system/vdp/simulate.py
*Generates coupled-VDP data using RK4 integrator

*Some notes:
    I believe a higher step integrator will be need for chaotic motion. The best bet is to
generate curves with known parameters displaying chaotic motion and compare to literature.
Or I could find a known implementation and see if they use a higher step integrator.
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
    Right-hand side of the coupled Van der Pol (Rayleigh-type) oscillator system.

    State vector
    ------------
    z = [x, xdot, y, ydot]

    where xdot = dx/dt and ydot = dy/dt.

    Equations
    ---------
        x'    = xdot
        xdot' = [a1 - (x + b y)^2] * xdot - (x + b y)

        y'    = ydot
        ydot' = [a2 - (y + a x)^2] * ydot - (y + a x)

    Parameters
    ----------
    :param t: Time (included for compatibility with ODE integrators; not used explicitly).
    :param z: State vector [x, xdot, y, ydot].
    :param a1: Rayleigh parameter for the x oscillator.
    :param a2: Rayleigh parameter for the y oscillator.
    :param a: Coupling strength from x → y.
    :param b: Coupling strength from y → x.

    Notes
    -----
    A common heuristic interpretation of the coupling is:
        |a| > |b|  ⇒ x tends to drive y,
        |a| < |b|  ⇒ y tends to drive x,
        |a| = |b|  ⇒ symmetric coupling.
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
    Perform a single fixed-step 4th-order Runge–Kutta update.

    :param f: callable defining the system dynamics, f(t, z, *f_args) → dz/dt.
    :param t: current time.
    :param z: current state vector.
    :param dt: time step.
    :param f_args: additional parameters passed to f.

    :return: State vector at time t + dt.
    """
    k1 = f(t,           z,              *f_args)
    k2 = f(t + 0.5*dt,  z + 0.5*dt*k1,  *f_args)
    k3 = f(t + 0.5*dt,  z + 0.5*dt*k2,  *f_args)
    k4 = f(t + dt,      z + dt*k3,      *f_args)

    return z + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)


# ---------------------------------------------------------------------------
# Simulation driver
# ---------------------------------------------------------------------------
def simulate_vdp_pair(
    a1: float,
    a2: float,
    a: float,
    b: float,
    dt: float,
    t_max: float,
    t_transient: float,
    z0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Integrate a pair of coupled Van der Pol oscillators using fixed-step RK4.

    :param a1: parameter for oscillator x
    :param a2: parameter for oscillator y
    :param a: coupling strength from x to y
    :param b: coupling strength from y to x
    :param dt: time step
    :param t_max: maximum time step
    :param t_transient: end time for transient, subtract t_transient from t_max
    :param z0: initial conditions.
        - If None, default initial conditions are used.
    :return:
        t_rec  – Recorded time array starting at 0 after transient removal.
        x_rec  – x(t) time series after transient removal.
        vx_rec – ẋ(t) time series after transient removal.
        y_rec  – y(t) time series after transient removal.
        vy_rec – ẏ(t) time series after transient removal.
    '''
    if dt <= 0.0:


        raise ValueError("dt must be positive")
    if t_max <= 0.0:
        raise ValueError("t_max must be positive")
    if t_transient < 0.0:
        raise ValueError("t_transient must be non-negative")
    if t_transient >= t_max:
        raise ValueError("t_transient must be less than t_max")

    # Number of stored time points (include both endpoints)
    n_steps = int(round(t_max / dt)) + 1

    t_all  = np.empty(n_steps, dtype=float)
    x_all  = np.empty(n_steps, dtype=float)
    vx_all = np.empty(n_steps, dtype=float)
    y_all  = np.empty(n_steps, dtype=float)
    vy_all = np.empty(n_steps, dtype=float)

    # Initial condition
    t = 0.0
    if z0 is None:
        z = np.array([1.2, 0.0, 1.1, 0.0], dtype=float)
    else:
        z = np.asarray(z0, dtype=float)
    for n in range(n_steps):
        # Store current state
        t_all[n]  = t
        x_all[n]  = z[0]
        vx_all[n] = z[1]
        y_all[n]  = z[2]
        vy_all[n] = z[3]

        # Advance to next time, except after the last stored point
        if n < n_steps - 1:
            z = rk4_step(vdp_coupled, t, z, dt, a1, a2, a, b)
            t += dt

    # Discard transient: keep t >= t_transient
    keep = t_all >= t_transient
    if not np.any(keep):
        raise RuntimeError(
            "Transient cut removed all data; decrease t_transient or increase t_max"
        )

    t_rec  = t_all[keep] - t_all[keep][0]
    x_rec  = x_all[keep]
    vx_rec = vx_all[keep]
    y_rec  = y_all[keep]
    vy_rec = vy_all[keep]

    return t_rec, x_rec, vx_rec, y_rec, vy_rec





