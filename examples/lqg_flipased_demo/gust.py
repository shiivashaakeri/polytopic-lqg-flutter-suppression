"""
Atmospheric gust disturbance for the FliPASED LQG demo.

Implements the FAR 25.341 1-cosine gust profile as a time-varying state
perturbation on the short-period mode, since the FliPASED public plant
data exposes only the control input and not a separate gust channel.

Public API:
    one_minus_cos_profile(t, U_ds, H, V) -> ndarray
    GustForcing(B_w, profile_func)
    short_period_B_w(channel, n_full) -> ndarray
    simulate_with_gust(synth, V, gust, T_final, ...) -> SimResult
    simulate_open_loop_with_gust(synth, V, gust, T_final, ...) -> SimResult

Author: Shiva Shakeri.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np

from examples.lqg_flipased_demo.dynamics import alpha_at
from examples.lqg_flipased_demo.pipeline import SimResult, _blend_poly, _blend_tp


# ----------------------------------------------------------------------
# 1-cosine gust profile
# ----------------------------------------------------------------------


def one_minus_cos_profile(t: Union[float, np.ndarray],
                            U_ds: float, H: float, V: float,
                            t_start: float = 0.0) -> np.ndarray:
    """FAR 25.341 1-cosine gust velocity profile."""
    t_arr = np.atleast_1d(t).astype(float)
    tau = t_arr - t_start
    duration = 2.0 * H / V
    w_g = np.zeros_like(t_arr)
    inside = (tau >= 0.0) & (tau <= duration)
    w_g[inside] = 0.5 * U_ds * (1.0 - np.cos(np.pi * V * tau[inside] / H))
    return float(w_g[0]) if np.isscalar(t) else w_g


# ----------------------------------------------------------------------
# disturbance coupling matrix
# ----------------------------------------------------------------------


def short_period_B_w(channel: str, n_full: int) -> np.ndarray:
    """Column vector with unit entries on the short-period mode states."""
    B_w = np.zeros((n_full, 1))
    if channel == "sym":
        B_w[2, 0] = 1.0
        B_w[3, 0] = 1.0
    elif channel == "asym":
        B_w[2, 0] = 1.0
    else:
        raise ValueError(f"channel must be 'sym' or 'asym', got {channel!r}")
    return B_w


# ----------------------------------------------------------------------
# GustForcing
# ----------------------------------------------------------------------


@dataclass
class GustForcing:
    B_w: np.ndarray
    profile: Callable[[float], float]
    label: str = ""


def make_one_minus_cos_gust(
    channel: str,
    n_full: int,
    U_ds: float,
    H: float,
    V: float,
    t_start: float = 0.0,
    B_w: Optional[np.ndarray] = None,
    label: Optional[str] = None,
) -> GustForcing:
    if B_w is None:
        B_w = short_period_B_w(channel, n_full)
    if label is None:
        label = f"H={H:.0f} m, $U_{{ds}}$={U_ds:.1f} m/s, V={V:.0f} m/s"
    profile = lambda t, _U=U_ds, _H=H, _V=V, _ts=t_start: \
        one_minus_cos_profile(t, _U, _H, _V, t_start=_ts)
    return GustForcing(B_w=B_w, profile=profile, label=label)


# ----------------------------------------------------------------------
# Closed-loop simulation with gust
# ----------------------------------------------------------------------


def simulate_with_gust(
    synth,
    V: Union[float, Callable[[float], float]],
    gust: GustForcing,
    T_final: float = 3.0,
    x0_full: Optional[np.ndarray] = None,
    xhat0_red: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> SimResult:
    """Sample-and-hold closed-loop sim with gust disturbance forcing."""
    poly_full_d = synth.poly_full_d
    poly_red_d  = synth.poly_red_d
    tp          = synth.tp
    domain      = tp.domain if hasattr(tp, "domain") else np.array([[40.0, 61.0]])

    n_full = poly_full_d.n
    n_red  = poly_red_d.n
    m      = poly_red_d.m
    p      = poly_red_d.p
    V_grid = poly_red_d.V_grid
    dt     = synth.dt

    if x0_full is None:
        x0_full = np.zeros(n_full)
    if xhat0_red is None:
        xhat0_red = np.zeros(n_red)

    V_func = V if callable(V) else (lambda t, _v=float(V): _v)

    n_steps = int(np.round(T_final / dt))
    t_grid  = np.arange(n_steps + 1) * dt

    x_full   = np.zeros((n_full, n_steps + 1))
    xhat_red = np.zeros((n_red,  n_steps + 1))
    u_log    = np.zeros((m, n_steps + 1))
    y_log    = np.zeros((p, n_steps + 1))
    V_log    = np.zeros(n_steps + 1)
    w_log    = np.zeros(n_steps + 1)

    x_full[:, 0]   = x0_full
    xhat_red[:, 0] = xhat0_red

    B_w = gust.B_w
    if B_w.ndim == 1:
        B_w = B_w.reshape(-1, 1)

    for k in range(n_steps):
        tk  = t_grid[k]
        Vk  = V_func(tk)
        V_log[k] = Vk
        wg_k = float(gust.profile(tk))
        w_log[k] = wg_k

        # plant matrices: blend via original polytope grid
        C_full_k = _blend_poly(poly_full_d.C, Vk, V_grid)
        A_full_k = _blend_poly(poly_full_d.A, Vk, V_grid)
        B_full_k = _blend_poly(poly_full_d.B, Vk, V_grid)
        A_red_k  = _blend_poly(poly_red_d.A,  Vk, V_grid)
        B_red_k  = _blend_poly(poly_red_d.B,  Vk, V_grid)
        C_red_k  = _blend_poly(poly_red_d.C,  Vk, V_grid)

        # gains: blend via TP weighting functions
        K_k = _blend_tp(synth.K, Vk, tp, domain)
        L_k = _blend_tp(synth.L, Vk, tp, domain)

        y_k = C_full_k @ x_full[:, k]
        y_log[:, k] = y_k

        u_k = -K_k @ xhat_red[:, k]
        u_log[:, k] = u_k

        # observer (unaware of gust)
        xhat_pred = A_red_k @ xhat_red[:, k] + B_red_k @ u_k
        xhat_red[:, k + 1] = xhat_pred + L_k @ (y_k - C_red_k @ xhat_pred)

        # truth plant + gust forcing
        x_full[:, k + 1] = (A_full_k @ x_full[:, k]
                            + B_full_k @ u_k
                            + B_w[:, 0] * wg_k * dt)

    # final tick
    V_log[-1] = V_func(t_grid[-1])
    w_log[-1] = float(gust.profile(t_grid[-1]))
    C_full_f = _blend_poly(poly_full_d.C, V_log[-1], V_grid)
    y_log[:, -1] = C_full_f @ x_full[:, -1]
    K_f = _blend_tp(synth.K, V_log[-1], tp, domain)
    u_log[:, -1] = -K_f @ xhat_red[:, -1]

    flutter_idx = (8, 9) if synth.channel == "sym" else (5, 6)
    flutter_amp = np.linalg.norm(x_full[list(flutter_idx), :], axis=0)

    metrics = {
        "max_|u|":               float(np.max(np.abs(u_log))),
        "max_flutter_amplitude": float(flutter_amp.max()),
        "final_flutter_amplitude": float(flutter_amp[-1]),
        "max_|w_g|":             float(np.max(np.abs(w_log))),
        "max_|x_full|":          float(np.max(np.abs(x_full))),
        "final_|x_full|":        float(np.linalg.norm(x_full[:, -1])),
    }
    if verbose:
        print(f"  gust sim: V={V_log[0]:.1f} m/s, "
              f"max|w_g|={metrics['max_|w_g|']:.2f} m/s")
        print(f"    max flutter = {metrics['max_flutter_amplitude']:.4e}")
        print(f"    max |u|     = {metrics['max_|u|']:.4e} "
              f"({metrics['max_|u|']*1e3:.2f} mrad)")

    sim = SimResult(t=t_grid, x_full=x_full, xhat_red=xhat_red,
                    u=u_log, y=y_log, V_traj=V_log, metrics=metrics)
    sim.w_g = w_log
    return sim


# ----------------------------------------------------------------------
# Open-loop simulation with gust
# ----------------------------------------------------------------------


def simulate_open_loop_with_gust(
    synth,
    V: Union[float, Callable[[float], float]],
    gust: GustForcing,
    T_final: float = 3.0,
    x0_full: Optional[np.ndarray] = None,
) -> SimResult:
    """Open-loop (u=0) counterpart for comparison plots."""
    poly_full_d = synth.poly_full_d
    n_full = poly_full_d.n
    n_red  = synth.poly_red_d.n
    p      = synth.poly_red_d.p
    V_grid = synth.poly_red_d.V_grid
    dt     = synth.dt

    if x0_full is None:
        x0_full = np.zeros(n_full)

    V_func = V if callable(V) else (lambda t, _v=float(V): _v)

    n_steps = int(np.round(T_final / dt))
    t = np.arange(n_steps + 1) * dt

    x_full = np.zeros((n_full, n_steps + 1))
    u      = np.zeros((1, n_steps + 1))
    y      = np.zeros((p, n_steps + 1))
    V_log  = np.zeros(n_steps + 1)
    w_log  = np.zeros(n_steps + 1)

    x_full[:, 0] = x0_full
    B_w = gust.B_w
    if B_w.ndim == 1:
        B_w = B_w.reshape(-1, 1)

    for k in range(n_steps):
        Vk   = V_func(t[k])
        wg_k = float(gust.profile(t[k]))
        V_log[k] = Vk
        w_log[k] = wg_k

        Af = _blend_poly(poly_full_d.A, Vk, V_grid)
        Cf = _blend_poly(poly_full_d.C, Vk, V_grid)
        y[:, k] = Cf @ x_full[:, k]
        x_full[:, k + 1] = Af @ x_full[:, k] + B_w[:, 0] * wg_k * dt

        if not np.all(np.isfinite(x_full[:, k + 1])) or \
           np.linalg.norm(x_full[:, k + 1]) > 1e15:
            x_full[:, k + 1:] = np.nan
            V_log[k + 1:] = Vk
            w_log[k + 1:] = wg_k
            break

    V_log[-1] = V_func(t[-1])
    w_log[-1] = float(gust.profile(t[-1]))
    Cf = _blend_poly(poly_full_d.C, V_log[-1], V_grid)
    if np.all(np.isfinite(x_full[:, -1])):
        y[:, -1] = Cf @ x_full[:, -1]

    sim = SimResult(t=t, x_full=x_full, xhat_red=np.zeros((n_red, n_steps + 1)),
                    u=u, y=y, V_traj=V_log, metrics={})
    sim.w_g = w_log
    return sim


# ----------------------------------------------------------------------
# self-test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("gust.py self-test")
    print("=" * 70)

    from examples.lqg_flipased_demo.pipeline import synthesize

    synth = synthesize(channel="sym", hull="cno", verbose=True)
    gust  = make_one_minus_cos_gust(
        channel="sym", n_full=synth.poly_full_d.n,
        U_ds=9.0, H=30.0, V=55.0, t_start=0.5)
    print(f"\ngust: {gust.label}")
    sim = simulate_with_gust(synth, V=55.0, gust=gust, T_final=3.0, verbose=True)
    print(f"t shape: {sim.t.shape}")