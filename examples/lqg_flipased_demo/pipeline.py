"""
FliPASED LQG synthesis and closed-loop simulation pipeline.

Two-piece API:
    synthesize(channel, ...) -> SynthResult
        Builds the reduced (modal-block) polytope, runs primal P-RHPG for
        state-feedback gain K and dual P-RHPG for observer gain L.

    simulate_flutter_suppression(synth, V, x0_full, T) -> SimResult
        Sample-and-hold closed-loop at the synthesis dt. Truth = the full
        18-state LPV plant evaluated at V(t); controller = the reduced
        14-state LQG (output-feedback). Returns time, full state, reduced
        estimate, control, output trajectories.

Note on noise covariances: the observer's V (measurement-noise covariance)
must reflect not just sensor noise but also the unmodeled-dynamics
contribution to the output. Empirically, V = 100 * I works across the
full design envelope; smaller V causes observer over-trust and divergence
because the dropped phugoid + fast-actuator modes feed through into the
IMU output, fooling a too-trusting observer.

Note on TP model transformation: synthesis now uses a proper HOSVD + convex
hull via tptrans (default hull='cno', gridsize=51). This replaces the old
identity-weight polytope which had no hull guarantee and no compression.
The hull type and gridsize are exposed as synthesize() arguments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Union

import numpy as np

from examples.lqg_flipased_demo.dynamics import (
    FlipasedPolytope,
    discretize_polytope,
    flipased_polytope,
    flipased_polytope_reduced,
    flipased_tpform,
    project_state,
    alpha_at,
)
from tptool.prhpg import ObserverPRHPGProblem, PRHPGProblem


DEFAULT_DT = 0.005
DEFAULT_HORIZON = 200
DEFAULT_HULL = "cno"
DEFAULT_GRIDSIZE = 51

DEFAULT_R_CTRL = np.array([[1.0]])

# Measurement noise covariance scale -- inflated to absorb unmodeled-dynamics
# contribution to the IMU output. See module docstring.
DEFAULT_V_MEAS_SCALE = 100.0


def _default_Q(n: int, channel: str) -> np.ndarray:
    """Heavy weight on flutter mode states.
    Reduced sym (14 states, blocks 1..7): flutter at indices 6,7
    Reduced asym (14 states, blocks 1..7): flutter-leading at indices 3,4
    """
    Q = np.eye(n)
    if channel == "sym":
        Q[6, 6] = 1e4
        Q[7, 7] = 1e4
    elif channel == "asym":
        Q[3, 3] = 1e4
        Q[4, 4] = 1e4
    return Q


def _default_W(n: int) -> np.ndarray:
    return np.eye(n) * 1.0


@dataclass
class SynthResult:
    channel: str
    dt: float
    horizon: int
    hull: str
    poly_full: FlipasedPolytope
    poly_red: FlipasedPolytope
    poly_red_d: FlipasedPolytope
    poly_full_d: FlipasedPolytope
    K: np.ndarray
    L: np.ndarray
    K_prhpg: np.ndarray
    J_K: float
    J_L: float
    state_indices: np.ndarray
    tp: object   # TPForm from tptrans, stored for PDC blending at runtime


@dataclass
class SimResult:
    t: np.ndarray
    x_full: np.ndarray
    xhat_red: np.ndarray
    u: np.ndarray
    y: np.ndarray
    V_traj: np.ndarray
    metrics: dict = field(default_factory=dict)


def synthesize(
    channel: str = "sym",
    dt: float = DEFAULT_DT,
    horizon: int = DEFAULT_HORIZON,
    hull: str = DEFAULT_HULL,
    gridsize: int = DEFAULT_GRIDSIZE,
    Q: Optional[np.ndarray] = None,
    R: Optional[np.ndarray] = None,
    W: Optional[np.ndarray] = None,
    V_noise: Optional[np.ndarray] = None,
    V_meas_scale: float = DEFAULT_V_MEAS_SCALE,
    keep_blocks: Optional[list] = None,
    verbose: bool = True,
) -> SynthResult:
    """Synthesize state-feedback K and observer L for the FliPASED plant.

    Parameters
    ----------
    hull : str
        TP convex hull type for tptrans: 'cno', 'irno', 'ino', 'snnn', 'close'.
        Default 'cno' (closest-to-origin normal-form outer hull).
    gridsize : int
        Number of V-grid samples for tptrans. More = better HOSVD accuracy.
    """
    poly_full = flipased_polytope(channel)
    poly_red  = flipased_polytope_reduced(channel, keep_blocks=keep_blocks)
    poly_red_d  = discretize_polytope(poly_red, dt=dt)
    poly_full_d = discretize_polytope(poly_full, dt=dt)
    n_red = poly_red_d.n
    p     = poly_red_d.p

    if Q is None:       Q = _default_Q(n_red, channel)
    if R is None:       R = DEFAULT_R_CTRL.copy()
    if W is None:       W = _default_W(n_red)
    if V_noise is None: V_noise = V_meas_scale * np.eye(p)

    if verbose:
        print(f"synthesize({channel}): n_full={poly_full_d.n}, n_red={n_red}, "
              f"R_vertices={poly_red_d.R}, dt={dt}, N={horizon}, "
              f"hull={hull}, gridsize={gridsize}, V_meas_scale={V_meas_scale}")

    # ---- Proper TP model transformation (HOSVD + convex hull) ----
    tp_d = flipased_tpform(poly_red_d, hull=hull, gridsize=gridsize,
                           verbose=verbose)
    if verbose:
        vc = [u.shape[1] for u in tp_d.U]
        print(f"  TP form: core {tp_d.S.shape}, compressed vertices={vc}")

    # ---- Primal P-RHPG: state-feedback gain K ----
    sf = PRHPGProblem(tp_d, n=n_red)
    sf.set_stage_cost(Q=Q, R=R)
    sf.set_horizon(N=horizon)
    K_prhpg, J_K = sf.solve()
    # reshape (R_total * m, n) -> (R_total, m, n)
    R_total = int(np.prod([u.shape[1] for u in tp_d.U]))
    K_prhpg = K_prhpg.reshape(R_total, sf._m, n_red)
    # P-RHPG minimizes J with u = K x (A+BK stable); deployment is u = -Kx
    K_dep = -K_prhpg

    # ---- Dual P-RHPG: observer gain L ----
    C_mid = poly_red_d.C[poly_red_d.R // 2]   # (p, n) at V_mean
    obs = ObserverPRHPGProblem(tp=tp_d, C=C_mid, n=n_red)
    obs.set_noise_covariances(W=W, V=V_noise)
    obs.set_horizon(N=horizon)
    L_flat, J_L = obs.solve()
    L = L_flat.reshape(R_total, n_red, p)

    if verbose:
        K_norms = [float(np.linalg.norm(K_dep[r])) for r in range(R_total)]
        L_norms = [float(np.linalg.norm(L[r]))     for r in range(R_total)]
        print(f"  J_K={J_K:.3e}, J_L={J_L:.3e}")
        print(f"  ||K|| in [{min(K_norms):.2f}, {max(K_norms):.2f}]")
        print(f"  ||L|| in [{min(L_norms):.2f}, {max(L_norms):.2f}]")

    return SynthResult(
        channel=channel, dt=dt, horizon=horizon, hull=hull,
        poly_full=poly_full, poly_red=poly_red,
        poly_red_d=poly_red_d, poly_full_d=poly_full_d,
        K=K_dep, L=L, K_prhpg=K_prhpg,
        J_K=J_K, J_L=J_L,
        state_indices=poly_red.state_indices,
        tp=tp_d,
    )


def _blend_tp(matrices: np.ndarray, V: float, tp, domain: np.ndarray) -> np.ndarray:
    """
    Blend vertex matrices at airspeed V using the TP weighting functions.

    matrices : (R_total, ...) array of per-vertex matrices
    tp       : TPForm with .U weighting matrices and domain
    """
    from tptool.util import tpcontroller  # reuse alpha evaluation
    # Build alpha vector from tp.U weighting functions
    U_list = tp.U
    V_grid_tp = np.linspace(domain[0, 0], domain[0, 1], U_list[0].shape[0])
    V_clamped = float(np.clip(V, domain[0, 0], domain[0, 1]))

    # Linear interp into the TP weight table
    idx = (V_clamped - domain[0, 0]) / (domain[0, 1] - domain[0, 0]) * (U_list[0].shape[0] - 1)
    i0 = int(np.floor(idx))
    i0 = max(0, min(i0, U_list[0].shape[0] - 2))
    frac = idx - i0
    alpha = (1 - frac) * U_list[0][i0] + frac * U_list[0][i0 + 1]  # shape (R_total,)

    return np.einsum("r,r...->...", alpha, matrices)


def _blend_poly(matrices: np.ndarray, V: float, V_grid: np.ndarray) -> np.ndarray:
    """Blend vertex matrices using the original polytope grid (for full plant)."""
    a = alpha_at(V, V_grid)
    return np.einsum("r,r...->...", a, matrices)


def simulate_flutter_suppression(
    synth: SynthResult,
    V: Union[float, Callable[[float], float]],
    x0_full: Optional[np.ndarray] = None,
    flutter_kick: float = 0.1,
    T_final: float = 2.0,
    xhat0_red: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> SimResult:
    """Sample-and-hold closed-loop sim: full plant + reduced LQG controller.

    The controller (K, L) lives on the HOSVD-compressed TP grid; blending
    uses the TP weighting functions from tptrans. The truth plant uses the
    original polytope grid for interpolation.
    """
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

    # default IC: small kick on the flutter mode's first state in the FULL plant
    if x0_full is None:
        x0_full = np.zeros(n_full)
        if synth.channel == "sym":
            x0_full[8] = flutter_kick     # block 4 starts at full-state index 8
        elif synth.channel == "asym":
            x0_full[5] = flutter_kick     # block 3 starts at index 5

    if xhat0_red is None:
        xhat0_red = np.zeros(n_red)

    if callable(V):
        V_func = V
    else:
        V_const = float(V)
        V_func = lambda t: V_const

    n_steps = int(np.round(T_final / dt))
    t_grid = np.arange(n_steps + 1) * dt

    x_full   = np.zeros((n_full, n_steps + 1))
    xhat_red = np.zeros((n_red,  n_steps + 1))
    u_log    = np.zeros((m, n_steps + 1))
    y_log    = np.zeros((p, n_steps + 1))
    V_log    = np.zeros(n_steps + 1)

    x_full[:, 0]   = x0_full
    xhat_red[:, 0] = xhat0_red

    for k in range(n_steps):
        tk = t_grid[k]
        Vk = V_func(tk)
        V_log[k] = Vk

        # full-plant measurement (truth uses original polytope grid)
        C_full_k = _blend_poly(poly_full_d.C, Vk, V_grid)
        y_k = C_full_k @ x_full[:, k]
        y_log[:, k] = y_k

        # reduced-model matrices for observer (use original polytope grid)
        A_red_k = _blend_poly(poly_red_d.A, Vk, V_grid)
        B_red_k = _blend_poly(poly_red_d.B, Vk, V_grid)
        C_red_k = _blend_poly(poly_red_d.C, Vk, V_grid)

        # gains blended via TP weighting functions
        K_k = _blend_tp(synth.K, Vk, tp, domain)
        L_k = _blend_tp(synth.L, Vk, tp, domain)

        u_k = -K_k @ xhat_red[:, k]
        u_log[:, k] = u_k

        # observer prediction-correction
        xhat_pred = A_red_k @ xhat_red[:, k] + B_red_k @ u_k
        xhat_red[:, k + 1] = xhat_pred + L_k @ (y_k - C_red_k @ xhat_pred)

        # full plant ZOH step
        A_full_k = _blend_poly(poly_full_d.A, Vk, V_grid)
        B_full_k = _blend_poly(poly_full_d.B, Vk, V_grid)
        x_full[:, k + 1] = A_full_k @ x_full[:, k] + B_full_k @ u_k

    # final tick
    V_log[-1] = V_func(t_grid[-1])
    Vf = V_log[-1]
    C_full_f = _blend_poly(poly_full_d.C, Vf, V_grid)
    y_log[:, -1] = C_full_f @ x_full[:, -1]
    K_f = _blend_tp(synth.K, Vf, tp, domain)
    u_log[:, -1] = -K_f @ xhat_red[:, -1]

    # metrics
    x_red_truth = x_full[synth.state_indices, :]
    flutter_idx_red = (6, 7) if synth.channel == "sym" else (3, 4)
    flutter_amp = np.linalg.norm(x_red_truth[list(flutter_idx_red), :], axis=0)
    est_err = np.linalg.norm(x_red_truth - xhat_red, axis=0)

    metrics = {
        "max_|u|": float(np.max(np.abs(u_log))),
        "max_flutter_amplitude": float(flutter_amp.max()),
        "final_flutter_amplitude": float(flutter_amp[-1]),
        "max_estimation_error": float(est_err.max()),
        "final_estimation_error": float(est_err[-1]),
        "max_|x_full|": float(np.max(np.abs(x_full))),
        "final_|x_full|": float(np.linalg.norm(x_full[:, -1])),
    }
    if verbose:
        print(f"sim done: V={V_log[0]:.1f}->{V_log[-1]:.1f} m/s over {T_final}s")
        print(f"  max |u|                   = {metrics['max_|u|']:.4e} (limit ~0.35)")
        print(f"  max flutter amplitude     = {metrics['max_flutter_amplitude']:.4e}")
        print(f"  final flutter amplitude   = {metrics['final_flutter_amplitude']:.4e}")
        print(f"  max  estimation error     = {metrics['max_estimation_error']:.4e}")
        print(f"  final estimation error    = {metrics['final_estimation_error']:.4e}")
        print(f"  final |x_full|            = {metrics['final_|x_full|']:.4e}")

    return SimResult(
        t=t_grid, x_full=x_full, xhat_red=xhat_red,
        u=u_log, y=y_log, V_traj=V_log, metrics=metrics,
    )


if __name__ == "__main__":
    print("=" * 70)
    print("FliPASED LQG pipeline self-test: flutter kick at V=55")
    print("=" * 70)
    synth = synthesize(channel="sym", hull="cno", verbose=True)
    print()
    sim = simulate_flutter_suppression(synth, V=55.0, T_final=2.0)
    print()
    print("trajectories ready:")
    print(f"  t:        {sim.t.shape}")
    print(f"  x_full:   {sim.x_full.shape}")
    print(f"  xhat_red: {sim.xhat_red.shape}")
    print(f"  u:        {sim.u.shape}")

    np.savez("/tmp/flipased_flutter_kick.npz",
             t=sim.t, x_full=sim.x_full, xhat_red=sim.xhat_red,
             u=sim.u, y=sim.y, V=sim.V_traj,
             state_indices=synth.state_indices)
    print("\nsaved /tmp/flipased_flutter_kick.npz")
