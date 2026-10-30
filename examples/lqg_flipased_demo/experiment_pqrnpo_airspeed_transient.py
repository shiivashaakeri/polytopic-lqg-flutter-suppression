"""
§6.6 Airspeed transient (LPV-in-action).

Demonstrates polytopic LQG controllers tracking a time-varying airspeed
profile through and past V_F, vs a non-LPV single-point LQG baseline.

Maneuver: round-trip V(t) = 40 → 60 → 40 m/s over 4 seconds, with smooth
sigmoid ramps. The aircraft accelerates from 40 to 60 m/s (passing through
V_F = 51.42 m/s mid-ramp), holds, then decelerates back to 40.

Controllers (4):
  1. Open-loop (no control)
  2. P-QRNPO LPV: gains K(V), L(V) blended via polytopic weights at each
                  timestep
  3. LMI LPV (loose, m=0.001): same architecture, LMI gains
  4. Single-point LQG at V_mid = 50 m/s: standard DARE + Kalman synthesis
                  at one airspeed, deployed UNCHANGED across the maneuver

Two figures (one per initial condition):
  A. Cold start: x_0 = 0, x̂_0 = 0, only process noise drives the system.
     Tests parameter-tracking behavior alone.
  B. Perturbed start: x_0 = flutter-mode eigenvector at V_0 = 40 m/s.
     Tests regulator transient + parameter tracking together.

Layout per figure (3 stacked panels):
  Top:    V(t) profile, with V_F marker
  Middle: y_sym(t) for all 4 controllers
  Bottom: u(t) for all controllers (open-loop has no u)

Run from project root:
    python -m examples.lqg_flipased_demo.experiment_pqrnpo_airspeed_transient
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_discrete_are

from examples.lqg_flipased_demo.dynamics import (
    discretize_polytope, flipased_polytope_reduced, flipased_tpform,
)
from examples.lqg_flipased_demo.experiment_p_qrnpo import (
    p_qrnpo, reference_optimum,
)
from examples.lqg_flipased_demo.experiment_p_qrnpo_dual import V_MEAS, W_PROC
from examples.lqg_flipased_demo.pipeline import (
    DEFAULT_DT, DEFAULT_GRIDSIZE, DEFAULT_R_CTRL, _default_Q,
)
from tptool.prhpg import ObserverPRHPGProblem
from tptool.prhpg.synthesis import (
    _build_quadrature_weights, _evaluate_AB_on_grid,
    _evaluate_weights_on_grid, _extract_vertices,
)
from tptool.lmi import LMIProblem
from tptool.lmi.synthesis import LMIObserverProblem


HULL = "close"   # CHull, where LMI is feasible (§6.5)
V_F_NOMINAL = 51.42

# Maneuver
V_LOW = 40.0
V_HIGH = 60.0
V_MID = 50.0   # synthesis point for single-point LQG
T_FINAL = 4.0
RAMP_UP_END = 1.5    # V reaches V_HIGH at t = 1.5 s
HOLD_END    = 2.0    # plateau at V_HIGH from 1.5 to 2.0 s
RAMP_DOWN_END = 3.5  # V reaches V_LOW at t = 3.5 s

# Synthesis settings
HORIZON_FOR_REF = 200
INIT_PERTURB = 0.3
MAX_ITER_PQRNPO = 15
LMI_MARGIN_LOOSE = 0.001

# Sim
PERTURB_AMP = 1.0
RNG_SEED = 42

COLORS = {
    "open":     "#999999",
    "pqrnpo":   "#0072B2",
    "lmi":      "#D55E00",
    "single":   "#CC79A7",
    "VF":       "#000000",
    "V_curve":  "#444444",
}


# --------------------------------------------------------------------------
# Maneuver: smooth round-trip V(t)
# --------------------------------------------------------------------------


def airspeed_profile(t):
    """Smooth round-trip V(t) ∈ [V_LOW, V_HIGH] over [0, T_FINAL].
    Uses smoothstep on each segment to avoid discontinuous V'."""
    def smoothstep(x):
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    V = np.zeros_like(t)
    # phase 1: 0 → RAMP_UP_END  : V_LOW → V_HIGH
    mask1 = t < RAMP_UP_END
    V[mask1] = V_LOW + (V_HIGH - V_LOW) * smoothstep(t[mask1] / RAMP_UP_END)
    # phase 2: RAMP_UP_END → HOLD_END  : V_HIGH plateau
    mask2 = (t >= RAMP_UP_END) & (t < HOLD_END)
    V[mask2] = V_HIGH
    # phase 3: HOLD_END → RAMP_DOWN_END  : V_HIGH → V_LOW
    mask3 = (t >= HOLD_END) & (t < RAMP_DOWN_END)
    s = (t[mask3] - HOLD_END) / (RAMP_DOWN_END - HOLD_END)
    V[mask3] = V_HIGH - (V_HIGH - V_LOW) * smoothstep(s)
    # phase 4: RAMP_DOWN_END → T_FINAL  : V_LOW plateau
    mask4 = t >= RAMP_DOWN_END
    V[mask4] = V_LOW
    return V


# --------------------------------------------------------------------------
# Synthesis helpers
# --------------------------------------------------------------------------


def evaluate_weights_at_V(tp, V_array):
    V_low, V_high = tp.domain[0]
    gridsize = tp.U[0].shape[0]
    V_table = np.linspace(V_low, V_high, gridsize)
    R_tilde = tp.U[0].shape[1]
    out = np.zeros((len(V_array), R_tilde))
    for r in range(R_tilde):
        out[:, r] = np.interp(V_array, V_table, tp.U[0][:, r])
    return out


def synthesize_pqrnpo(tp_d, n, p, Q, R, W_proc, V_meas, Sigma_0,
                        A_grid, B_grid, alpha, weights, C_mid):
    kappa_seed, _ = reference_optimum(tp_d, n, Q, R, Sigma_0, HORIZON_FOR_REF)
    kappa_init = (1.0 - INIT_PERTURB) * kappa_seed
    kappa, _ = p_qrnpo(kappa_init, A_grid, B_grid, alpha, weights, Q, R,
                          Sigma_0, max_iter=MAX_ITER_PQRNPO, verbose=False)
    obs = ObserverPRHPGProblem(tp=tp_d, C=C_mid, n=n)
    obs.set_noise_covariances(W=W_proc, V=V_meas)
    obs.set_horizon(N=HORIZON_FOR_REF)
    lam, _ = obs.solve()
    return kappa, lam


def synthesize_lmi(tp_d, n, A_v_d, B_v_d, R_total, C_mid, margin):
    reg = LMIProblem(tp_d.S, n=n, discrete=True)
    reg.add_stability(margin=margin)
    K_lmi, _ = reg.solve(solver="MOSEK", verbose=False)
    if K_lmi.ndim > 3:
        K_lmi = K_lmi.reshape(-1, K_lmi.shape[-2], K_lmi.shape[-1])
    v_mid = R_total // 2
    rho_minus = float(np.max(np.abs(np.linalg.eigvals(
        A_v_d[v_mid] - B_v_d[v_mid] @ K_lmi[v_mid]))))
    rho_plus = float(np.max(np.abs(np.linalg.eigvals(
        A_v_d[v_mid] + B_v_d[v_mid] @ K_lmi[v_mid]))))
    if rho_plus < 1.0 and rho_minus >= 1.0:
        K_lmi = -K_lmi

    obs = LMIObserverProblem(tp_d.S, C=C_mid, n=n, discrete=True)
    obs.add_stability(margin=margin)
    L_lmi, _ = obs.solve(solver="MOSEK", verbose=False)
    if L_lmi.ndim > 3:
        L_lmi = L_lmi.reshape(-1, L_lmi.shape[-2], L_lmi.shape[-1])
    return K_lmi, L_lmi


def synthesize_singlepoint_lqg(A_mid, B_mid, C, Q, R, W_proc, V_meas):
    """Standard discrete-time LQG at one fixed airspeed.

    Solves regulator DARE: A^T P A - P - A^T P B (R + B^T P B)^-1 B^T P A + Q = 0
    Solves Kalman DARE:    A P A^T - P - A P C^T (V + C P C^T)^-1 C P A^T + W = 0
    Returns (K_sp, L_sp).
    """
    P_reg = solve_discrete_are(A_mid, B_mid, Q, R)
    K_sp = np.linalg.solve(R + B_mid.T @ P_reg @ B_mid, B_mid.T @ P_reg @ A_mid)

    P_kal = solve_discrete_are(A_mid.T, C.T, W_proc, V_meas)
    L_sp = (P_kal @ C.T) @ np.linalg.inv(V_meas + C @ P_kal @ C.T)

    return K_sp, L_sp


# --------------------------------------------------------------------------
# Time-varying simulation
# --------------------------------------------------------------------------


def simulate_open_loop(A_t, B_t, x0, T_steps, W_proc, rng):
    """Open-loop with no control."""
    n = A_t.shape[1]
    m = B_t.shape[2]
    x = np.zeros((T_steps + 1, n))
    u = np.zeros((T_steps, m))
    x[0] = x0
    sqrt_W = np.linalg.cholesky(W_proc + 1e-12 * np.eye(n))
    for t in range(T_steps):
        w_proc = sqrt_W @ rng.standard_normal(n)
        x[t + 1] = A_t[t] @ x[t] + w_proc
    return x, u


def simulate_lqg_lpv(A_t, B_t, C, K_t, L_t, x0, xhat0, T_steps,
                       W_proc, V_meas, rng):
    """LPV LQG with time-varying gains."""
    n = A_t.shape[1]
    m = B_t.shape[2]
    p = C.shape[0]
    x = np.zeros((T_steps + 1, n))
    xhat = np.zeros((T_steps + 1, n))
    u = np.zeros((T_steps, m))
    x[0] = x0
    xhat[0] = xhat0
    sqrt_W = np.linalg.cholesky(W_proc + 1e-12 * np.eye(n))
    sqrt_V = np.linalg.cholesky(V_meas + 1e-12 * np.eye(p))
    for t in range(T_steps):
        v_meas = sqrt_V @ rng.standard_normal(p)
        y = C @ x[t] + v_meas
        u[t] = -K_t[t] @ xhat[t]
        w_proc = sqrt_W @ rng.standard_normal(n)
        x[t + 1] = A_t[t] @ x[t] + B_t[t] @ u[t] + w_proc
        innov = y - C @ xhat[t]
        xhat[t + 1] = A_t[t] @ xhat[t] + B_t[t] @ u[t] + L_t[t] @ innov
    return x, xhat, u


def simulate_singlepoint(A_t, B_t, C, K_sp, L_sp, x0, xhat0, T_steps,
                            W_proc, V_meas, rng):
    """LQG with constant single-point gains across time-varying plant.

    Note: K_sp and L_sp are constant, but the true plant A_t, B_t varies.
    The observer model also uses A_sp, B_sp (constant) — that's the whole
    point, the controller doesn't know V is changing.
    """
    n = A_t.shape[1]
    m = B_t.shape[2]
    p = C.shape[0]
    # Single-point uses A_t[T_steps // 2], B_t[T_steps // 2] in its INTERNAL
    # observer model. But that's already baked into K_sp, L_sp.
    A_sp = A_t[T_steps // 2]   # synthesis-point plant matrices
    B_sp = B_t[T_steps // 2]
    x = np.zeros((T_steps + 1, n))
    xhat = np.zeros((T_steps + 1, n))
    u = np.zeros((T_steps, m))
    x[0] = x0
    xhat[0] = xhat0
    sqrt_W = np.linalg.cholesky(W_proc + 1e-12 * np.eye(n))
    sqrt_V = np.linalg.cholesky(V_meas + 1e-12 * np.eye(p))
    for t in range(T_steps):
        v_meas = sqrt_V @ rng.standard_normal(p)
        y = C @ x[t] + v_meas
        u[t] = -K_sp @ xhat[t]
        w_proc = sqrt_W @ rng.standard_normal(n)
        # TRUE plant evolves with time-varying A_t, B_t
        x[t + 1] = A_t[t] @ x[t] + B_t[t] @ u[t] + w_proc
        # OBSERVER uses fixed A_sp, B_sp (it doesn't know V is changing)
        innov = y - C @ xhat[t]
        xhat[t + 1] = A_sp @ xhat[t] + B_sp @ u[t] + L_sp @ innov
    return x, xhat, u


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main():
    out_dir = Path("examples/lqg_flipased_demo")
    fig_dir = out_dir / "paper_figures"
    res_dir = out_dir / "results"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    channel = "sym"
    print("=" * 72)
    print(f"§6.6 Airspeed transient (channel={channel}, hull={HULL})")
    print("=" * 72)

    poly_red = flipased_polytope_reduced(channel)
    poly_red_d = discretize_polytope(poly_red, dt=DEFAULT_DT)
    n = poly_red_d.n
    p = poly_red_d.p
    tp_d = flipased_tpform(poly_red_d, hull=HULL, gridsize=DEFAULT_GRIDSIZE,
                            verbose=False)
    print(f"  TP: core {tp_d.S.shape}, hull = {HULL}")

    Q = _default_Q(n, channel)
    R = DEFAULT_R_CTRL.copy()
    W_proc = W_PROC * np.eye(n)
    V_meas = V_MEAS * np.eye(p)
    Sigma_0 = np.eye(n)

    A_v_d, B_v_d, R_total, m = _extract_vertices(tp_d.S, n)
    alpha_synth = _evaluate_weights_on_grid(tp_d.U)
    A_grid_synth, B_grid_synth = _evaluate_AB_on_grid(A_v_d, B_v_d, alpha_synth)
    weights_synth = _build_quadrature_weights(tp_d.U, tp_d.domain)
    C_mid = poly_red_d.C[poly_red_d.R // 2]

    # ---- synthesize P-QRNPO and LMI ----
    print("\n[synth] P-QRNPO LQG ...")
    kappa_pq, lambda_pq = synthesize_pqrnpo(tp_d, n, p, Q, R, W_proc, V_meas,
                                                Sigma_0, A_grid_synth,
                                                B_grid_synth, alpha_synth,
                                                weights_synth, C_mid)
    print(f"  ||K_pq|| = {np.linalg.norm(kappa_pq):.4e}, "
          f"||L_pq|| = {np.linalg.norm(lambda_pq):.4e}")

    print("\n[synth] LMI LQG (loose) ...")
    try:
        K_lmi, L_lmi = synthesize_lmi(tp_d, n, A_v_d, B_v_d, R_total, C_mid,
                                         LMI_MARGIN_LOOSE)
        print(f"  ||K_lmi|| = {np.linalg.norm(K_lmi):.4e}, "
              f"||L_lmi|| = {np.linalg.norm(L_lmi):.4e}")
        lmi_ok = True
    except Exception as e:
        print(f"  FAILED: {e}")
        K_lmi, L_lmi = None, None
        lmi_ok = False

    # ---- build time-varying plant ----
    T_steps = int(T_FINAL / DEFAULT_DT)
    t_grid = np.arange(T_steps + 1) * DEFAULT_DT
    V_t = airspeed_profile(t_grid[:-1])

    weights_t = evaluate_weights_at_V(tp_d, V_t)
    A_t = np.einsum("ti,iab->tab", weights_t, A_v_d)
    B_t = np.einsum("ti,iab->tab", weights_t, B_v_d)
    K_pq_t = np.einsum("ti,iab->tab", weights_t, kappa_pq)
    L_pq_t = np.einsum("ti,iab->tab", weights_t, lambda_pq)
    if lmi_ok:
        K_lmi_t = np.einsum("ti,iab->tab", weights_t, K_lmi)
        L_lmi_t = np.einsum("ti,iab->tab", weights_t, L_lmi)

    # cold start: x0 = 0, xhat0 = 0
    x0 = np.zeros(n)
    xhat0 = np.zeros(n)
    print(f"\n[sim] cold start (x0 = 0), running 2 controllers, T = {T_FINAL}s")

    rng_seed = RNG_SEED
    rng = np.random.default_rng(rng_seed)
    x_pq, _, u_pq = simulate_lqg_lpv(A_t, B_t, C_mid, K_pq_t, L_pq_t,
                                          x0, xhat0, T_steps, W_proc, V_meas, rng)

    if lmi_ok:
        rng = np.random.default_rng(rng_seed)
        x_lm, _, u_lm = simulate_lqg_lpv(A_t, B_t, C_mid, K_lmi_t, L_lmi_t,
                                              x0, xhat0, T_steps, W_proc, V_meas, rng)

    y_pq = (C_mid @ x_pq.T).T[:, 0]
    y_lm = (C_mid @ x_lm.T).T[:, 0] if lmi_ok else None

    print(f"\n[diagnostics]")
    print(f"  P-QRNPO    ||y||_inf = {np.max(np.abs(y_pq)):.3e}, "
          f"||u||_inf = {np.max(np.abs(u_pq)):.3e}")
    if lmi_ok:
        print(f"  LMI        ||y||_inf = {np.max(np.abs(y_lm)):.3e}, "
              f"||u||_inf = {np.max(np.abs(u_lm)):.3e}")

    # ---- plot 3 stacked panels ----
    fig, axes = plt.subplots(3, 1, figsize=(11, 8),
                                gridspec_kw={"hspace": 0.30,
                                             "height_ratios": [1.0, 2.0, 1.5]})

    # panel 1: V(t)
    ax = axes[0]
    ax.plot(t_grid[:-1], V_t, "-", color=COLORS["V_curve"], lw=1.8)
    ax.axhline(V_F_NOMINAL, color=COLORS["VF"], lw=0.6, ls="--",
                 label=rf"$V_F = {V_F_NOMINAL}$ m/s")
    ax.set_ylabel(r"$V$ (m/s)")
    ax.set_title("(a) Airspeed profile $V(t)$", fontsize=10)
    ax.set_xlim(0, T_FINAL)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in")

    # panel 2: y(t)
    ax = axes[1]
    ax.plot(t_grid, y_pq, "-", color=COLORS["pqrnpo"], lw=1.6,
              label="P-QRNPO LPV")
    if lmi_ok:
        ax.plot(t_grid, y_lm, "--", color=COLORS["lmi"], lw=1.4,
                  label="LMI LPV (loose)")
    ax.axhline(0, color="k", lw=0.4, alpha=0.4)
    ax.set_ylabel(r"output $y_{\rm sym}$ (rad/s)")
    ax.set_title("(b) Closed-loop output", fontsize=10)
    ax.set_xlim(0, T_FINAL)
    cl_max = max(np.max(np.abs(y_pq)),
                   np.max(np.abs(y_lm)) if lmi_ok else 0.0)
    ax.set_ylim(-1.3 * cl_max, 1.3 * cl_max)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in")

    # panel 3: u(t)
    ax = axes[2]
    ax.plot(t_grid[:-1], u_pq[:, 0], "-", color=COLORS["pqrnpo"], lw=1.4,
              label="P-QRNPO LPV")
    if lmi_ok:
        ax.plot(t_grid[:-1], u_lm[:, 0], "--", color=COLORS["lmi"], lw=1.2,
                  label="LMI LPV (loose)")
    ax.axhline(0, color="k", lw=0.4, alpha=0.4)
    ax.set_xlabel(r"time $t$ (s)")
    ax.set_ylabel(r"control $u$ (rad)")
    ax.set_title("(c) Control input", fontsize=10)
    ax.set_xlim(0, T_FINAL)
    u_max = max(np.max(np.abs(u_pq)),
                  np.max(np.abs(u_lm)) if lmi_ok else 0.0)
    ax.set_ylim(-1.3 * u_max, 1.3 * u_max)
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in")

    fig.suptitle(rf"FliPASED [{channel}, hull = {HULL}] airspeed transient: "
                 rf"round-trip $V$ = {V_LOW} → {V_HIGH} → {V_LOW} m/s",
                 fontsize=10, y=0.995)

    out_path = fig_dir / f"airspeed_transient_{channel}.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out_path.name}")

    save_dict = {
        "t_grid": t_grid, "V_t": V_t,
        "y_pq": y_pq, "u_pq": u_pq,
    }
    if lmi_ok:
        save_dict["y_lm"] = y_lm
        save_dict["u_lm"] = u_lm
    np.savez(res_dir / f"airspeed_transient_{channel}.npz", **save_dict)
    print(f"saved results/airspeed_transient_{channel}.npz")


if __name__ == "__main__":
    main()
