"""
§6.9 Actuator saturation.

Stress test of both LPV controllers (P-QRNPO, LMI) under hard actuator
saturation |u| ≤ u_max while executing the §6.6 round-trip airspeed
maneuver, with initial state perturbation in the flutter direction at V_LOW.

Sweep ||x_0|| amplitude on a log grid; record peak |y|, peak commanded
|u|, peak applied |u|, saturation duty cycle, and stability per sample.

Saturation feedback: the observer is fed the SATURATED (applied) u, not
the commanded u. This is standard practice — the observer should track
the true plant evolution, which depends on the actual applied control.

Output:
  paper_figures/saturation_sweep_sym.png  (3 panels: time-domain, peak |y|, duty cycle)
  results/saturation_sweep_sym.npz        (sweep arrays + time-domain at the
                                            illustrative amplitude)

Run from project root:
    python -m examples.lqg_flipased_demo.experiment_pqrnpo_saturation
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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


HULL = "close"
V_F_NOMINAL = 51.42
V_LOW, V_HIGH = 40.0, 60.0
T_FINAL = 4.0
RAMP_UP_END = 1.5
HOLD_END = 2.0
RAMP_DOWN_END = 3.5

U_MAX = 0.44  # 25°  in radians (FliPASED full actuator range)

# Synthesis settings
HORIZON_FOR_REF = 200
INIT_PERTURB = 0.3
MAX_ITER_PQRNPO = 15
LMI_MARGIN_LOOSE = 0.001

# Sweep settings
N_AMPLITUDES = 20
AMP_MIN = 0.1
AMP_MAX = 50.0

# Stability threshold
STAB_Y_THRESHOLD = 200.0   # rad/s — if |y|_inf exceeds this, declare unstable

# Illustrative amplitude for time-domain panel
AMP_ILLUSTRATIVE = 8.0   # likely near saturation boundary

RNG_SEED = 42

COLORS = {
    "pqrnpo": "#0072B2",
    "lmi":    "#D55E00",
    "VF":     "#CC79A7",
    "umax":   "#888888",
    "V_curve": "#444444",
}


# --------------------------------------------------------------------------
# Maneuver
# --------------------------------------------------------------------------


def airspeed_profile(t):
    def smoothstep(x):
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)
    V = np.zeros_like(t)
    mask1 = t < RAMP_UP_END
    V[mask1] = V_LOW + (V_HIGH - V_LOW) * smoothstep(t[mask1] / RAMP_UP_END)
    mask2 = (t >= RAMP_UP_END) & (t < HOLD_END)
    V[mask2] = V_HIGH
    mask3 = (t >= HOLD_END) & (t < RAMP_DOWN_END)
    s = (t[mask3] - HOLD_END) / (RAMP_DOWN_END - HOLD_END)
    V[mask3] = V_HIGH - (V_HIGH - V_LOW) * smoothstep(s)
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


# --------------------------------------------------------------------------
# Saturated-control simulation
# --------------------------------------------------------------------------


def simulate_lqg_lpv_saturated(A_t, B_t, C, K_t, L_t, x0, xhat0, T_steps,
                                    W_proc, V_meas, u_max, rng):
    """LPV LQG simulation with hard actuator saturation.

    Observer is fed the SATURATED u (anti-windup-aware feedback path).

    Returns:
        x: (T+1, n)
        u_cmd: (T, m)        — what the controller wanted
        u_appl: (T, m)       — what was actually applied (saturated)
        sat_count: int       — # of timesteps where any |u_cmd[i]| > u_max
    """
    n = A_t.shape[1]
    m = B_t.shape[2]
    p = C.shape[0]
    x = np.zeros((T_steps + 1, n))
    xhat = np.zeros((T_steps + 1, n))
    u_cmd = np.zeros((T_steps, m))
    u_appl = np.zeros((T_steps, m))
    x[0] = x0
    xhat[0] = xhat0
    sqrt_W = np.linalg.cholesky(W_proc + 1e-12 * np.eye(n))
    sqrt_V = np.linalg.cholesky(V_meas + 1e-12 * np.eye(p))
    sat_count = 0
    for t in range(T_steps):
        v_meas = sqrt_V @ rng.standard_normal(p)
        y = C @ x[t] + v_meas
        u_cmd[t] = -K_t[t] @ xhat[t]
        # apply saturation
        u_appl[t] = np.clip(u_cmd[t], -u_max, u_max)
        if np.any(np.abs(u_cmd[t]) > u_max + 1e-12):
            sat_count += 1
        # plant evolves with APPLIED u
        w_proc = sqrt_W @ rng.standard_normal(n)
        x[t + 1] = A_t[t] @ x[t] + B_t[t] @ u_appl[t] + w_proc
        # observer is fed the APPLIED u (anti-windup-aware)
        innov = y - C @ xhat[t]
        xhat[t + 1] = A_t[t] @ xhat[t] + B_t[t] @ u_appl[t] + L_t[t] @ innov
    return x, u_cmd, u_appl, sat_count


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
    print(f"§6.9 Actuator saturation (channel={channel}, hull={HULL}, "
          f"u_max={U_MAX:.3f} rad = {np.rad2deg(U_MAX):.1f}°)")
    print("=" * 72)

    poly_red = flipased_polytope_reduced(channel)
    poly_red_d = discretize_polytope(poly_red, dt=DEFAULT_DT)
    n = poly_red_d.n
    p = poly_red_d.p
    tp_d = flipased_tpform(poly_red_d, hull=HULL, gridsize=DEFAULT_GRIDSIZE,
                            verbose=False)

    Q = _default_Q(n, channel)
    R_ctrl = DEFAULT_R_CTRL.copy()
    W_proc = W_PROC * np.eye(n)
    V_meas = V_MEAS * np.eye(p)
    Sigma_0 = np.eye(n)

    A_v_d, B_v_d, R_total, m = _extract_vertices(tp_d.S, n)
    alpha_synth = _evaluate_weights_on_grid(tp_d.U)
    A_grid_synth, B_grid_synth = _evaluate_AB_on_grid(A_v_d, B_v_d, alpha_synth)
    weights_synth = _build_quadrature_weights(tp_d.U, tp_d.domain)
    C_mid = poly_red_d.C[poly_red_d.R // 2]
    print(f"\nn={n}, m={m}, p={p}, R̃={R_total}")

    # ---- synthesize ----
    print("\n[synth] P-QRNPO LQG ...")
    kappa_pq, lambda_pq = synthesize_pqrnpo(tp_d, n, p, Q, R_ctrl, W_proc,
                                                V_meas, Sigma_0, A_grid_synth,
                                                B_grid_synth, alpha_synth,
                                                weights_synth, C_mid)
    print(f"  ||K_pq|| = {np.linalg.norm(kappa_pq):.4e}")
    print(f"  ||L_pq|| = {np.linalg.norm(lambda_pq):.4e}")

    print("\n[synth] LMI LQG (loose) ...")
    K_lmi, L_lmi = synthesize_lmi(tp_d, n, A_v_d, B_v_d, R_total, C_mid,
                                       LMI_MARGIN_LOOSE)
    print(f"  ||K_lmi|| = {np.linalg.norm(K_lmi):.4e}")
    print(f"  ||L_lmi|| = {np.linalg.norm(L_lmi):.4e}")

    # ---- build time-varying plant (round-trip V) ----
    T_steps = int(T_FINAL / DEFAULT_DT)
    t_grid = np.arange(T_steps + 1) * DEFAULT_DT
    V_t = airspeed_profile(t_grid[:-1])
    weights_t = evaluate_weights_at_V(tp_d, V_t)
    A_t = np.einsum("ti,iab->tab", weights_t, A_v_d)
    B_t = np.einsum("ti,iab->tab", weights_t, B_v_d)
    K_pq_t = np.einsum("ti,iab->tab", weights_t, kappa_pq)
    L_pq_t = np.einsum("ti,iab->tab", weights_t, lambda_pq)
    K_lmi_t = np.einsum("ti,iab->tab", weights_t, K_lmi)
    L_lmi_t = np.einsum("ti,iab->tab", weights_t, L_lmi)

    # flutter eigenvector at V_LOW
    A_at_VL = np.einsum("vi,iab->vab",
                          evaluate_weights_at_V(tp_d, np.array([V_LOW])),
                          A_v_d)[0]
    w_eig, V_eig = np.linalg.eig(A_at_VL)
    worst = int(np.argmax(np.abs(w_eig)))
    v_flutter = np.real(V_eig[:, worst])
    v_flutter = v_flutter / (np.linalg.norm(v_flutter) + 1e-14)

    # ---- amplitude sweep ----
    amplitudes = np.logspace(np.log10(AMP_MIN), np.log10(AMP_MAX), N_AMPLITUDES)
    print(f"\n[sweep] {N_AMPLITUDES} amplitudes from {AMP_MIN} to {AMP_MAX}")

    peak_y_pq = np.zeros(N_AMPLITUDES)
    peak_u_cmd_pq = np.zeros(N_AMPLITUDES)
    sat_frac_pq = np.zeros(N_AMPLITUDES)
    stable_pq = np.zeros(N_AMPLITUDES, dtype=bool)

    peak_y_lmi = np.zeros(N_AMPLITUDES)
    peak_u_cmd_lmi = np.zeros(N_AMPLITUDES)
    sat_frac_lmi = np.zeros(N_AMPLITUDES)
    stable_lmi = np.zeros(N_AMPLITUDES, dtype=bool)

    # also keep the time-domain trajectory at the illustrative amplitude
    illustrative_idx = int(np.argmin(np.abs(amplitudes - AMP_ILLUSTRATIVE)))
    AMP_ILL_USED = float(amplitudes[illustrative_idx])
    illustrative = {}

    xhat0 = np.zeros(n)

    for i, amp in enumerate(amplitudes):
        x0 = amp * v_flutter

        # P-QRNPO
        rng = np.random.default_rng(RNG_SEED)
        x_pq, u_cmd_pq, u_appl_pq, sat_pq = simulate_lqg_lpv_saturated(
            A_t, B_t, C_mid, K_pq_t, L_pq_t,
            x0, xhat0, T_steps, W_proc, V_meas, U_MAX, rng)
        y_pq = (C_mid @ x_pq.T).T[:, 0]
        peak_y_pq[i] = np.max(np.abs(y_pq))
        peak_u_cmd_pq[i] = np.max(np.abs(u_cmd_pq))
        sat_frac_pq[i] = sat_pq / T_steps
        stable_pq[i] = peak_y_pq[i] < STAB_Y_THRESHOLD

        # LMI
        rng = np.random.default_rng(RNG_SEED)
        x_lmi, u_cmd_lmi, u_appl_lmi, sat_lmi = simulate_lqg_lpv_saturated(
            A_t, B_t, C_mid, K_lmi_t, L_lmi_t,
            x0, xhat0, T_steps, W_proc, V_meas, U_MAX, rng)
        y_lmi = (C_mid @ x_lmi.T).T[:, 0]
        peak_y_lmi[i] = np.max(np.abs(y_lmi))
        peak_u_cmd_lmi[i] = np.max(np.abs(u_cmd_lmi))
        sat_frac_lmi[i] = sat_lmi / T_steps
        stable_lmi[i] = peak_y_lmi[i] < STAB_Y_THRESHOLD

        if i == illustrative_idx:
            illustrative = {
                "y_pq": y_pq, "u_cmd_pq": u_cmd_pq, "u_appl_pq": u_appl_pq,
                "y_lmi": y_lmi, "u_cmd_lmi": u_cmd_lmi, "u_appl_lmi": u_appl_lmi,
            }

        marker_pq = "✓" if stable_pq[i] else "✗"
        marker_lmi = "✓" if stable_lmi[i] else "✗"
        print(f"  amp={amp:6.2f}  P-QRNPO {marker_pq} |y|={peak_y_pq[i]:7.2f} "
              f"|u|_cmd={peak_u_cmd_pq[i]:6.3f} sat={sat_frac_pq[i]:5.1%}  |  "
              f"LMI {marker_lmi} |y|={peak_y_lmi[i]:7.2f} "
              f"|u|_cmd={peak_u_cmd_lmi[i]:6.3f} sat={sat_frac_lmi[i]:5.1%}")

    # ---- SAVE FIRST (lesson learned from §6.8) ----
    save_dict = {
        "amplitudes": amplitudes,
        "peak_y_pq": peak_y_pq,    "peak_u_cmd_pq": peak_u_cmd_pq,
        "sat_frac_pq": sat_frac_pq, "stable_pq": stable_pq,
        "peak_y_lmi": peak_y_lmi,    "peak_u_cmd_lmi": peak_u_cmd_lmi,
        "sat_frac_lmi": sat_frac_lmi, "stable_lmi": stable_lmi,
        "U_MAX": U_MAX, "AMP_ILL_USED": AMP_ILL_USED,
        "t_grid": t_grid, "V_t": V_t,
        "ill_y_pq": illustrative["y_pq"],
        "ill_u_cmd_pq": illustrative["u_cmd_pq"],
        "ill_u_appl_pq": illustrative["u_appl_pq"],
        "ill_y_lmi": illustrative["y_lmi"],
        "ill_u_cmd_lmi": illustrative["u_cmd_lmi"],
        "ill_u_appl_lmi": illustrative["u_appl_lmi"],
    }
    np.savez(res_dir / f"saturation_sweep_{channel}.npz", **save_dict)
    print(f"\nsaved results/saturation_sweep_{channel}.npz")

    # ---- plot 3 stacked panels ----
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 9.0),
                                gridspec_kw={"hspace": 0.32,
                                             "height_ratios": [2.5, 1.5, 1.5]})

    # Panel (a): time-domain at illustrative amplitude (3 sub-axes via twinx)
    ax = axes[0]
    ax.plot(t_grid, illustrative["y_pq"], "-", color=COLORS["pqrnpo"],
              lw=1.4, label=r"$y_{\rm sym}$ P-QRNPO")
    ax.plot(t_grid, illustrative["y_lmi"], "--", color=COLORS["lmi"],
              lw=1.2, label=r"$y_{\rm sym}$ LMI")
    ax.set_xlabel(r"time $t$ (s)")
    ax.set_ylabel(r"output $y_{\rm sym}$ (rad/s)")
    ax.set_title(rf"(a) Time-domain at $\|x_0\| = {AMP_ILL_USED:.2f}$  "
                   rf"(round-trip $V$ profile, $u_{{\max}} = "
                   rf"{np.rad2deg(U_MAX):.1f}^\circ$)", fontsize=10)
    ax.set_xlim(0, T_FINAL)
    y_lim = max(np.max(np.abs(illustrative["y_pq"])),
                  np.max(np.abs(illustrative["y_lmi"]))) * 1.2
    ax.set_ylim(-y_lim, y_lim)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in")

    # twinx: u_cmd
    ax2 = ax.twinx()
    ax2.plot(t_grid[:-1], illustrative["u_cmd_pq"][:, 0], "-",
                color=COLORS["pqrnpo"], lw=0.9, alpha=0.5, label="u_cmd P-QRNPO")
    ax2.plot(t_grid[:-1], illustrative["u_cmd_lmi"][:, 0], "--",
                color=COLORS["lmi"], lw=0.9, alpha=0.5, label="u_cmd LMI")
    ax2.axhline(U_MAX, color=COLORS["umax"], lw=0.6, ls="--")
    ax2.axhline(-U_MAX, color=COLORS["umax"], lw=0.6, ls="--")
    ax2.text(T_FINAL * 0.95, U_MAX * 1.05, rf"$\pm u_{{\max}}$", fontsize=8,
                color=COLORS["umax"], ha="right")
    ax2.set_ylabel(r"control $u$ (rad)")
    u_lim = max(U_MAX * 1.5,
                  np.max(np.abs(illustrative["u_cmd_pq"])) * 1.1,
                  np.max(np.abs(illustrative["u_cmd_lmi"])) * 1.1)
    ax2.set_ylim(-u_lim, u_lim)
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(direction="in")

    # Panel (b): peak |y| vs amplitude
    ax = axes[1]
    ax.loglog(amplitudes[stable_pq], peak_y_pq[stable_pq], "o-",
                color=COLORS["pqrnpo"], lw=1.6, ms=5, label="P-QRNPO LPV (stable)")
    ax.loglog(amplitudes[~stable_pq], peak_y_pq[~stable_pq], "x",
                color=COLORS["pqrnpo"], ms=8, mew=2, label="P-QRNPO LPV (unstable)")
    ax.loglog(amplitudes[stable_lmi], peak_y_lmi[stable_lmi], "s--",
                color=COLORS["lmi"], lw=1.4, ms=4, label="LMI LPV (stable)")
    ax.loglog(amplitudes[~stable_lmi], peak_y_lmi[~stable_lmi], "x",
                color=COLORS["lmi"], ms=8, mew=2, label="LMI LPV (unstable)")
    ax.axhline(STAB_Y_THRESHOLD, color="k", lw=0.5, ls=":", alpha=0.5)
    ax.text(amplitudes[0] * 1.2, STAB_Y_THRESHOLD * 1.1,
              rf"stability threshold $\|y\|_\infty < {STAB_Y_THRESHOLD:.0f}$",
              fontsize=8, color="#666666")
    ax.set_xlabel(r"initial perturbation amplitude $\|x_0\|$")
    ax.set_ylabel(r"peak $\|y\|_\infty$ (rad/s)")
    ax.set_title(r"(b) Peak output vs perturbation amplitude", fontsize=10)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, ncol=2)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in")

    # Panel (c): saturation duty cycle vs amplitude
    ax = axes[2]
    ax.semilogx(amplitudes, sat_frac_pq * 100, "o-",
                  color=COLORS["pqrnpo"], lw=1.6, ms=5, label="P-QRNPO LPV")
    ax.semilogx(amplitudes, sat_frac_lmi * 100, "s--",
                  color=COLORS["lmi"], lw=1.4, ms=4, label="LMI LPV")
    ax.set_xlabel(r"initial perturbation amplitude $\|x_0\|$")
    ax.set_ylabel(r"saturation duty cycle (\% of timesteps)")
    ax.set_title(r"(c) Saturation duty cycle vs perturbation amplitude",
                   fontsize=10)
    ax.set_ylim(0, 105)
    ax.legend(loc="best", frameon=False, fontsize=9)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in")

    fig.suptitle(rf"FliPASED [{channel}, hull = {HULL}] actuator saturation: "
                 rf"$u_{{\max}} = {np.rad2deg(U_MAX):.1f}^\circ$, "
                 rf"round-trip $V$ profile, "
                 rf"flutter-direction perturbation",
                 fontsize=10, y=0.995)

    out_path = fig_dir / f"saturation_sweep_{channel}.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path.name}")

    # ---- summary ----
    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    onset_pq = np.where(sat_frac_pq > 0.01)[0]
    onset_lmi = np.where(sat_frac_lmi > 0.01)[0]
    onset_pq_amp = amplitudes[onset_pq[0]] if len(onset_pq) > 0 else None
    onset_lmi_amp = amplitudes[onset_lmi[0]] if len(onset_lmi) > 0 else None
    print(f"Saturation onset (>1% of timesteps):")
    if onset_pq_amp is not None:
        print(f"  P-QRNPO: ||x_0|| = {onset_pq_amp:.2f}")
    else:
        print(f"  P-QRNPO: no saturation observed")
    if onset_lmi_amp is not None:
        print(f"  LMI:     ||x_0|| = {onset_lmi_amp:.2f}")
    else:
        print(f"  LMI:     no saturation observed")
    fail_pq = np.where(~stable_pq)[0]
    fail_lmi = np.where(~stable_lmi)[0]
    fail_pq_amp = amplitudes[fail_pq[0]] if len(fail_pq) > 0 else None
    fail_lmi_amp = amplitudes[fail_lmi[0]] if len(fail_lmi) > 0 else None
    print(f"\nFailure threshold (peak |y| > {STAB_Y_THRESHOLD}):")
    if fail_pq_amp is not None:
        print(f"  P-QRNPO: ||x_0|| = {fail_pq_amp:.2f}")
    else:
        print(f"  P-QRNPO: no failures up to ||x_0|| = {AMP_MAX}")
    if fail_lmi_amp is not None:
        print(f"  LMI:     ||x_0|| = {fail_lmi_amp:.2f}")
    else:
        print(f"  LMI:     no failures up to ||x_0|| = {AMP_MAX}")


if __name__ == "__main__":
    main()
