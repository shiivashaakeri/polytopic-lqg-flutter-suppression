"""
§6.7 Robustness margins (gain and phase margins across airspeed envelope).

Computes classical SISO gain and phase margins at the controller-input
breakpoint for both polytopic LQG controllers, swept across V ∈ [40, 61] m/s.

Loop transfer at the input:
    L(z) = C_ctrl(z) · P(z)
where
    P(z) = C_plant (z I - A)^-1 B                 (plant)
    C_ctrl(z) = -K (z I - A + BK + LC)^-1 L       (LQG controller, output-to-input)

For single-input plant (m=1), L(z) is scalar at each z. Compute margins by:
  - sweeping ω over a fine grid
  - finding the gain crossover ω_c (where |L| = 1)
  - finding the phase crossover ω_180 (where ∠L = -180°)
  - GM = 1 / |L(ω_180)|        (in dB: -20 log10 |L|)
  - PM = 180° + ∠L(ω_c)

Output:
  paper_figures/robustness_margins_sym.png       (GM and PM vs V for both methods)
  results/robustness_margins_sym.npz

Run from project root:
    python -m examples.lqg_flipased_demo.experiment_pqrnpo_robustness_margins
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import control as ct

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
V_MIN, V_MAX = 40.0, 61.0
N_V_SWEEP = 61   # 1 m/s spacing across the envelope

# Frequency grid for margin computation
N_FREQ = 4000
F_MIN_HZ = 0.01
F_MAX_HZ = 100.0   # Nyquist is 100 Hz at dt=5ms

HORIZON_FOR_REF = 200
INIT_PERTURB = 0.3
MAX_ITER_PQRNPO = 15
LMI_MARGIN_LOOSE = 0.001

COLORS = {
    "pqrnpo":  "#0072B2",
    "lmi":     "#D55E00",
    "VF":      "#CC79A7",
}


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
# Loop transfer at the input and margin computation
# --------------------------------------------------------------------------


def loop_transfer_at_input_ss(A, B, C, K, L_obs, dt):
    """Build the SISO loop transfer at the actuator input as a discrete-time
    state-space system.

    Plant: P(z) = C (zI - A)^-1 B
    Controller: x_hat_+ = (A - B K - L_obs C) x_hat + L_obs y, u = -K x_hat
    Loop at input (broken at u): u_broken -> P(z) -> y -> C_ctrl(z) -> u

    Augmented state x_aug = (x, x_hat):
        A_loop = [[A,             0                  ],
                  [L_obs C,       A - BK - L_obs C   ]]
        B_loop = [[B], [0]]
        C_loop = [[0,  -K]]

    Returns ct.StateSpace object.
    """
    n = A.shape[0]
    A_ctrl = A - B @ K - L_obs @ C
    A_loop = np.block([
        [A,                  np.zeros((n, n))],
        [L_obs @ C,          A_ctrl           ],
    ])
    B_loop = np.vstack([B, np.zeros_like(B)])
    C_loop = np.hstack([np.zeros((K.shape[0], n)), -K])
    D_loop = np.zeros((K.shape[0], B.shape[1]))
    return ct.ss(A_loop, B_loop, C_loop, D_loop, dt)


def sensitivity_peak(A, B, C, K, L_obs, dt, omega_rad_s):
    """Compute the sensitivity peak M_S = ||S||_∞ by sweeping ω.

    For SISO loop transfer L(jω):
        S(jω) = 1 / (1 + L(jω))
        M_S = sup_ω |S(jω)|

    Returns (M_S, omega_peak).
    """
    sys = loop_transfer_at_input_ss(A, B, C, K, L_obs, dt)
    mag, phase, _ = ct.frequency_response(sys, omega_rad_s)
    Lz = (mag * np.exp(1j * phase)).flatten()
    Sz = 1.0 / (1.0 + Lz)
    abs_S = np.abs(Sz)
    idx_peak = int(np.argmax(abs_S))
    return float(abs_S[idx_peak]), float(omega_rad_s[idx_peak])


def disk_margin_from_MS(M_S):
    """Disk margin α = 1/M_S. Returns equivalent GM (dB) and PM (deg)."""
    alpha = 1.0 / M_S
    if alpha >= 1.0:
        gm_dB_eq = np.inf
        pm_deg_eq = 180.0
    else:
        gm_dB_eq = 20.0 * np.log10((1.0 + alpha) / (1.0 - alpha))
        pm_deg_eq = 2.0 * np.rad2deg(np.arcsin(alpha / 2.0))
    return alpha, gm_dB_eq, pm_deg_eq


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
    print(f"§6.7 Robustness margins (channel={channel}, hull={HULL})")
    print("=" * 72)

    poly_red = flipased_polytope_reduced(channel)
    poly_red_d = discretize_polytope(poly_red, dt=DEFAULT_DT)
    n = poly_red_d.n
    p = poly_red_d.p
    tp_d = flipased_tpform(poly_red_d, hull=HULL, gridsize=DEFAULT_GRIDSIZE,
                            verbose=False)

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
    print(f"\nn={n}, m={m}, p={p}")

    # ---- synthesize ----
    print("\n[synth] P-QRNPO LQG ...")
    kappa_pq, lambda_pq = synthesize_pqrnpo(tp_d, n, p, Q, R, W_proc, V_meas,
                                                Sigma_0, A_grid_synth,
                                                B_grid_synth, alpha_synth,
                                                weights_synth, C_mid)
    print(f"  ||K|| = {np.linalg.norm(kappa_pq):.4e}")
    print(f"  ||L|| = {np.linalg.norm(lambda_pq):.4e}")

    print("\n[synth] LMI LQG (loose) ...")
    try:
        K_lmi, L_lmi = synthesize_lmi(tp_d, n, A_v_d, B_v_d, R_total, C_mid,
                                         LMI_MARGIN_LOOSE)
        print(f"  ||K|| = {np.linalg.norm(K_lmi):.4e}")
        print(f"  ||L|| = {np.linalg.norm(L_lmi):.4e}")
        lmi_ok = True
    except Exception as e:
        print(f"  FAILED: {e}")
        lmi_ok = False

    # ---- sweep V ----
    V_grid = np.linspace(V_MIN, V_MAX, N_V_SWEEP)
    weights_V = evaluate_weights_at_V(tp_d, V_grid)
    A_V = np.einsum("vi,iab->vab", weights_V, A_v_d)
    B_V = np.einsum("vi,iab->vab", weights_V, B_v_d)

    # gains at each V
    K_pq_V = np.einsum("vi,iab->vab", weights_V, kappa_pq)
    L_pq_V = np.einsum("vi,iab->vab", weights_V, lambda_pq)
    if lmi_ok:
        K_lmi_V = np.einsum("vi,iab->vab", weights_V, K_lmi)
        L_lmi_V = np.einsum("vi,iab->vab", weights_V, L_lmi)

    freqs_hz = np.logspace(np.log10(F_MIN_HZ), np.log10(F_MAX_HZ), N_FREQ)
    omega_rad_s = 2.0 * np.pi * freqs_hz

    MS_pq = np.full(N_V_SWEEP, np.nan)
    MS_lmi = np.full(N_V_SWEEP, np.nan)
    alpha_pq = np.full(N_V_SWEEP, np.nan)
    alpha_lmi = np.full(N_V_SWEEP, np.nan)

    print(f"\n[margins] sweeping V across {N_V_SWEEP} points ...")
    for v_idx in range(N_V_SWEEP):
        # P-QRNPO
        ms, _ = sensitivity_peak(A_V[v_idx], B_V[v_idx], C_mid,
                                       K_pq_V[v_idx], L_pq_V[v_idx],
                                       DEFAULT_DT, omega_rad_s)
        MS_pq[v_idx] = ms
        alpha_pq[v_idx], _, _ = disk_margin_from_MS(ms)

        if lmi_ok:
            ms, _ = sensitivity_peak(A_V[v_idx], B_V[v_idx], C_mid,
                                           K_lmi_V[v_idx], L_lmi_V[v_idx],
                                           DEFAULT_DT, omega_rad_s)
            MS_lmi[v_idx] = ms
            alpha_lmi[v_idx], _, _ = disk_margin_from_MS(ms)

        if v_idx % 10 == 0 or v_idx == N_V_SWEEP - 1:
            ms_pq_dB = 20 * np.log10(MS_pq[v_idx])
            line = (f"  V = {V_grid[v_idx]:5.2f}  "
                    f"P-QRNPO: M_S = {MS_pq[v_idx]:5.2f} ({ms_pq_dB:5.2f} dB), "
                    f"α = {alpha_pq[v_idx]:5.3f}")
            if lmi_ok:
                ms_lmi_dB = 20 * np.log10(MS_lmi[v_idx])
                line += (f"  |  LMI: M_S = {MS_lmi[v_idx]:5.2f} "
                         f"({ms_lmi_dB:5.2f} dB), α = {alpha_lmi[v_idx]:5.3f}")
            print(line)

    # ---- diagnostic: Nyquist plot at V_F for both methods ----
    v_F_idx = int(np.argmin(np.abs(V_grid - V_F_NOMINAL)))
    sys_pq_VF = loop_transfer_at_input_ss(A_V[v_F_idx], B_V[v_F_idx], C_mid,
                                                K_pq_V[v_F_idx], L_pq_V[v_F_idx],
                                                DEFAULT_DT)
    mag_pq, phase_pq, _ = ct.frequency_response(sys_pq_VF, omega_rad_s)
    Lz_pq_VF = (mag_pq * np.exp(1j * phase_pq)).flatten()

    if lmi_ok:
        sys_lmi_VF = loop_transfer_at_input_ss(A_V[v_F_idx], B_V[v_F_idx], C_mid,
                                                     K_lmi_V[v_F_idx], L_lmi_V[v_F_idx],
                                                     DEFAULT_DT)
        mag_lmi, phase_lmi, _ = ct.frequency_response(sys_lmi_VF, omega_rad_s)
        Lz_lmi_VF = (mag_lmi * np.exp(1j * phase_lmi)).flatten()

    # ---- plot: 3 panels (M_S in dB, disk margin, Nyquist diagnostic) ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0),
                                gridspec_kw={"wspace": 0.35})

    # M_S in dB
    ax = axes[0]
    ax.plot(V_grid, 20 * np.log10(MS_pq), "-", color=COLORS["pqrnpo"],
              lw=1.8, label="P-QRNPO LPV")
    if lmi_ok:
        ax.plot(V_grid, 20 * np.log10(MS_lmi), "--", color=COLORS["lmi"],
                  lw=1.5, label="LMI LPV (loose)")
    ax.axvline(V_F_NOMINAL, color=COLORS["VF"], lw=0.8, ls=":",
                 label=rf"$V_F = {V_F_NOMINAL}$ m/s")
    ax.axhline(6.0, color="k", lw=0.4, alpha=0.4)
    ax.text(V_grid[0] + 0.5, 6.5, r"6 dB ($M_S = 2$, acceptable)",
              fontsize=8, color="#666666")
    ax.axhline(3.0, color="k", lw=0.4, alpha=0.4, ls="--")
    ax.text(V_grid[0] + 0.5, 3.5, r"3 dB ($M_S = 1.4$, excellent)",
              fontsize=8, color="#666666")
    ax.set_xlabel(r"airspeed $V$ (m/s)")
    ax.set_ylabel(r"$\|S\|_\infty$ (dB)")
    ax.set_title(r"(a) Sensitivity peak vs $V$", fontsize=10)
    ax.set_xlim(V_MIN, V_MAX)
    ax.legend(frameon=False, loc="best", fontsize=9)
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in")

    # Disk margin α
    ax = axes[1]
    ax.plot(V_grid, alpha_pq, "-", color=COLORS["pqrnpo"], lw=1.8,
              label="P-QRNPO LPV")
    if lmi_ok:
        ax.plot(V_grid, alpha_lmi, "--", color=COLORS["lmi"], lw=1.5,
                  label="LMI LPV (loose)")
    ax.axvline(V_F_NOMINAL, color=COLORS["VF"], lw=0.8, ls=":",
                 label=rf"$V_F = {V_F_NOMINAL}$ m/s")
    ax.axhline(0.5, color="k", lw=0.4, alpha=0.4)
    ax.text(V_grid[0] + 0.5, 0.51, r"$\alpha = 0.5$ (≥9.5 dB GM, ≥29° PM eq)",
              fontsize=8, color="#666666")
    ax.set_xlabel(r"airspeed $V$ (m/s)")
    ax.set_ylabel(r"disk margin $\alpha = 1 / \|S\|_\infty$")
    ax.set_title(r"(b) Disk margin vs $V$", fontsize=10)
    ax.set_xlim(V_MIN, V_MAX)
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, loc="best", fontsize=9)
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in")

    # Nyquist at V_F (sanity check)
    ax = axes[2]
    ax.plot(Lz_pq_VF.real, Lz_pq_VF.imag, "-", color=COLORS["pqrnpo"],
              lw=1.4, label="P-QRNPO LPV")
    if lmi_ok:
        ax.plot(Lz_lmi_VF.real, Lz_lmi_VF.imag, "--", color=COLORS["lmi"],
                  lw=1.2, label="LMI LPV (loose)")
    ax.plot([-1], [0], "x", color="k", ms=8, mew=2)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="k", lw=0.4, ls=":", alpha=0.5)
    ax.set_xlabel(r"Re $L(j\omega)$")
    ax.set_ylabel(r"Im $L(j\omega)$")
    ax.set_title(rf"(c) Nyquist at $V_F = {V_F_NOMINAL}$ m/s", fontsize=10)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.axhline(0, color="k", lw=0.3, alpha=0.4)
    ax.axvline(0, color="k", lw=0.3, alpha=0.4)
    ax.legend(frameon=False, loc="best", fontsize=9)
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in")

    fig.suptitle(rf"FliPASED [{channel}, hull = {HULL}] robustness margins: "
                 rf"sensitivity peak and disk margin across the airspeed envelope",
                 fontsize=10, y=1.02)

    out_path = fig_dir / f"robustness_margins_{channel}.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out_path.name}")

    # ---- save data ----
    save_dict = {
        "V_grid": V_grid, "freqs_hz": freqs_hz,
        "MS_pq": MS_pq, "alpha_pq": alpha_pq,
    }
    if lmi_ok:
        save_dict["MS_lmi"] = MS_lmi
        save_dict["alpha_lmi"] = alpha_lmi
    np.savez(res_dir / f"robustness_margins_{channel}.npz", **save_dict)
    print(f"saved results/robustness_margins_{channel}.npz")


if __name__ == "__main__":
    main()
