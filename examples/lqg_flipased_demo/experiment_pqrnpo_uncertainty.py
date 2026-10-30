"""
§6.8 Structured uncertainty Monte Carlo.

Perturbs the vertex matrices of the LPV plant by a relative amount ε,
independently per vertex, and tests both controllers (synthesized on
the NOMINAL plant) against the perturbed plant. For each sample:
  - sweep V across the envelope
  - compute sensitivity peak M_S(V) at each V
  - record stability (binary) and worst-case M_S across V

Run for ε ∈ {5%, 10%, 20%} with 1000 samples each.

Perturbation model (per vertex r):
    A_pert^r = A^r + ε * ||A^r||_F * Delta_A^r / ||Delta_A^r||_F
    B_pert^r = B^r + ε * ||B^r||_F * Delta_B^r / ||Delta_B^r||_F
where Delta is iid Gaussian. This perturbs each vertex independently —
worst-case scenario where parameter errors are uncorrelated across
operating points.

Output figure (3 panels):
  (a) Stability probability vs ε — bar chart for both methods
  (b) Empirical CDF of worst-case M_S at ε = 10% — both methods
  (c) Box plots of worst-case M_S at all three ε levels — both methods

Plus a summary table printed.

Run from project root:
    python -m examples.lqg_flipased_demo.experiment_pqrnpo_uncertainty
"""

from __future__ import annotations

from pathlib import Path
import time

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
N_V_SWEEP_MC = 11   # coarser: every ~2.1 m/s, sufficient for worst-case capture

# Monte Carlo settings
EPSILON_LEVELS = [0.05, 0.10, 0.20]
N_SAMPLES = 1000
RNG_SEED = 0

# Frequency grid for M_S
N_FREQ = 1000
F_MIN_HZ = 0.05
F_MAX_HZ = 100.0

# Synthesis settings
HORIZON_FOR_REF = 200
INIT_PERTURB = 0.3
MAX_ITER_PQRNPO = 15
LMI_MARGIN_LOOSE = 0.001

# Stability cap: if M_S exceeds this, we declare instability robustly
M_S_INSTABILITY_THRESHOLD = 1e6
SPECTRAL_RADIUS_TOL = 1.0  # ρ ≥ 1 → unstable

COLORS = {
    "pqrnpo":  "#0072B2",
    "lmi":     "#D55E00",
    "VF":      "#CC79A7",
}


# --------------------------------------------------------------------------
# Synthesis helpers (same as §6.7)
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
# Robustness analysis on a perturbed plant
# --------------------------------------------------------------------------


def perturb_vertices(A_v, B_v, eps, rng):
    """Apply ε-magnitude relative perturbation to each vertex independently.

    For discretized matrices A_d ≈ I + A_c·dt, the identity dominates ||A_d||_F,
    so scaling ε to ||A_d||_F gives misleadingly small ε that's still
    destabilizing in continuous-time terms.

    Instead we perturb relative to the dynamics part:
        ||Delta_A||_F = ε * ||A_d - I||_F        (size of non-trivial dynamics)
        ||Delta_B||_F = ε * ||B_d||_F            (B is already small, OK)

    Returns (A_v_pert, B_v_pert), same shapes as inputs.
    """
    A_p = A_v.copy()
    B_p = B_v.copy()
    n = A_v.shape[1]
    for r in range(A_v.shape[0]):
        # Dynamics part of A: A - I (the deviation from no-dynamics)
        A_dyn_norm = np.linalg.norm(A_v[r] - np.eye(n), "fro")
        normB = np.linalg.norm(B_v[r], "fro")
        if A_dyn_norm > 0:
            dA = rng.standard_normal(A_v[r].shape)
            dA = dA / np.linalg.norm(dA, "fro")
            A_p[r] = A_v[r] + eps * A_dyn_norm * dA
        if normB > 0:
            dB = rng.standard_normal(B_v[r].shape)
            dB = dB / np.linalg.norm(dB, "fro")
            B_p[r] = B_v[r] + eps * normB * dB
    return A_p, B_p


def closed_loop_spectral_radius(A_p, B_p, C_p, A_n, B_n, C_n, K, L_obs):
    """Maximum |eigenvalue| of the closed loop with PERTURBED plant
    and NOMINAL-MODEL observer.

    True plant evolves with (A_p, B_p, C_p). Controller's internal observer
    model uses nominal (A_n, B_n, C_n). Augmented system on (x, x_hat):

        x_+    = A_p x      + B_p u
        x_hat_+ = (A_n - B_n K - L_obs C_n) x_hat + L_obs (C_p x)
        u      = -K x_hat

    so:
        x_+    = A_p x - B_p K x_hat
        x_hat_+ = L_obs C_p x + (A_n - B_n K - L_obs C_n) x_hat
    """
    A_aug = np.block([
        [A_p,                  -B_p @ K                ],
        [L_obs @ C_p,          A_n - B_n @ K - L_obs @ C_n],
    ])
    return float(np.max(np.abs(np.linalg.eigvals(A_aug))))


def loop_transfer_at_input_ss_perturbed(A_p, B_p, C_p, A_n, B_n, C_n,
                                              K, L_obs, dt):
    """Loop transfer at the input with PERTURBED plant and NOMINAL-MODEL
    observer. Augmented state x_aug = (x, x_hat); broken at u_in:

        x_+      = A_p x       + B_p u_broken
        x_hat_+  = L_obs C_p x + (A_n - B_n K - L_obs C_n) x_hat
        u        = -K x_hat
    """
    n = A_p.shape[0]
    A_ctrl_int = A_n - B_n @ K - L_obs @ C_n
    A_loop = np.block([
        [A_p,                    np.zeros((n, n))],
        [L_obs @ C_p,            A_ctrl_int        ],
    ])
    B_loop = np.vstack([B_p, np.zeros_like(B_p)])
    C_loop = np.hstack([np.zeros((K.shape[0], n)), -K])
    D_loop = np.zeros((K.shape[0], B_p.shape[1]))
    return ct.ss(A_loop, B_loop, C_loop, D_loop, dt)


def loop_transfer_at_input_ss(A, B, C, K, L_obs, dt):
    """Nominal version (kept for compatibility with §6.7-style calls)."""
    return loop_transfer_at_input_ss_perturbed(A, B, C, A, B, C, K, L_obs, dt)


def sensitivity_peak(A, B, C, K, L_obs, dt, omega_rad_s):
    """Nominal-plant sensitivity peak."""
    sys = loop_transfer_at_input_ss(A, B, C, K, L_obs, dt)
    mag, phase, _ = ct.frequency_response(sys, omega_rad_s)
    Lz = (mag * np.exp(1j * phase)).flatten()
    Sz = 1.0 / (1.0 + Lz)
    return float(np.max(np.abs(Sz)))


def sensitivity_peak_perturbed(A_p, B_p, C_p, A_n, B_n, C_n, K, L_obs,
                                     dt, omega_rad_s):
    """Sensitivity peak with perturbed plant + nominal-model observer."""
    sys = loop_transfer_at_input_ss_perturbed(A_p, B_p, C_p, A_n, B_n, C_n,
                                                    K, L_obs, dt)
    mag, phase, _ = ct.frequency_response(sys, omega_rad_s)
    Lz = (mag * np.exp(1j * phase)).flatten()
    Sz = 1.0 / (1.0 + Lz)
    return float(np.max(np.abs(Sz)))


def evaluate_perturbed_sample(A_v_pert, B_v_pert, A_v_nom, B_v_nom,
                                  weights_V, C_mid, K_blend_V, L_blend_V,
                                  dt, omega_rad_s):
    """Sweep V for perturbed plant + nominal-model observer + given controller.

    Returns (is_stable, worst_M_S).
    """
    A_V_p = np.einsum("vi,iab->vab", weights_V, A_v_pert)
    B_V_p = np.einsum("vi,iab->vab", weights_V, B_v_pert)
    A_V_n = np.einsum("vi,iab->vab", weights_V, A_v_nom)
    B_V_n = np.einsum("vi,iab->vab", weights_V, B_v_nom)

    is_stable = True
    worst_MS = -np.inf
    for v_idx in range(len(weights_V)):
        rho = closed_loop_spectral_radius(
            A_V_p[v_idx], B_V_p[v_idx], C_mid,
            A_V_n[v_idx], B_V_n[v_idx], C_mid,
            K_blend_V[v_idx], L_blend_V[v_idx])
        if rho >= SPECTRAL_RADIUS_TOL:
            return False, float(M_S_INSTABILITY_THRESHOLD)
        try:
            ms = sensitivity_peak_perturbed(
                A_V_p[v_idx], B_V_p[v_idx], C_mid,
                A_V_n[v_idx], B_V_n[v_idx], C_mid,
                K_blend_V[v_idx], L_blend_V[v_idx],
                dt, omega_rad_s)
            if not np.isfinite(ms):
                return False, float(M_S_INSTABILITY_THRESHOLD)
            if ms > worst_MS:
                worst_MS = ms
        except Exception:
            return False, float(M_S_INSTABILITY_THRESHOLD)
    return is_stable, worst_MS


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
    print(f"§6.8 Structured uncertainty Monte Carlo (channel={channel}, hull={HULL})")
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
    print(f"\nn={n}, m={m}, p={p}, R̃={R_total}")

    # ---- synthesize on NOMINAL plant ----
    print("\n[synth on nominal] P-QRNPO LQG ...")
    kappa_pq, lambda_pq = synthesize_pqrnpo(tp_d, n, p, Q, R, W_proc, V_meas,
                                                Sigma_0, A_grid_synth,
                                                B_grid_synth, alpha_synth,
                                                weights_synth, C_mid)
    print(f"  ||K|| = {np.linalg.norm(kappa_pq):.4e}")
    print(f"  ||L|| = {np.linalg.norm(lambda_pq):.4e}")

    print("\n[synth on nominal] LMI LQG (loose) ...")
    K_lmi, L_lmi = synthesize_lmi(tp_d, n, A_v_d, B_v_d, R_total, C_mid,
                                       LMI_MARGIN_LOOSE)
    print(f"  ||K|| = {np.linalg.norm(K_lmi):.4e}")
    print(f"  ||L|| = {np.linalg.norm(L_lmi):.4e}")

    # ---- precompute V grid and weights ----
    V_grid_mc = np.linspace(V_MIN, V_MAX, N_V_SWEEP_MC)
    weights_V = evaluate_weights_at_V(tp_d, V_grid_mc)
    K_pq_V = np.einsum("vi,iab->vab", weights_V, kappa_pq)
    L_pq_V = np.einsum("vi,iab->vab", weights_V, lambda_pq)
    K_lmi_V = np.einsum("vi,iab->vab", weights_V, K_lmi)
    L_lmi_V = np.einsum("vi,iab->vab", weights_V, L_lmi)

    freqs_hz = np.logspace(np.log10(F_MIN_HZ), np.log10(F_MAX_HZ), N_FREQ)
    omega_rad_s = 2.0 * np.pi * freqs_hz

    # ---- Monte Carlo ----
    results = {}  # results[eps][method] = {"stable": [], "worst_MS": []}
    for eps in EPSILON_LEVELS:
        results[eps] = {
            "pqrnpo": {"stable": np.zeros(N_SAMPLES, dtype=bool),
                          "worst_MS": np.zeros(N_SAMPLES)},
            "lmi":    {"stable": np.zeros(N_SAMPLES, dtype=bool),
                          "worst_MS": np.zeros(N_SAMPLES)},
        }

    rng = np.random.default_rng(RNG_SEED)

    print(f"\n[MC] running {N_SAMPLES} samples × {len(EPSILON_LEVELS)} ε levels"
          f" × 2 methods × {N_V_SWEEP_MC} V grid points ...")
    t0 = time.time()
    for eps_idx, eps in enumerate(EPSILON_LEVELS):
        print(f"\n  ε = {eps:.0%} ...")
        for s in range(N_SAMPLES):
            A_v_pert, B_v_pert = perturb_vertices(A_v_d, B_v_d, eps, rng)

            # P-QRNPO (perturbed plant + nominal-model observer)
            stab, ms = evaluate_perturbed_sample(A_v_pert, B_v_pert,
                                                       A_v_d, B_v_d,
                                                       weights_V, C_mid,
                                                       K_pq_V, L_pq_V,
                                                       DEFAULT_DT, omega_rad_s)
            results[eps]["pqrnpo"]["stable"][s] = stab
            results[eps]["pqrnpo"]["worst_MS"][s] = ms

            # LMI (perturbed plant + nominal-model observer)
            stab, ms = evaluate_perturbed_sample(A_v_pert, B_v_pert,
                                                       A_v_d, B_v_d,
                                                       weights_V, C_mid,
                                                       K_lmi_V, L_lmi_V,
                                                       DEFAULT_DT, omega_rad_s)
            results[eps]["lmi"]["stable"][s] = stab
            results[eps]["lmi"]["worst_MS"][s] = ms

            if (s + 1) % 100 == 0:
                elapsed = time.time() - t0
                pq_stab = np.mean(results[eps]["pqrnpo"]["stable"][:s + 1])
                lmi_stab = np.mean(results[eps]["lmi"]["stable"][:s + 1])
                print(f"    [{s + 1}/{N_SAMPLES}] elapsed={elapsed:5.1f}s  "
                      f"P-QRNPO P[stable]={pq_stab:.3f}  "
                      f"LMI P[stable]={lmi_stab:.3f}")

    total_time = time.time() - t0
    print(f"\n[MC] done in {total_time:.1f}s")

    # ---- print summary ----
    print("\n" + "=" * 72)
    print("Summary: P[stable across V envelope] and worst-case M_S statistics")
    print("=" * 72)
    print(f"{'ε':>5} | {'method':>10} | {'P[stable]':>10} | "
          f"{'median M_S':>10} | {'p95 M_S':>8} | {'max M_S':>10}")
    print("-" * 72)
    for eps in EPSILON_LEVELS:
        for method in ["pqrnpo", "lmi"]:
            stab = results[eps][method]["stable"]
            ms = results[eps][method]["worst_MS"]
            ms_stable = ms[stab]
            p_stable = np.mean(stab)
            if len(ms_stable) > 0:
                med = np.median(ms_stable)
                p95 = np.percentile(ms_stable, 95)
                mx = np.max(ms_stable)
            else:
                med = p95 = mx = np.nan
            method_label = "P-QRNPO" if method == "pqrnpo" else "LMI"
            print(f"{eps:>5.0%} | {method_label:>10} | "
                  f"{p_stable:>9.3f} | "
                  f"{med:>10.3f} | {p95:>8.3f} | {mx:>10.3f}")

    # ---- plot: 3 panels ----
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0),
                                gridspec_kw={"wspace": 0.32})

    # Panel (a): stability probability vs ε
    ax = axes[0]
    eps_arr = np.array(EPSILON_LEVELS)
    pq_p = np.array([np.mean(results[e]["pqrnpo"]["stable"]) for e in EPSILON_LEVELS])
    lmi_p = np.array([np.mean(results[e]["lmi"]["stable"]) for e in EPSILON_LEVELS])
    width = 0.012
    ax.bar(eps_arr - width / 2, pq_p, width=width, color=COLORS["pqrnpo"],
             label="P-QRNPO LPV", edgecolor="white", lw=0.5)
    ax.bar(eps_arr + width / 2, lmi_p, width=width, color=COLORS["lmi"],
             label="LMI LPV (loose)", edgecolor="white", lw=0.5,
             hatch="///", alpha=0.8)
    ax.set_xticks(eps_arr)
    ax.set_xticklabels([f"{e:.0%}" for e in EPSILON_LEVELS])
    ax.set_xlabel(r"perturbation level $\varepsilon$")
    ax.set_ylabel(r"P[stable across $V \in [40, 61]$]")
    ax.set_title("(a) Stability probability", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="k", lw=0.4, alpha=0.4)
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    ax.grid(True, axis="y", lw=0.3, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in")

    # Panel (b): empirical CDF of worst-case M_S at ε = 10%
    ax = axes[1]
    eps_focus = 0.10
    pq_ms = results[eps_focus]["pqrnpo"]["worst_MS"]
    lmi_ms = results[eps_focus]["lmi"]["worst_MS"]
    pq_stab_mask = results[eps_focus]["pqrnpo"]["stable"]
    lmi_stab_mask = results[eps_focus]["lmi"]["stable"]
    pq_ms_finite = pq_ms[pq_stab_mask]
    lmi_ms_finite = lmi_ms[lmi_stab_mask]
    pq_sorted = np.sort(pq_ms_finite)
    lmi_sorted = np.sort(lmi_ms_finite)
    pq_cdf = np.linspace(1.0 / N_SAMPLES, np.mean(pq_stab_mask), len(pq_sorted))
    lmi_cdf = np.linspace(1.0 / N_SAMPLES, np.mean(lmi_stab_mask), len(lmi_sorted))
    ax.plot(pq_sorted, pq_cdf, "-", color=COLORS["pqrnpo"], lw=1.8,
              label="P-QRNPO LPV")
    ax.plot(lmi_sorted, lmi_cdf, "--", color=COLORS["lmi"], lw=1.5,
              label="LMI LPV (loose)")
    ax.axvline(2.0, color="k", lw=0.4, alpha=0.4)
    ax.text(2.05, 0.05, r"$M_S = 2$ (acceptable)", fontsize=8, color="#666666")
    ax.set_xlabel(r"worst-case $M_S$ across envelope")
    ax.set_ylabel("empirical CDF")
    ax.set_title(r"(b) CDF of worst-case $M_S$ at $\varepsilon = 10\%$",
                   fontsize=10)
    ax.set_xlim(left=1.0)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="best", fontsize=9)
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in")

    # Panel (c): box plots of worst-case M_S at all three ε levels
    ax = axes[2]
    box_data = []
    box_labels = []
    box_colors = []
    for eps in EPSILON_LEVELS:
        pq_stab_mask = results[eps]["pqrnpo"]["stable"]
        lmi_stab_mask = results[eps]["lmi"]["stable"]
        pq_ms = results[eps]["pqrnpo"]["worst_MS"][pq_stab_mask]
        lmi_ms = results[eps]["lmi"]["worst_MS"][lmi_stab_mask]
        box_data.append(pq_ms)
        box_data.append(lmi_ms)
        box_labels.append(rf"P-Q" + "\n" + rf"$\varepsilon={int(eps*100)}\%$")
        box_labels.append(rf"LMI" + "\n" + rf"$\varepsilon={int(eps*100)}\%$")
        box_colors.append(COLORS["pqrnpo"])
        box_colors.append(COLORS["lmi"])
    bp = ax.boxplot(box_data, tick_labels=box_labels, widths=0.55,
                       patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
        patch.set_edgecolor(color)
    for median in bp["medians"]:
        median.set_color("k")
        median.set_linewidth(1.0)
    for whisker in bp["whiskers"] + bp["caps"]:
        whisker.set_color("#444444")
    ax.axhline(2.0, color="k", lw=0.4, alpha=0.4)
    ax.text(0.6, 2.1, r"$M_S = 2$ (acceptable)", fontsize=8, color="#666666")
    ax.set_ylabel(r"worst-case $M_S$")
    ax.set_title("(c) $M_S$ distribution by perturbation level", fontsize=10)
    ax.grid(True, axis="y", lw=0.3, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in")

    fig.suptitle(rf"FliPASED [{channel}, hull = {HULL}] structured uncertainty: "
                 rf"{N_SAMPLES} Monte Carlo samples per ε, "
                 rf"vertex-level relative perturbations, V swept [40, 61] m/s",
                 fontsize=10, y=1.02)

    out_path = fig_dir / f"uncertainty_mc_{channel}.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out_path.name}")

    # ---- save data ----
    save_dict = {"V_grid_mc": V_grid_mc, "EPSILON_LEVELS": np.array(EPSILON_LEVELS),
                  "N_SAMPLES": N_SAMPLES}
    for eps in EPSILON_LEVELS:
        for method in ["pqrnpo", "lmi"]:
            tag = f"{method}_eps{int(eps * 100)}"
            save_dict[f"{tag}_stable"] = results[eps][method]["stable"]
            save_dict[f"{tag}_worstMS"] = results[eps][method]["worst_MS"]
    np.savez(res_dir / f"uncertainty_mc_{channel}.npz", **save_dict)
    print(f"saved results/uncertainty_mc_{channel}.npz")


if __name__ == "__main__":
    main()
