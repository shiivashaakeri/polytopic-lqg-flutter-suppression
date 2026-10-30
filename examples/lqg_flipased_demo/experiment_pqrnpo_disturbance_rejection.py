"""
§6.5 Frequency-domain disturbance rejection.

Compares closed-loop frequency response from process noise w to output y for:
  - Open loop
  - P-QRNPO LQG
  - LMI-loose LQG (margin = 0.001)
  - LMI-strict LQG (margin = 0.05) [INFEASIBLE on CHull, reported only]

Plant: FliPASED reduced sym channel, hull = "close" (CHull, the loosest hull
where LMI is reliably feasible). P-QRNPO is hull-invariant (per §6.3.5), so
switching CNO→CHull does not change P-QRNPO results, but it gives LMI its
strongest deployment for fair comparison.

For each method, the closed-loop augmented system on z = (x, x_e) is:
    A_aug = [ A - BK,   BK         ]
            [ 0,        A - LC     ]
    B_w   = [ I; I ]            (process noise enters both halves through L)
    C_aug = [ C, 0 ]
The transfer w -> y is:
    G(z) = C_aug (zI - A_aug)^-1 B_w

For open loop: u = 0, so the transfer is just C (zI - A)^-1 W^{1/2}.

Validation metrics (per airspeed, per method):
  1. Bode magnitude |G(e^{j ω T_s})|, plotted vs ω across log frequency grid
  2. H2 norm: ||G||_2  (= LQG cost in the white-noise sense)
  3. Hinf norm: ||G||_∞ = sup_ω σ_max(G)

Output:
  paper_figures/disturbance_rejection_sym.png  (3-panel Bode plot)
  results/disturbance_rejection_sym.npz        (saved norms)

Run from project root:
    python -m examples.lqg_flipased_demo.experiment_pqrnpo_disturbance_rejection
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_discrete_lyapunov

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


# Test airspeeds (matches §6.4)
V_F_NOMINAL = 51.42
V_TEST = [45.0, V_F_NOMINAL, 58.0]
V_LABELS = [r"$V = 45$ m/s",
              rf"$V = V_F = {V_F_NOMINAL}$ m/s",
              r"$V = 58$ m/s"]

HULL = "close"   # CHull — loosest hull, LMI most reliably feasible

# Synthesis settings
HORIZON_FOR_REF = 200
INIT_PERTURB = 0.3
MAX_ITER_PQRNPO = 15

LMI_MARGIN_STRICT = 0.05
LMI_MARGIN_LOOSE  = 0.001

# Frequency grid (rad/s)
N_FREQ = 400
F_MIN_HZ = 0.1
F_MAX_HZ = 100.0    # Nyquist for dt=5ms is 100 Hz

COLORS = {
    "open":       "#999999",
    "pqrnpo":     "#0072B2",
    "lmi_loose":  "#D55E00",
    "lmi_strict": "#E69F00",
    "VF_marker":  "#CC79A7",
}


# --------------------------------------------------------------------------
# Synthesis helpers (same as §6.3 / §6.4)
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
# Closed-loop transfer function and norms
# --------------------------------------------------------------------------


def closed_loop_aug(A, B, C, K, L, n):
    """Build (A_aug, B_w, C_aug) for closed-loop augmented system on (x, x_e).

    Augmented dynamics:
        [x_+   ]   [ A - BK     BK     ] [x   ]   [I]      [ 0]
        [x_e,+ ] = [   0      A - LC   ] [x_e ] + [I] w  + [-L] v

    Output: y = C x + v = [C 0] (x, x_e) + v.
    For purposes of disturbance rejection plot, the "input" is w and we ignore v.

    Returns A_aug (2n × 2n), B_w (2n × n), C_aug (p × 2n).
    """
    A_BK = A - B @ K
    A_LC = A - L @ C
    A_aug = np.block([
        [A_BK, B @ K],
        [np.zeros_like(A), A_LC],
    ])
    # process noise enters both x and x_e (the full state, then the error dynamics
    # has w too via x_e = x - x_hat, since x evolves with w but x_hat doesn't)
    B_w = np.block([
        [np.eye(n)],
        [np.eye(n)],
    ])
    C_aug = np.hstack([C, np.zeros((C.shape[0], n))])
    return A_aug, B_w, C_aug


def open_loop_transfer(A, C, n):
    """Open-loop transfer w -> y with u = 0:
        x_+ = A x + I w
        y   = C x
    Returns A_ol = A, B_w_ol = I, C_ol = C.
    """
    return A, np.eye(n), C


def bode_magnitude(A_sys, B_sys, C_sys, freqs_hz, dt):
    """Compute |G(e^{jωT_s})| at given frequencies (Hz).

    G(z) = C (zI - A)^-1 B
    Returns array of shape (len(freqs),) — sigma_max of G at each freq.
    """
    omega_rad = 2.0 * np.pi * freqs_hz   # rad/s
    z = np.exp(1j * omega_rad * dt)       # e^{jωT_s}, shape (n_freq,)
    n_aug = A_sys.shape[0]
    mag = np.zeros(len(freqs_hz))
    for k in range(len(freqs_hz)):
        try:
            M = z[k] * np.eye(n_aug) - A_sys
            G = C_sys @ np.linalg.solve(M, B_sys)
            mag[k] = np.linalg.norm(G, ord=2)   # σ_max
        except np.linalg.LinAlgError:
            mag[k] = np.nan
    return mag


def h2_norm(A_sys, B_sys, C_sys):
    """Discrete-time H2 norm: ||G||_2^2 = tr(C P C^T) where A P A^T - P + B B^T = 0.
    Returns sqrt of that.
    """
    try:
        # solve_discrete_lyapunov solves: A_sys @ P @ A_sys.T - P + Q = 0 with Q = B_sys B_sys^T
        Q = B_sys @ B_sys.T
        P = solve_discrete_lyapunov(A_sys, Q)
        return float(np.sqrt(np.trace(C_sys @ P @ C_sys.T)))
    except np.linalg.LinAlgError:
        return np.nan


def hinf_norm(A_sys, B_sys, C_sys, freqs_hz, dt):
    """Approximate H_inf = sup_ω σ_max(G(e^{jωT_s})) over our frequency grid.
    (Exact discrete-time Hinf via Riccati would be better but this is fine
    for visualization-grade.)"""
    mag = bode_magnitude(A_sys, B_sys, C_sys, freqs_hz, dt)
    return float(np.nanmax(mag))


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
    print(f"§6.5 Frequency-domain disturbance rejection (channel={channel}, "
          f"hull={HULL})")
    print("=" * 72)

    poly_red = flipased_polytope_reduced(channel)
    poly_red_d = discretize_polytope(poly_red, dt=DEFAULT_DT)
    n = poly_red_d.n
    p = poly_red_d.p

    tp_d = flipased_tpform(poly_red_d, hull=HULL, gridsize=DEFAULT_GRIDSIZE,
                            verbose=True)

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
    print(f"\n[synth] P-QRNPO LQG ...")
    kappa_pq, lambda_pq = synthesize_pqrnpo(tp_d, n, p, Q, R_ctrl, W_proc,
                                                V_meas, Sigma_0, A_grid_synth,
                                                B_grid_synth, alpha_synth,
                                                weights_synth, C_mid)
    print(f"  ||K|| = {np.linalg.norm(kappa_pq):.4e}")
    print(f"  ||L|| = {np.linalg.norm(lambda_pq):.4e}")

    print(f"\n[synth] LMI LQG (loose, margin={LMI_MARGIN_LOOSE}) ...")
    try:
        K_lmi_loose, L_lmi_loose = synthesize_lmi(tp_d, n, A_v_d, B_v_d,
                                                       R_total, C_mid,
                                                       LMI_MARGIN_LOOSE)
        print(f"  ||K|| = {np.linalg.norm(K_lmi_loose):.4e}")
        print(f"  ||L|| = {np.linalg.norm(L_lmi_loose):.4e}")
        lmi_loose_ok = True
    except Exception as e:
        print(f"  FAILED: {e}")
        K_lmi_loose, L_lmi_loose = None, None
        lmi_loose_ok = False

    print(f"\n[synth] LMI LQG (strict, margin={LMI_MARGIN_STRICT}) ...")
    try:
        K_lmi_strict, L_lmi_strict = synthesize_lmi(tp_d, n, A_v_d, B_v_d,
                                                         R_total, C_mid,
                                                         LMI_MARGIN_STRICT)
        print(f"  ||K|| = {np.linalg.norm(K_lmi_strict):.4e}")
        print(f"  ||L|| = {np.linalg.norm(L_lmi_strict):.4e}")
        lmi_strict_ok = True
    except Exception as e:
        print(f"  FAILED (expected): {e}")
        K_lmi_strict, L_lmi_strict = None, None
        lmi_strict_ok = False

    # ---- evaluate at test airspeeds ----
    weights_test = evaluate_weights_at_V(tp_d, np.array(V_TEST))
    A_test = np.einsum("vi,iab->vab", weights_test, A_v_d)
    B_test = np.einsum("vi,iab->vab", weights_test, B_v_d)

    # blend gains at each V
    K_pq_test = np.einsum("vi,iab->vab", weights_test, kappa_pq)
    L_pq_test = np.einsum("vi,iab->vab", weights_test, lambda_pq)
    if lmi_loose_ok:
        K_lmi_test = np.einsum("vi,iab->vab", weights_test, K_lmi_loose)
        L_lmi_test = np.einsum("vi,iab->vab", weights_test, L_lmi_loose)

    # frequency grid
    freqs_hz = np.logspace(np.log10(F_MIN_HZ), np.log10(F_MAX_HZ), N_FREQ)

    # storage for norms
    h2_norms = {"open": [], "pqrnpo": [], "lmi_loose": []}
    hinf_norms = {"open": [], "pqrnpo": [], "lmi_loose": []}
    bode = {"open": [], "pqrnpo": [], "lmi_loose": []}

    for v_idx, V_val in enumerate(V_TEST):
        print(f"\n[freq-domain] V = {V_val} m/s")
        A = A_test[v_idx]
        B = B_test[v_idx]

        # open loop
        A_ol, B_ol, C_ol = open_loop_transfer(A, C_mid, n)
        # only stable systems have well-defined H2 norm; check
        rho_ol = max(np.abs(np.linalg.eigvals(A_ol)))
        if rho_ol < 1.0:
            mag_ol = bode_magnitude(A_ol, B_ol, C_ol, freqs_hz, DEFAULT_DT)
            h2_ol = h2_norm(A_ol, B_ol, C_ol)
            hinf_ol = hinf_norm(A_ol, B_ol, C_ol, freqs_hz, DEFAULT_DT)
        else:
            mag_ol = bode_magnitude(A_ol, B_ol, C_ol, freqs_hz, DEFAULT_DT)
            h2_ol = np.inf
            hinf_ol = np.inf
        bode["open"].append(mag_ol)
        h2_norms["open"].append(h2_ol)
        hinf_norms["open"].append(hinf_ol)
        print(f"  open loop:    ρ={rho_ol:.4f}  H2={h2_ol:.3e}  Hinf={hinf_ol:.3e}")

        # P-QRNPO
        A_pq, B_pq, C_pq = closed_loop_aug(A, B, C_mid,
                                                K_pq_test[v_idx],
                                                L_pq_test[v_idx], n)
        mag_pq = bode_magnitude(A_pq, B_pq, C_pq, freqs_hz, DEFAULT_DT)
        h2_pq = h2_norm(A_pq, B_pq, C_pq)
        hinf_pq = hinf_norm(A_pq, B_pq, C_pq, freqs_hz, DEFAULT_DT)
        bode["pqrnpo"].append(mag_pq)
        h2_norms["pqrnpo"].append(h2_pq)
        hinf_norms["pqrnpo"].append(hinf_pq)
        print(f"  P-QRNPO LQG:  H2={h2_pq:.3e}  Hinf={hinf_pq:.3e}")

        # LMI-loose
        if lmi_loose_ok:
            A_lm, B_lm, C_lm = closed_loop_aug(A, B, C_mid,
                                                    K_lmi_test[v_idx],
                                                    L_lmi_test[v_idx], n)
            mag_lm = bode_magnitude(A_lm, B_lm, C_lm, freqs_hz, DEFAULT_DT)
            h2_lm = h2_norm(A_lm, B_lm, C_lm)
            hinf_lm = hinf_norm(A_lm, B_lm, C_lm, freqs_hz, DEFAULT_DT)
            bode["lmi_loose"].append(mag_lm)
            h2_norms["lmi_loose"].append(h2_lm)
            hinf_norms["lmi_loose"].append(hinf_lm)
            print(f"  LMI LQG:      H2={h2_lm:.3e}  Hinf={hinf_lm:.3e}")
        else:
            bode["lmi_loose"].append(None)
            h2_norms["lmi_loose"].append(np.nan)
            hinf_norms["lmi_loose"].append(np.nan)

    # ---- plot: 3 columns of Bode magnitude (P-QRNPO + open-loop only) ----
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2),
                                gridspec_kw={"wspace": 0.30})

    for v_idx, V_val in enumerate(V_TEST):
        ax = axes[v_idx]

        if bode["open"][v_idx] is not None:
            ax.semilogx(freqs_hz, 20 * np.log10(bode["open"][v_idx] + 1e-30),
                          "-", color=COLORS["open"], lw=1.2, label="open loop")
        if bode["pqrnpo"][v_idx] is not None:
            ax.semilogx(freqs_hz, 20 * np.log10(bode["pqrnpo"][v_idx] + 1e-30),
                          "-", color=COLORS["pqrnpo"], lw=1.6,
                          label="P-QRNPO LQG")

        ax.set_title(V_LABELS[v_idx], fontsize=10)
        ax.set_xlabel("frequency (Hz)")
        if v_idx == 0:
            ax.set_ylabel(r"$|G_{w \to y}|$ (dB)")
        ax.grid(True, which="both", lw=0.3, alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="in")
        if v_idx == 0:
            ax.legend(frameon=False, loc="upper right", fontsize=8.5)

    fig.suptitle(rf"FliPASED [{channel}, hull = {HULL}] disturbance rejection: "
                 rf"closed-loop transfer from process noise $w$ to output $y$ "
                 rf"under polytopic LQG via P-QRNPO",
                 fontsize=10, y=1.02)

    out_path = fig_dir / f"disturbance_rejection_{channel}.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out_path.name}")

    # ---- summary table printed ----
    print("\n" + "=" * 72)
    print(f"H2 and Hinf norms (CHull, both methods):")
    print("=" * 72)
    print(f"{'V (m/s)':>10} | {'method':>16} | {'H2':>12} | {'Hinf':>12}")
    print("-" * 60)
    for v_idx, V_val in enumerate(V_TEST):
        print(f"{V_val:>10.2f} | {'open loop':>16} | "
              f"{h2_norms['open'][v_idx]:>12.3e} | "
              f"{hinf_norms['open'][v_idx]:>12.3e}")
        print(f"{V_val:>10.2f} | {'P-QRNPO LQG':>16} | "
              f"{h2_norms['pqrnpo'][v_idx]:>12.3e} | "
              f"{hinf_norms['pqrnpo'][v_idx]:>12.3e}")
        if lmi_loose_ok:
            print(f"{V_val:>10.2f} | {'LMI LQG (loose)':>16} | "
                  f"{h2_norms['lmi_loose'][v_idx]:>12.3e} | "
                  f"{hinf_norms['lmi_loose'][v_idx]:>12.3e}")
        print()

    # ---- save .npz ----
    save_dict = {
        "V_TEST": np.array(V_TEST), "freqs_hz": freqs_hz, "hull": HULL,
        "kappa_pqrnpo": kappa_pq, "lambda_pqrnpo": lambda_pq,
        "h2_open":   np.array(h2_norms["open"]),
        "h2_pqrnpo": np.array(h2_norms["pqrnpo"]),
        "h2_lmi":    np.array(h2_norms["lmi_loose"]),
        "hinf_open":   np.array(hinf_norms["open"]),
        "hinf_pqrnpo": np.array(hinf_norms["pqrnpo"]),
        "hinf_lmi":    np.array(hinf_norms["lmi_loose"]),
        "lmi_loose_ok": lmi_loose_ok,
        "lmi_strict_ok": lmi_strict_ok,
    }
    if lmi_loose_ok:
        save_dict["K_lmi_loose"] = K_lmi_loose
        save_dict["L_lmi_loose"] = L_lmi_loose
    np.savez(res_dir / f"disturbance_rejection_{channel}.npz", **save_dict)
    print(f"saved results/disturbance_rejection_{channel}.npz")


if __name__ == "__main__":
    main()
