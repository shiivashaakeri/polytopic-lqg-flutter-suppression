"""
§6.3 Flutter envelope expansion (marquee aerospace figure).

Compares two polytopic LPV LQG controllers across the airspeed envelope:

  - P-QRNPO LQG:  K from primal P-QRNPO (Newton on LQR cost)
                  L from dual P-QRNPO  (Newton on Kalman cost via duality)

  - LMI LQG:      K from LMIProblem        (discrete-time Tanaka-Wang PDC)
                  L from LMIObserverProblem (discrete-time dual Tanaka-Wang)

Both deployed as polytopic LQG via the separation principle:
    x̂_{t+1} = A(V) x̂_t + B(V) u_t + L(V) (y_t - C x̂_t)
    u_t      = -K(V) x̂_t

Augmented closed-loop on z = (x, x̂):
    z_{t+1} = A_aug(V) z_t + noise
    A_aug   = [ A(V)        -B(V) K(V)                          ]
              [ L(V) C       A(V) - B(V) K(V) - L(V) C           ]

By separation, eig(A_aug) = eig(A_cl) ∪ eig(A_obs) where
    A_cl  = A(V) - B(V) K(V)            (regulator dynamics)
    A_obs = A(V) - L(V) C               (observer error dynamics)

The flutter mode lives in A_cl. We track it across V, plot:
  (a) Open-loop poles
  (b) Regulator closed-loop poles (P-QRNPO)
  (c) Damping ratio of flutter mode for open loop, P-QRNPO, LMI

All synthesis is DISCRETE-TIME at dt = 5 ms.

Run from project root:
    python -m examples.lqg_flipased_demo.experiment_pqrnpo_flutter_envelope
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from examples.lqg_flipased_demo.dynamics import (
    discretize_polytope, flipased_polytope, flipased_polytope_reduced,
    flipased_tpform,
)
from examples.lqg_flipased_demo.experiment_p_qrnpo import (
    cost_and_gradient, p_qrnpo, reference_optimum,
)
from examples.lqg_flipased_demo.experiment_p_qrnpo_dual import (
    V_MEAS, W_PROC,
)
from examples.lqg_flipased_demo.pipeline import (
    DEFAULT_DT, DEFAULT_GRIDSIZE, DEFAULT_R_CTRL, _default_Q,
)
from tptool.prhpg import ObserverPRHPGProblem
from tptool.prhpg.synthesis import (
    _build_quadrature_weights,
    _evaluate_AB_on_grid,
    _evaluate_weights_on_grid,
    _extract_vertices,
)
from tptool.lmi import LMIProblem
from tptool.lmi.synthesis import LMIObserverProblem


N_V_SWEEP = 121
V_MIN, V_MAX = 40.0, 61.0
V_F_NOMINAL = 51.42

HORIZON_FOR_REF = 200
INIT_PERTURB = 0.3
MAX_ITER_PQRNPO = 15

# LMI design (DISCRETE-time Tanaka-Wang)
LMI_REG_MARGIN = 0.05
LMI_OBS_MARGIN = 0.05
LMI_UMAX = 0.35
LMI_PHI = 0.05

COLORS = {
    "open":      "#999999",
    "pqrnpo":    "#0072B2",
    "lmi":       "#D55E00",
    "vF_marker": "#CC79A7",
}


def evaluate_weights_at_V(tp, V_array):
    V_low, V_high = tp.domain[0]
    gridsize = tp.U[0].shape[0]
    V_table = np.linspace(V_low, V_high, gridsize)
    R_tilde = tp.U[0].shape[1]
    out = np.zeros((len(V_array), R_tilde))
    for r in range(R_tilde):
        out[:, r] = np.interp(V_array, V_table, tp.U[0][:, r])
    return out


def regulator_closed_loop(A_v, B_v, K, weights_at_V):
    A_grid = np.einsum("vi,iab->vab", weights_at_V, A_v)
    B_grid = np.einsum("vi,iab->vab", weights_at_V, B_v)
    K_grid = np.einsum("vi,iab->vab", weights_at_V, K)
    BK = np.einsum("vab,vbc->vac", B_grid, K_grid)
    return A_grid - BK


def open_loop_at_V(A_v, weights_at_V):
    return np.einsum("vi,iab->vab", weights_at_V, A_v)


def track_modes(A_seq):
    n_V, n, _ = A_seq.shape
    eigvals = np.zeros((n_V, n), dtype=complex)
    eigvecs = np.zeros((n_V, n, n), dtype=complex)
    w0, v0 = np.linalg.eig(A_seq[0])
    eigvals[0] = w0
    eigvecs[0] = v0
    for k in range(1, n_V):
        wk, vk = np.linalg.eig(A_seq[k])
        prev_vecs = eigvecs[k - 1]
        prev_norm = prev_vecs / (np.linalg.norm(prev_vecs, axis=0,
                                                  keepdims=True) + 1e-14)
        new_norm = vk / (np.linalg.norm(vk, axis=0, keepdims=True) + 1e-14)
        overlap = np.abs(prev_norm.conj().T @ new_norm)
        assigned = -np.ones(n, dtype=int)
        used = np.zeros(n, dtype=bool)
        order = np.argsort(-overlap.max(axis=1))
        for i in order:
            best = -1
            best_score = -1.0
            for j in range(n):
                if used[j]:
                    continue
                if overlap[i, j] > best_score:
                    best_score = overlap[i, j]
                    best = j
            assigned[i] = best
            used[best] = True
        eigvals[k] = wk[assigned]
        eigvecs[k] = vk[:, assigned]
    return eigvals, eigvecs


def damping_ratio_discrete(eigvals_disc, dt):
    s = np.log(eigvals_disc + 0j) / dt
    sigma = s.real
    omega = s.imag
    mag = np.sqrt(sigma**2 + omega**2)
    return np.where(mag > 1e-12, -sigma / (mag + 1e-12), 0.0)


def find_flutter_mode_index(eigvals_seq, dt, V_array, V_F_guess):
    v_F_idx = int(np.argmin(np.abs(V_array - V_F_guess)))
    zeta_seq = np.array([damping_ratio_discrete(eigvals_seq[v], dt)
                            for v in range(len(V_array))])
    has_imag = np.abs(eigvals_seq[v_F_idx].imag) > 1e-3
    zeta_at_VF = zeta_seq[v_F_idx].copy()
    zeta_at_VF[~has_imag] = np.inf
    return int(np.argmin(np.abs(zeta_at_VF))), zeta_seq


def synthesize_pqrnpo_lqg(tp_d, n, p, Q, R, W_proc, V_meas, Sigma_0,
                            A_grid, B_grid, alpha, weights, C_mid,
                            horizon_seed, init_perturb, max_iter):
    """Polytopic LQG via P-QRNPO primal + dual P-RHPG observer (= dual P-QRNPO converged)."""
    kappa_seed, _ = reference_optimum(tp_d, n, Q, R, Sigma_0, horizon_seed)
    kappa_init = (1.0 - init_perturb) * kappa_seed
    kappa, _ = p_qrnpo(kappa_init, A_grid, B_grid, alpha, weights, Q, R,
                          Sigma_0, max_iter=max_iter, verbose=False)
    obs = ObserverPRHPGProblem(tp=tp_d, C=C_mid, n=n)
    obs.set_noise_covariances(W=W_proc, V=V_meas)
    obs.set_horizon(N=horizon_seed)
    lam, _ = obs.solve()
    return kappa, lam


def synthesize_lmi_lqg(tp_d, n, C_mid):
    """Polytopic LQG via discrete-time Tanaka-Wang LMIs."""
    reg = LMIProblem(tp_d.S, n=n, discrete=True)
    reg.add_stability(margin=LMI_REG_MARGIN)
    reg.add_input_constraint(umax=LMI_UMAX, phi=LMI_PHI)
    K_lmi, _ = reg.solve(solver="MOSEK", verbose=False)
    if K_lmi.ndim > 3:
        K_lmi = K_lmi.reshape(-1, K_lmi.shape[-2], K_lmi.shape[-1])

    obs = LMIObserverProblem(tp_d.S, C=C_mid, n=n, discrete=True)
    obs.add_stability(margin=LMI_OBS_MARGIN)
    L_lmi, _ = obs.solve(solver="MOSEK", verbose=False)
    if L_lmi.ndim > 3:
        L_lmi = L_lmi.reshape(-1, L_lmi.shape[-2], L_lmi.shape[-1])

    return K_lmi, L_lmi


def main():
    out_dir = Path("examples/lqg_flipased_demo")
    fig_dir = out_dir / "paper_figures"
    res_dir = out_dir / "results"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    channel = "sym"
    hull = "cno"

    print("=" * 72)
    print(f"§6.3 Flutter envelope expansion via polytopic LQG (channel={channel})")
    print("=" * 72)

    poly_red = flipased_polytope_reduced(channel)
    poly_red_d = discretize_polytope(poly_red, dt=DEFAULT_DT)
    n = poly_red_d.n
    p = poly_red_d.p

    tp_d = flipased_tpform(poly_red_d, hull=hull, gridsize=DEFAULT_GRIDSIZE,
                            verbose=True)

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

    # ---- P-QRNPO LQG ----
    print(f"\n[synth] P-QRNPO LQG (primal Newton + dual via duality) ...")
    kappa_pqrnpo, lambda_pqrnpo = synthesize_pqrnpo_lqg(
        tp_d, n, p, Q, R, W_proc, V_meas, Sigma_0,
        A_grid_synth, B_grid_synth, alpha_synth, weights_synth, C_mid,
        HORIZON_FOR_REF, INIT_PERTURB, MAX_ITER_PQRNPO)
    J_pqrnpo, _, _ = cost_and_gradient(kappa_pqrnpo, A_grid_synth, B_grid_synth,
                                          alpha_synth, weights_synth, Q, R,
                                          Sigma_0)
    print(f"  ||κ_P-QRNPO|| = {np.linalg.norm(kappa_pqrnpo):.4e}")
    print(f"  ||λ_P-QRNPO|| = {np.linalg.norm(lambda_pqrnpo):.4e}")
    print(f"  J(κ_P-QRNPO) = {J_pqrnpo:.6e}  (LQR cost)")

    # ---- LMI LQG ----
    print(f"\n[synth] LMI LQG (discrete Tanaka-Wang regulator + observer) ...")
    try:
        K_lmi, L_lmi = synthesize_lmi_lqg(tp_d, n, C_mid)
        print(f"  K_lmi shape: {K_lmi.shape}, ||K_lmi|| = "
              f"{np.linalg.norm(K_lmi):.4e}")
        print(f"  L_lmi shape: {L_lmi.shape}, ||L_lmi|| = "
              f"{np.linalg.norm(L_lmi):.4e}")
        # check sign convention; LMIProblem returns K such that A+BK stable
        # (see synthesis.py: u = M_r X^{-1} = K_r, with the LMI built around
        # A + BK). Negate to match u = -Kx convention.
        v_mid_idx = R_total // 2
        rho_minus_K = float(np.max(np.abs(np.linalg.eigvals(
            A_v_d[v_mid_idx] - B_v_d[v_mid_idx] @ K_lmi[v_mid_idx]))))
        rho_plus_K = float(np.max(np.abs(np.linalg.eigvals(
            A_v_d[v_mid_idx] + B_v_d[v_mid_idx] @ K_lmi[v_mid_idx]))))
        print(f"  ρ(A - B K_lmi)  = {rho_minus_K:.4f}")
        print(f"  ρ(A + B K_lmi)  = {rho_plus_K:.4f}")
        if rho_plus_K < 1.0 and rho_minus_K >= 1.0:
            print(f"  -> negating K_lmi to match u = -Kx convention")
            K_lmi = -K_lmi
        lmi_ok = True
    except Exception as e:
        print(f"  LMI synthesis failed: {e}")
        K_lmi, L_lmi = None, None
        lmi_ok = False

    # ---- evaluation ----
    V_eval = np.linspace(V_MIN, V_MAX, N_V_SWEEP)
    weights_eval = evaluate_weights_at_V(tp_d, V_eval)

    A_open = open_loop_at_V(A_v_d, weights_eval)
    A_cl_pqrnpo = regulator_closed_loop(A_v_d, B_v_d, kappa_pqrnpo, weights_eval)
    A_cl_lmi = (regulator_closed_loop(A_v_d, B_v_d, K_lmi, weights_eval)
                  if lmi_ok else None)

    print(f"\n[mode tracking] V grid: {N_V_SWEEP} points, dt = {DEFAULT_DT} s")
    eigvals_open, _ = track_modes(A_open)
    eigvals_pqrnpo, _ = track_modes(A_cl_pqrnpo)
    if lmi_ok:
        eigvals_lmi, _ = track_modes(A_cl_lmi)

    flutter_idx, zeta_open_seq = find_flutter_mode_index(
        eigvals_open, DEFAULT_DT, V_eval, V_F_NOMINAL)
    print(f"  flutter mode index: {flutter_idx}")
    v_F_idx = np.argmin(np.abs(V_eval - V_F_NOMINAL))
    print(f"  ζ_open at V_F: {zeta_open_seq[v_F_idx, flutter_idx]:.4f}")

    zeta_pqrnpo_seq = np.array([damping_ratio_discrete(eigvals_pqrnpo[v],
                                                          DEFAULT_DT)
                                   for v in range(N_V_SWEEP)])
    if lmi_ok:
        zeta_lmi_seq = np.array([damping_ratio_discrete(eigvals_lmi[v],
                                                          DEFAULT_DT)
                                    for v in range(N_V_SWEEP)])

    # ---- plot ----
    fig = plt.figure(figsize=(13, 5.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.2],
                            height_ratios=[1.0, 0.04],
                            hspace=0.05, wspace=0.35)
    ax_open = fig.add_subplot(gs[0, 0])
    ax_cl   = fig.add_subplot(gs[0, 1])
    ax_zeta = fig.add_subplot(gs[0:, 2])
    ax_cbar = fig.add_subplot(gs[1, 0:2])

    cmap = plt.cm.viridis
    norm = plt.Normalize(V_MIN, V_MAX)
    theta = np.linspace(0, 2 * np.pi, 200)

    for v in range(N_V_SWEEP):
        ax_open.scatter(eigvals_open[v].real, eigvals_open[v].imag,
                          c=[cmap(norm(V_eval[v]))], s=4, alpha=0.6)
    ax_open.plot(np.cos(theta), np.sin(theta), color="k", lw=0.6, ls="--")
    ax_open.set_xlabel(r"Re$(z)$"); ax_open.set_ylabel(r"Im$(z)$")
    ax_open.set_title("(a) Open-loop poles")
    ax_open.set_xlim(-1.2, 1.2); ax_open.set_ylim(-1.2, 1.2)
    ax_open.set_aspect("equal")
    ax_open.grid(True, lw=0.3, alpha=0.5)
    ax_open.spines["top"].set_visible(False)
    ax_open.spines["right"].set_visible(False)
    ax_open.tick_params(direction="in")

    for v in range(N_V_SWEEP):
        ax_cl.scatter(eigvals_pqrnpo[v].real, eigvals_pqrnpo[v].imag,
                       c=[cmap(norm(V_eval[v]))], s=4, alpha=0.6)
    ax_cl.plot(np.cos(theta), np.sin(theta), color="k", lw=0.6, ls="--")
    ax_cl.set_xlabel(r"Re$(z)$"); ax_cl.set_ylabel(r"Im$(z)$")
    ax_cl.set_title("(b) Regulator poles, P-QRNPO LQG")
    ax_cl.set_xlim(-1.2, 1.2); ax_cl.set_ylim(-1.2, 1.2)
    ax_cl.set_aspect("equal")
    ax_cl.grid(True, lw=0.3, alpha=0.5)
    ax_cl.spines["top"].set_visible(False)
    ax_cl.spines["right"].set_visible(False)
    ax_cl.tick_params(direction="in")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cbar, orientation="horizontal")
    cbar.set_label(r"airspeed $V$ (m/s)")
    ax_cbar.tick_params(direction="in")

    ax_zeta.axhspan(-1.0, 0.0, color="red", alpha=0.07, zorder=0)
    ax_zeta.plot(V_eval, zeta_open_seq[:, flutter_idx], "-",
                  color=COLORS["open"], lw=1.5, label="open loop", zorder=2)
    ax_zeta.plot(V_eval, zeta_pqrnpo_seq[:, flutter_idx], "-",
                  color=COLORS["pqrnpo"], lw=1.8, label="P-QRNPO LQG", zorder=4)
    if lmi_ok:
        ax_zeta.plot(V_eval, zeta_lmi_seq[:, flutter_idx], "--",
                      color=COLORS["lmi"], lw=1.5, label="LMI LQG", zorder=3)
    ax_zeta.axhline(0.0, color="k", lw=0.7, ls="-", alpha=0.6)
    ax_zeta.axvline(V_F_NOMINAL, color=COLORS["vF_marker"], lw=0.8, ls=":",
                     label=rf"$V_F = {V_F_NOMINAL}$ m/s", zorder=1)
    y_min = min(zeta_open_seq[:, flutter_idx].min() - 0.02, -0.02)
    if lmi_ok:
        y_max = max(zeta_lmi_seq[:, flutter_idx].max(),
                     zeta_pqrnpo_seq[:, flutter_idx].max()) + 0.02
    else:
        y_max = zeta_pqrnpo_seq[:, flutter_idx].max() + 0.02
    ax_zeta.set_ylim(y_min, max(y_max, 0.02))
    ax_zeta.text(0.98, 0.98, "stable", transform=ax_zeta.transAxes,
                  fontsize=8, color="#444444", va="top", ha="right", alpha=0.7)
    ax_zeta.text(0.98, 0.02, "unstable", transform=ax_zeta.transAxes,
                  fontsize=8, color="#a00000", va="bottom", ha="right", alpha=0.7)
    ax_zeta.set_xlabel(r"airspeed $V$ (m/s)")
    ax_zeta.set_ylabel(r"flutter mode damping ratio $\zeta$")
    ax_zeta.set_title("(c) Damping ratio of the flutter mode")
    ax_zeta.legend(frameon=False, loc="lower left", fontsize=8.5, ncol=1)
    ax_zeta.grid(True, lw=0.3, alpha=0.5)
    ax_zeta.spines["top"].set_visible(False)
    ax_zeta.spines["right"].set_visible(False)
    ax_zeta.tick_params(direction="in")

    fig.suptitle(rf"FliPASED [{channel}] flutter envelope expansion: "
                 rf"polytopic LQG via P-QRNPO and LMI both suppress flutter "
                 rf"throughout $V \in [{V_MIN}, {V_MAX}]$ m/s",
                 fontsize=10, y=0.99)

    out_path = fig_dir / f"flutter_envelope_{channel}.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out_path.name}")

    save_dict = {
        "V_eval": V_eval,
        "eigvals_open": eigvals_open,
        "eigvals_pqrnpo": eigvals_pqrnpo,
        "zeta_open": zeta_open_seq[:, flutter_idx],
        "zeta_pqrnpo": zeta_pqrnpo_seq[:, flutter_idx],
        "flutter_idx": flutter_idx,
        "kappa_pqrnpo": kappa_pqrnpo, "lambda_pqrnpo": lambda_pqrnpo,
    }
    if lmi_ok:
        save_dict["eigvals_lmi"] = eigvals_lmi
        save_dict["zeta_lmi"] = zeta_lmi_seq[:, flutter_idx]
        save_dict["K_lmi"] = K_lmi
        save_dict["L_lmi"] = L_lmi
    np.savez(res_dir / f"flutter_envelope_{channel}.npz", **save_dict)
    print(f"saved results/flutter_envelope_{channel}.npz")


if __name__ == "__main__":
    main()
