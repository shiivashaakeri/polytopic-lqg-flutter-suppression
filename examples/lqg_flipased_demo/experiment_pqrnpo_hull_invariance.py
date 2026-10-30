"""
§6.3.5 Hull invariance of polytopic LPV LQG synthesis.

Sweeps over 5 polytopic hulls, 3 pairing sets, 2 methods, and reports
5 validation metrics per (hull, set, method).

  Hulls  : snnn, close, ino, irno, cno
  Sets   :
    A — K-only sweep: synthesize K at each hull, hold L fixed at CNO
    B — L-only sweep: synthesize L at each hull, hold K fixed at CNO
    C — both swept:   synthesize K and L at each hull
  Methods: P-QRNPO LQG (primal Newton + dual via Kalman duality)
           LMI LQG    (Tanaka-Wang stability-only, MOSEK)

  Metrics (per cell):
    1. feasibility (binary, only LMI can fail)
    2. LQG cost  J = ∫ tr(P_V Σ_0) dμ
    3. gain norms ||K||_F and ||L||_F
    4. transient peaks ||y||_inf and ||u||_inf at V_F (cold observer)
    5. wall-clock synthesis time

Output:
  6 figures: hull_invariance_<set>_<method>.png in paper_figures/
  1 results .npz with everything: hull_invariance_<channel>.npz

Run from project root:
    python -m examples.lqg_flipased_demo.experiment_pqrnpo_hull_invariance
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from examples.lqg_flipased_demo.dynamics import (
    discretize_polytope, flipased_polytope_reduced, flipased_tpform,
)
from examples.lqg_flipased_demo.experiment_p_qrnpo import (
    cost_and_gradient, p_qrnpo, reference_optimum,
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


HULLS = ["snnn", "close", "ino", "irno", "cno"]
HULL_LABELS = ["SNNN", "CHull", "INO", "IRNO", "CNO"]
REFERENCE_HULL = "cno"   # baseline used for "fixed K" / "fixed L" in sets A and B

V_F_NOMINAL = 51.42
T_FINAL = 1.0
PERTURB_AMP = 1.0
RNG_SEED = 42

# synthesis settings
HORIZON_FOR_REF = 200
INIT_PERTURB = 0.3
MAX_ITER_PQRNPO = 15

# LMI (stability only, input constraint infeasible)
# Two margin settings: strict (matches §6.3) and loose (more feasibility)
LMI_REG_MARGIN_STRICT = 0.05
LMI_OBS_MARGIN_STRICT = 0.05
LMI_REG_MARGIN_LOOSE  = 0.001
LMI_OBS_MARGIN_LOOSE  = 0.001

COLORS = {
    "pqrnpo":      "#0072B2",
    "lmi_strict":  "#D55E00",
    "lmi_loose":   "#E69F00",
    "fail":        "#999999",
    "K":           "#0072B2",
    "L":           "#E69F00",
    "y":           "#009E73",
    "u":           "#D55E00",
}


# =========================================================================
# Synthesis primitives
# =========================================================================


def synthesize_pqrnpo_K(tp_d, n, Q, R, Sigma_0, A_grid, B_grid, alpha, weights,
                          horizon_seed, init_perturb, max_iter):
    """Primal P-QRNPO Newton."""
    kappa_seed, _ = reference_optimum(tp_d, n, Q, R, Sigma_0, horizon_seed)
    kappa_init = (1.0 - init_perturb) * kappa_seed
    kappa, _ = p_qrnpo(kappa_init, A_grid, B_grid, alpha, weights, Q, R,
                          Sigma_0, max_iter=max_iter, verbose=False)
    return kappa


def synthesize_pqrnpo_L(tp_d, n, p, W_proc, V_meas, C_mid, horizon_seed):
    """Dual P-QRNPO via ObserverPRHPGProblem at large horizon."""
    obs = ObserverPRHPGProblem(tp=tp_d, C=C_mid, n=n)
    obs.set_noise_covariances(W=W_proc, V=V_meas)
    obs.set_horizon(N=horizon_seed)
    lam, _ = obs.solve()
    return lam


def synthesize_lmi_K(tp_d, n, A_v_d, B_v_d, R_total, margin):
    """Tanaka-Wang stability-only LMI regulator (MOSEK)."""
    reg = LMIProblem(tp_d.S, n=n, discrete=True)
    reg.add_stability(margin=margin)
    K_lmi, _ = reg.solve(solver="MOSEK", verbose=False)
    if K_lmi.ndim > 3:
        K_lmi = K_lmi.reshape(-1, K_lmi.shape[-2], K_lmi.shape[-1])
    # sign convention check
    v_mid = R_total // 2
    rho_minus = float(np.max(np.abs(np.linalg.eigvals(
        A_v_d[v_mid] - B_v_d[v_mid] @ K_lmi[v_mid]))))
    rho_plus = float(np.max(np.abs(np.linalg.eigvals(
        A_v_d[v_mid] + B_v_d[v_mid] @ K_lmi[v_mid]))))
    if rho_plus < 1.0 and rho_minus >= 1.0:
        K_lmi = -K_lmi
    return K_lmi


def synthesize_lmi_L(tp_d, n, C_mid, margin):
    """Tanaka-Wang stability-only LMI observer (MOSEK)."""
    obs = LMIObserverProblem(tp_d.S, C=C_mid, n=n, discrete=True)
    obs.add_stability(margin=margin)
    L_lmi, _ = obs.solve(solver="MOSEK", verbose=False)
    if L_lmi.ndim > 3:
        L_lmi = L_lmi.reshape(-1, L_lmi.shape[-2], L_lmi.shape[-1])
    return L_lmi


# =========================================================================
# Cost evaluation: tr(P_V Σ_0) integrated over V given a regulator gain
# =========================================================================


def lqg_cost(K, A_grid, B_grid, alpha_grid, weights, Q, R, Sigma_0):
    """Compute J = ∫ tr(P_V Σ_0) dμ for a given regulator K on a fixed plant.

    K          : (R̃, m, n) regulator gain in vertex form (u = -K x convention)
    A_grid     : (n_pts, n, n) blended plant A on quadrature grid
    B_grid     : (n_pts, n, m) blended plant B on quadrature grid
    alpha_grid : (n_pts, R̃) polytopic weights at each quadrature point
    weights    : (n_pts,) quadrature weights summing to ~1
    """
    n_pts = A_grid.shape[0]
    # blend K onto the same quadrature grid: K_grid[k] = sum_r alpha[k,r] K[r]
    K_grid = np.einsum("kr,rmn->kmn", alpha_grid, K)
    total = 0.0
    for k in range(n_pts):
        A_cl = A_grid[k] - B_grid[k] @ K_grid[k]
        Q_cl = Q + K_grid[k].T @ R @ K_grid[k]
        rho = max(np.abs(np.linalg.eigvals(A_cl)))
        if rho >= 1.0:
            return np.inf
        try:
            P = solve_discrete_lyapunov(A_cl.T, Q_cl)
        except np.linalg.LinAlgError:
            return np.inf
        total += weights[k] * np.trace(P @ Sigma_0)
    return float(total)


# =========================================================================
# Closed-loop transient simulation (V_F, cold observer, with noise)
# =========================================================================


def transient_peaks(K_at_V, L_at_V, A_at_V, B_at_V, C, n, m, p,
                      W_proc, V_meas, T_steps, rng_seed=42):
    """Run closed-loop simulation at fixed V (V_F), cold observer.
    Initial state = unit vector in direction of largest |eig| of A.

    Returns peak |y_sym| and peak |u| (max over time)."""
    # initial perturbation: worst eigenvector of A at this V
    w, V = np.linalg.eig(A_at_V)
    worst = int(np.argmax(np.abs(w)))
    v_real = np.real(V[:, worst])
    v_real = v_real / (np.linalg.norm(v_real) + 1e-14)
    x0 = PERTURB_AMP * v_real
    xhat0 = np.zeros(n)

    rng = np.random.default_rng(rng_seed)
    sqrt_W = np.linalg.cholesky(W_proc + 1e-12 * np.eye(n))
    sqrt_V = np.linalg.cholesky(V_meas + 1e-12 * np.eye(p))

    x = x0.copy()
    xhat = xhat0.copy()
    peak_y = 0.0
    peak_u = 0.0

    for _ in range(T_steps):
        v_meas_n = sqrt_V @ rng.standard_normal(p)
        y = C @ x + v_meas_n
        u = -K_at_V @ xhat
        peak_y = max(peak_y, abs(y[0]))   # y_sym = first row of C
        peak_u = max(peak_u, np.max(np.abs(u)))
        w_proc_n = sqrt_W @ rng.standard_normal(n)
        x = A_at_V @ x + B_at_V @ u + w_proc_n
        innov = y - C @ xhat
        xhat = A_at_V @ xhat + B_at_V @ u + L_at_V @ innov

    return float(peak_y), float(peak_u)


def evaluate_at_V(arr_v, weights_at_V):
    """Polytopic blend of a (R̃, ...) array using weights at one V."""
    return np.einsum("i,iab->ab", weights_at_V[0], arr_v)


def weights_at_VF(tp, V_target):
    """Return polytopic weights at a single V."""
    V_low, V_high = tp.domain[0]
    gridsize = tp.U[0].shape[0]
    V_table = np.linspace(V_low, V_high, gridsize)
    R_tilde = tp.U[0].shape[1]
    out = np.zeros((1, R_tilde))
    for r in range(R_tilde):
        out[0, r] = np.interp(V_target, V_table, tp.U[0][:, r])
    return out


# =========================================================================
# Per-hull synthesis result container
# =========================================================================


def synthesize_one_hull(hull, channel, dt, V_target, n_pqrnpo_horizon,
                          init_perturb, max_iter):
    """Synthesize all 4 gains (P-QRNPO K, P-QRNPO L, LMI K, LMI L) at this hull.
    Records wall-clock time and feasibility flags. Returns a dict with everything
    needed to evaluate any pairing.

    Returns dict with keys:
      tp_d, n, m, p, R_total, A_v_d, B_v_d, A_grid_synth, B_grid_synth,
      weights_synth, weights_VF, A_VF, B_VF, C_mid, kappa_pq, lam_pq,
      K_lmi, L_lmi, ok_pq_K, ok_pq_L, ok_lmi_K, ok_lmi_L,
      t_pq_K, t_pq_L, t_lmi_K, t_lmi_L, Q, R, Sigma_0
    """
    poly_red = flipased_polytope_reduced(channel)
    poly_red_d = discretize_polytope(poly_red, dt=dt)
    n, p = poly_red_d.n, poly_red_d.p
    tp_d = flipased_tpform(poly_red_d, hull=hull, gridsize=DEFAULT_GRIDSIZE,
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

    # plant blended at V_F (used for transient simulation)
    weights_VF = weights_at_VF(tp_d, V_target)
    A_VF = evaluate_at_V(A_v_d, weights_VF)
    B_VF = evaluate_at_V(B_v_d, weights_VF)

    # ---- P-QRNPO K ----
    t0 = time.time()
    kappa_pq, ok_pq_K, t_pq_K = None, False, None
    try:
        kappa_pq = synthesize_pqrnpo_K(tp_d, n, Q, R, Sigma_0,
                                          A_grid_synth, B_grid_synth,
                                          alpha_synth, weights_synth,
                                          n_pqrnpo_horizon, init_perturb,
                                          max_iter)
        ok_pq_K = True
    except Exception as e:
        print(f"    P-QRNPO K failed: {e}")
    t_pq_K = time.time() - t0

    # ---- P-QRNPO L ----
    t0 = time.time()
    lam_pq, ok_pq_L, t_pq_L = None, False, None
    try:
        lam_pq = synthesize_pqrnpo_L(tp_d, n, p, W_proc, V_meas, C_mid,
                                        n_pqrnpo_horizon)
        ok_pq_L = True
    except Exception as e:
        print(f"    P-QRNPO L failed: {e}")
    t_pq_L = time.time() - t0

    # ---- LMI K (strict) ----
    t0 = time.time()
    K_lmi_s, ok_lmi_K_s, t_lmi_K_s = None, False, None
    try:
        K_lmi_s = synthesize_lmi_K(tp_d, n, A_v_d, B_v_d, R_total,
                                       LMI_REG_MARGIN_STRICT)
        ok_lmi_K_s = True
    except Exception as e:
        print(f"    LMI K (strict) failed: {e}")
    t_lmi_K_s = time.time() - t0

    # ---- LMI L (strict) ----
    t0 = time.time()
    L_lmi_s, ok_lmi_L_s, t_lmi_L_s = None, False, None
    try:
        L_lmi_s = synthesize_lmi_L(tp_d, n, C_mid, LMI_OBS_MARGIN_STRICT)
        ok_lmi_L_s = True
    except Exception as e:
        print(f"    LMI L (strict) failed: {e}")
    t_lmi_L_s = time.time() - t0

    # ---- LMI K (loose) ----
    t0 = time.time()
    K_lmi_l, ok_lmi_K_l, t_lmi_K_l = None, False, None
    try:
        K_lmi_l = synthesize_lmi_K(tp_d, n, A_v_d, B_v_d, R_total,
                                       LMI_REG_MARGIN_LOOSE)
        ok_lmi_K_l = True
    except Exception as e:
        print(f"    LMI K (loose) failed: {e}")
    t_lmi_K_l = time.time() - t0

    # ---- LMI L (loose) ----
    t0 = time.time()
    L_lmi_l, ok_lmi_L_l, t_lmi_L_l = None, False, None
    try:
        L_lmi_l = synthesize_lmi_L(tp_d, n, C_mid, LMI_OBS_MARGIN_LOOSE)
        ok_lmi_L_l = True
    except Exception as e:
        print(f"    LMI L (loose) failed: {e}")
    t_lmi_L_l = time.time() - t0

    return {
        "hull": hull, "tp_d": tp_d, "n": n, "m": m, "p": p, "R_total": R_total,
        "A_v_d": A_v_d, "B_v_d": B_v_d,
        "A_grid_synth": A_grid_synth, "B_grid_synth": B_grid_synth,
        "weights_synth": weights_synth, "weights_VF": weights_VF,
        "A_VF": A_VF, "B_VF": B_VF, "C_mid": C_mid,
        "kappa_pq": kappa_pq, "lam_pq": lam_pq,
        "K_lmi_s": K_lmi_s, "L_lmi_s": L_lmi_s,
        "K_lmi_l": K_lmi_l, "L_lmi_l": L_lmi_l,
        "ok_pq_K": ok_pq_K, "ok_pq_L": ok_pq_L,
        "ok_lmi_K_s": ok_lmi_K_s, "ok_lmi_L_s": ok_lmi_L_s,
        "ok_lmi_K_l": ok_lmi_K_l, "ok_lmi_L_l": ok_lmi_L_l,
        "t_pq_K": t_pq_K, "t_pq_L": t_pq_L,
        "t_lmi_K_s": t_lmi_K_s, "t_lmi_L_s": t_lmi_L_s,
        "t_lmi_K_l": t_lmi_K_l, "t_lmi_L_l": t_lmi_L_l,
        "Q": Q, "R": R, "W_proc": W_proc, "V_meas": V_meas,
        "Sigma_0": Sigma_0,
    }


# =========================================================================
# Cross-hull deployment helper: evaluate K from hull A on plant of hull B
# =========================================================================


def evaluate_pairing(hull_data_K, hull_data_L, T_steps):
    """Evaluate (K, L) on a *consistent* plant. K and L may come from different
    hulls. The 'true plant' for cost evaluation uses the hull where K was
    synthesized (for cost calc) and at V_F for transient (using K's hull blend).

    For the purpose of validation we deploy on K's hull.
    Returns dict: cost, K_norm, L_norm, peak_y, peak_u, ok
    """
    if hull_data_K is None or hull_data_L is None:
        return {"ok": False}

    K = hull_data_K["_K_to_use"]
    L = hull_data_L["_L_to_use"]
    if K is None or L is None:
        return {"ok": False}

    # use plant from K's hull for cost (the plant K was synthesized for)
    A_grid = hull_data_K["A_grid_synth"]
    B_grid = hull_data_K["B_grid_synth"]
    alpha_grid = _evaluate_weights_on_grid(hull_data_K["tp_d"].U)
    weights = hull_data_K["weights_synth"]
    Q = hull_data_K["Q"]
    R_mat = hull_data_K["R"]
    Sigma_0 = hull_data_K["Sigma_0"]

    # cost
    cost = lqg_cost(K, A_grid, B_grid, alpha_grid, weights, Q, R_mat, Sigma_0)

    # gain norms
    K_norm = float(np.linalg.norm(K))
    L_norm = float(np.linalg.norm(L))

    # transient: deploy K and L at V_F. The plant at V_F comes from K's hull
    # for K, but L was synthesized for L's hull. We blend L using K's hull
    # weights at V_F (this is an approximation — strictly L should be blended
    # using L's hull weights, but they're similar near V_F). Acceptable
    # for the diagnostic transient at one V.
    A_VF = hull_data_K["A_VF"]
    B_VF = hull_data_K["B_VF"]
    C_mid = hull_data_K["C_mid"]

    # blend K at V_F (uses K's hull weights)
    K_at_VF = evaluate_at_V(K, hull_data_K["weights_VF"])

    # blend L at V_F (uses L's hull weights)
    L_at_VF = evaluate_at_V(L, hull_data_L["weights_VF"])

    n = hull_data_K["n"]
    m = hull_data_K["m"]
    p = hull_data_K["p"]
    W_proc = hull_data_K["W_proc"]
    V_meas = hull_data_K["V_meas"]
    try:
        peak_y, peak_u = transient_peaks(K_at_VF, L_at_VF, A_VF, B_VF, C_mid,
                                            n, m, p, W_proc, V_meas, T_steps,
                                            rng_seed=RNG_SEED)
    except Exception as e:
        peak_y, peak_u = np.nan, np.nan

    return {
        "ok": True, "cost": cost, "K_norm": K_norm, "L_norm": L_norm,
        "peak_y": peak_y, "peak_u": peak_u,
    }


# =========================================================================
# Plotting
# =========================================================================


def _vals(results, key):
    return [r[key] if r["ok"] else np.nan for r in results]


def _bare_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in")
    ax.grid(True, lw=0.3, alpha=0.5)


def plot_set_single(set_name, method_name, results, fig_path, color):
    """5 metrics for one (set, method) combination — single curve per panel."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5),
                                gridspec_kw={"hspace": 0.45, "wspace": 0.40})
    x = np.arange(len(HULLS))

    # 1. feasibility
    ax = axes[0, 0]
    feas = [1 if r["ok"] else 0 for r in results]
    bar_colors = [color if f == 1 else COLORS["fail"] for f in feas]
    ax.bar(x, feas, color=bar_colors, edgecolor="k", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(HULL_LABELS, fontsize=9)
    ax.set_ylabel("feasible (1/0)")
    ax.set_title("Feasibility", fontsize=10)
    ax.set_ylim(0, 1.1); _bare_axes(ax)

    # 2. cost
    ax = axes[0, 1]
    cost_vals = _vals(results, "cost")
    ax.plot(x, cost_vals, "o-", color=color, lw=1.5, ms=7,
              markerfacecolor=color, markeredgecolor="k", markeredgewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(HULL_LABELS, fontsize=9)
    ax.set_ylabel(r"LQG cost $J$")
    ax.set_title("LQG cost", fontsize=10); _bare_axes(ax)

    # 3. gain norms
    ax = axes[0, 2]
    K_vals = _vals(results, "K_norm")
    L_vals = _vals(results, "L_norm")
    ax.plot(x, K_vals, "o-", color=COLORS["K"], lw=1.5, ms=7,
              markerfacecolor=COLORS["K"], markeredgecolor="k", markeredgewidth=0.5,
              label=r"$\|K\|_F$")
    ax2 = ax.twinx()
    ax2.plot(x, L_vals, "s--", color=COLORS["L"], lw=1.5, ms=6,
               markerfacecolor=COLORS["L"], markeredgecolor="k", markeredgewidth=0.5,
               label=r"$\|L\|_F$")
    ax.set_xticks(x); ax.set_xticklabels(HULL_LABELS, fontsize=9)
    ax.set_ylabel(r"$\|K\|_F$", color=COLORS["K"])
    ax2.set_ylabel(r"$\|L\|_F$", color=COLORS["L"])
    ax.tick_params(axis="y", colors=COLORS["K"])
    ax2.tick_params(axis="y", colors=COLORS["L"])
    ax.set_title("Gain norms", fontsize=10); _bare_axes(ax)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="best", fontsize=8)

    # 4. peak |y|
    ax = axes[1, 0]
    y_vals = _vals(results, "peak_y")
    ax.plot(x, y_vals, "o-", color=COLORS["y"], lw=1.5, ms=7,
              markerfacecolor=COLORS["y"], markeredgecolor="k", markeredgewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(HULL_LABELS, fontsize=9)
    ax.set_ylabel(r"$\|y\|_\infty$ at $V_F$")
    ax.set_title("Transient peak |y|", fontsize=10); _bare_axes(ax)

    # 5. peak |u|
    ax = axes[1, 1]
    u_vals = _vals(results, "peak_u")
    ax.plot(x, u_vals, "o-", color=COLORS["u"], lw=1.5, ms=7,
              markerfacecolor=COLORS["u"], markeredgecolor="k", markeredgewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(HULL_LABELS, fontsize=9)
    ax.set_ylabel(r"$\|u\|_\infty$ at $V_F$")
    ax.set_title("Transient peak |u|", fontsize=10); _bare_axes(ax)

    # 6. wall-clock
    ax = axes[1, 2]
    t_vals = [r.get("t", np.nan) for r in results]
    ax.bar(x, t_vals, color=color, edgecolor="k", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(HULL_LABELS, fontsize=9)
    ax.set_ylabel("time (s)")
    ax.set_title("Synthesis wall-clock", fontsize=10); _bare_axes(ax)

    fig.suptitle(rf"{set_name}  —  {method_name} LQG  "
                  rf"(channel = sym, $V_F = 51.4$ m/s)",
                  fontsize=11, y=0.99)
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_set_lmi_overlay(set_name, results_strict, results_loose, fig_path):
    """5 metrics for LMI variants — strict + loose overlaid per panel."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5),
                                gridspec_kw={"hspace": 0.45, "wspace": 0.40})
    x = np.arange(len(HULLS))
    c_s = COLORS["lmi_strict"]
    c_l = COLORS["lmi_loose"]

    # 1. feasibility (grouped bars)
    ax = axes[0, 0]
    feas_s = [1 if r["ok"] else 0 for r in results_strict]
    feas_l = [1 if r["ok"] else 0 for r in results_loose]
    width = 0.35
    bs = [c_s if f == 1 else COLORS["fail"] for f in feas_s]
    bl = [c_l if f == 1 else COLORS["fail"] for f in feas_l]
    ax.bar(x - width/2, feas_s, width, color=bs, edgecolor="k",
             linewidth=0.5, label="strict (m=0.05)")
    ax.bar(x + width/2, feas_l, width, color=bl, edgecolor="k",
             linewidth=0.5, label="loose  (m=0.001)")
    ax.set_xticks(x); ax.set_xticklabels(HULL_LABELS, fontsize=9)
    ax.set_ylabel("feasible (1/0)")
    ax.set_title("Feasibility", fontsize=10)
    ax.set_ylim(0, 1.15); _bare_axes(ax)
    ax.legend(frameon=False, fontsize=7, loc="upper right")

    # 2. cost
    ax = axes[0, 1]
    ax.plot(x, _vals(results_strict, "cost"), "o-", color=c_s, lw=1.5, ms=7,
              markerfacecolor=c_s, markeredgecolor="k", markeredgewidth=0.5,
              label="strict")
    ax.plot(x, _vals(results_loose, "cost"), "s--", color=c_l, lw=1.5, ms=7,
              markerfacecolor=c_l, markeredgecolor="k", markeredgewidth=0.5,
              label="loose")
    ax.set_xticks(x); ax.set_xticklabels(HULL_LABELS, fontsize=9)
    ax.set_ylabel(r"LQG cost $J$")
    ax.set_title("LQG cost", fontsize=10); _bare_axes(ax)
    ax.legend(frameon=False, fontsize=8, loc="best")

    # 3. ||K||, ||L||  (split: top half K, bottom half L? Or just K with strict/loose)
    ax = axes[0, 2]
    ax.plot(x, _vals(results_strict, "K_norm"), "o-", color=c_s, lw=1.5, ms=7,
              markerfacecolor=c_s, markeredgecolor="k", markeredgewidth=0.5,
              label="strict, K")
    ax.plot(x, _vals(results_loose, "K_norm"), "s--", color=c_l, lw=1.5, ms=7,
              markerfacecolor=c_l, markeredgecolor="k", markeredgewidth=0.5,
              label="loose, K")
    ax2 = ax.twinx()
    ax2.plot(x, _vals(results_strict, "L_norm"), "^-", color=c_s, lw=1.0, ms=6,
               markerfacecolor="white", markeredgecolor=c_s, markeredgewidth=1.0,
               label="strict, L", alpha=0.8)
    ax2.plot(x, _vals(results_loose, "L_norm"), "v--", color=c_l, lw=1.0, ms=6,
               markerfacecolor="white", markeredgecolor=c_l, markeredgewidth=1.0,
               label="loose, L", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(HULL_LABELS, fontsize=9)
    ax.set_ylabel(r"$\|K\|_F$")
    ax2.set_ylabel(r"$\|L\|_F$")
    ax.set_title("Gain norms", fontsize=10); _bare_axes(ax)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7, loc="best", ncol=2)

    # 4. peak |y|
    ax = axes[1, 0]
    ax.plot(x, _vals(results_strict, "peak_y"), "o-", color=c_s, lw=1.5, ms=7,
              markerfacecolor=c_s, markeredgecolor="k", markeredgewidth=0.5,
              label="strict")
    ax.plot(x, _vals(results_loose, "peak_y"), "s--", color=c_l, lw=1.5, ms=7,
              markerfacecolor=c_l, markeredgecolor="k", markeredgewidth=0.5,
              label="loose")
    ax.set_xticks(x); ax.set_xticklabels(HULL_LABELS, fontsize=9)
    ax.set_ylabel(r"$\|y\|_\infty$ at $V_F$")
    ax.set_title("Transient peak |y|", fontsize=10); _bare_axes(ax)

    # 5. peak |u|
    ax = axes[1, 1]
    ax.plot(x, _vals(results_strict, "peak_u"), "o-", color=c_s, lw=1.5, ms=7,
              markerfacecolor=c_s, markeredgecolor="k", markeredgewidth=0.5,
              label="strict")
    ax.plot(x, _vals(results_loose, "peak_u"), "s--", color=c_l, lw=1.5, ms=7,
              markerfacecolor=c_l, markeredgecolor="k", markeredgewidth=0.5,
              label="loose")
    ax.set_xticks(x); ax.set_xticklabels(HULL_LABELS, fontsize=9)
    ax.set_ylabel(r"$\|u\|_\infty$ at $V_F$")
    ax.set_title("Transient peak |u|", fontsize=10); _bare_axes(ax)

    # 6. wall-clock
    ax = axes[1, 2]
    t_s = [r.get("t", np.nan) for r in results_strict]
    t_l = [r.get("t", np.nan) for r in results_loose]
    ax.bar(x - width/2, t_s, width, color=c_s, edgecolor="k", linewidth=0.5,
             label="strict")
    ax.bar(x + width/2, t_l, width, color=c_l, edgecolor="k", linewidth=0.5,
             label="loose")
    ax.set_xticks(x); ax.set_xticklabels(HULL_LABELS, fontsize=9)
    ax.set_ylabel("time (s)")
    ax.set_title("Synthesis wall-clock", fontsize=10); _bare_axes(ax)

    fig.suptitle(rf"{set_name}  —  LMI LQG (strict m=0.05 vs loose m=0.001)  "
                  rf"(channel = sym, $V_F = 51.4$ m/s)",
                  fontsize=11, y=0.99)
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# =========================================================================
# Main: synthesize once per hull, then assemble pairings
# =========================================================================


def main():
    out_dir = Path("examples/lqg_flipased_demo")
    fig_dir = out_dir / "paper_figures"
    res_dir = out_dir / "results"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    channel = "sym"
    print("=" * 72)
    print(f"§6.3.5 Hull invariance (channel={channel})")
    print("=" * 72)

    # synthesize all gains at every hull (single pass)
    hull_data = {}
    for hull in HULLS:
        print(f"\n[hull = {hull}] synthesizing all gains ...")
        hull_data[hull] = synthesize_one_hull(hull, channel, DEFAULT_DT,
                                                  V_F_NOMINAL, HORIZON_FOR_REF,
                                                  INIT_PERTURB, MAX_ITER_PQRNPO)
        d = hull_data[hull]
        print(f"  P-QRNPO K       : ok={d['ok_pq_K']}  t={d['t_pq_K']:.2f}s")
        print(f"  P-QRNPO L       : ok={d['ok_pq_L']}  t={d['t_pq_L']:.2f}s")
        print(f"  LMI K (strict)  : ok={d['ok_lmi_K_s']} t={d['t_lmi_K_s']:.2f}s")
        print(f"  LMI L (strict)  : ok={d['ok_lmi_L_s']} t={d['t_lmi_L_s']:.2f}s")
        print(f"  LMI K (loose)   : ok={d['ok_lmi_K_l']} t={d['t_lmi_K_l']:.2f}s")
        print(f"  LMI L (loose)   : ok={d['ok_lmi_L_l']} t={d['t_lmi_L_l']:.2f}s")

    T_steps = int(T_FINAL / DEFAULT_DT)
    ref = hull_data[REFERENCE_HULL]

    # Helper: pull (K, L, K_owner_data, L_owner_data, t, ok) for one
    # (hull, set, method) cell.
    def get_cell(hull, set_kind, method):
        """Return dict with keys: K, L, K_owner, L_owner, t, ok."""
        hd = hull_data[hull]
        # what gain comes from where?
        if method == "P-QRNPO":
            K_h = hd["kappa_pq"];      ok_K_h = hd["ok_pq_K"];    t_K_h = hd["t_pq_K"]
            L_h = hd["lam_pq"];        ok_L_h = hd["ok_pq_L"];    t_L_h = hd["t_pq_L"]
            K_r = ref["kappa_pq"];     ok_K_r = ref["ok_pq_K"]
            L_r = ref["lam_pq"];       ok_L_r = ref["ok_pq_L"]
        elif method == "LMI-strict":
            K_h = hd["K_lmi_s"];       ok_K_h = hd["ok_lmi_K_s"]; t_K_h = hd["t_lmi_K_s"]
            L_h = hd["L_lmi_s"];       ok_L_h = hd["ok_lmi_L_s"]; t_L_h = hd["t_lmi_L_s"]
            K_r = ref["K_lmi_s"];      ok_K_r = ref["ok_lmi_K_s"]
            L_r = ref["L_lmi_s"];      ok_L_r = ref["ok_lmi_L_s"]
        elif method == "LMI-loose":
            K_h = hd["K_lmi_l"];       ok_K_h = hd["ok_lmi_K_l"]; t_K_h = hd["t_lmi_K_l"]
            L_h = hd["L_lmi_l"];       ok_L_h = hd["ok_lmi_L_l"]; t_L_h = hd["t_lmi_L_l"]
            K_r = ref["K_lmi_l"];      ok_K_r = ref["ok_lmi_K_l"]
            L_r = ref["L_lmi_l"];      ok_L_r = ref["ok_lmi_L_l"]
        else:
            raise ValueError(method)

        if set_kind == "A":      # K varies, L fixed at REFERENCE_HULL
            K, K_owner, ok_K, t_total = K_h, hd, ok_K_h, t_K_h
            L, L_owner, ok_L          = L_r, ref, ok_L_r
        elif set_kind == "B":    # L varies, K fixed at REFERENCE_HULL
            K, K_owner, ok_K          = K_r, ref, ok_K_r
            L, L_owner, ok_L, t_total = L_h, hd, ok_L_h, t_L_h
        elif set_kind == "C":    # both vary
            K, K_owner, ok_K = K_h, hd, ok_K_h
            L, L_owner, ok_L = L_h, hd, ok_L_h
            t_total = (t_K_h + t_L_h) if (ok_K_h and ok_L_h) else (t_K_h if ok_K_h else t_L_h)
        else:
            raise ValueError(set_kind)

        ok = ok_K and ok_L and (K is not None) and (L is not None)
        return {"K": K, "L": L, "K_owner": K_owner, "L_owner": L_owner,
                  "t": t_total, "ok": ok}

    # ---- compute all 9 (set, method) cells: 3 sets × 3 methods ----
    # method names: P-QRNPO, LMI-strict, LMI-loose
    METHODS = ["P-QRNPO", "LMI-strict", "LMI-loose"]
    SETS = [("Set A: K-only sweep", "A"),
              ("Set B: L-only sweep", "B"),
              ("Set C: both swept", "C")]

    all_results = {}  # key: (set_kind, method) -> list of per-hull eval dicts
    for set_name, set_kind in SETS:
        for method in METHODS:
            key = (set_kind, method)
            print(f"\n[{set_name} | {method}]")
            results = []
            for hull in HULLS:
                cell = get_cell(hull, set_kind, method)
                if not cell["ok"]:
                    results.append({"ok": False, "t": cell["t"],
                                      "cost": np.nan, "K_norm": np.nan,
                                      "L_norm": np.nan, "peak_y": np.nan,
                                      "peak_u": np.nan})
                    print(f"  hull={hull}: FAIL (synthesis)")
                    continue

                K_owner_rec = dict(cell["K_owner"])
                K_owner_rec["_K_to_use"] = cell["K"]
                L_owner_rec = dict(cell["L_owner"])
                L_owner_rec["_L_to_use"] = cell["L"]

                ev = evaluate_pairing(K_owner_rec, L_owner_rec, T_steps)
                ev["t"] = cell["t"]
                if not ev.get("ok", False):
                    print(f"  hull={hull}: FAIL (evaluation)")
                else:
                    print(f"  hull={hull}: J={ev['cost']:.3e}  "
                          f"||K||={ev['K_norm']:.2f}  ||L||={ev['L_norm']:.2f}  "
                          f"|y|={ev['peak_y']:.2f}  |u|={ev['peak_u']:.2f}  "
                          f"t={cell['t']:.2f}s")
                results.append(ev)
            all_results[key] = results

    # ---- plot 6 figures: 3 sets × 2 method-categories (P-QRNPO, LMI) ----
    # Each LMI figure overlays strict and loose curves.
    for set_name, set_kind in SETS:
        # P-QRNPO solo
        results = all_results[(set_kind, "P-QRNPO")]
        fig_path = fig_dir / f"hull_invariance_set{set_kind}_pqrnpo.png"
        plot_set_single(set_name, "P-QRNPO", results, fig_path,
                          color=COLORS["pqrnpo"])
        print(f"saved {fig_path.name}")

        # LMI strict + loose overlay
        results_s = all_results[(set_kind, "LMI-strict")]
        results_l = all_results[(set_kind, "LMI-loose")]
        fig_path = fig_dir / f"hull_invariance_set{set_kind}_lmi.png"
        plot_set_lmi_overlay(set_name, results_s, results_l, fig_path)
        print(f"saved {fig_path.name}")

    print(f"\nDone. 6 figures in paper_figures/.")


if __name__ == "__main__":
    main()
