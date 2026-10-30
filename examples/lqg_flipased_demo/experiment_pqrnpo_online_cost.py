"""
§6.10 Online cost and sample-rate sensitivity.

Two parts:

Part 1: Synthesis time scaling with horizon N.
  - For both controllers, time the synthesis at N ∈ {50, 100, 200, 400, 800}
  - LMI's wallclock is constant in N (it's an SDP, not horizon-based) but we
    report it on the same axis for comparison
  - Each timing repeats 3× and reports min/median.

Part 2: Sample-rate sensitivity.
  - Sweep dt ∈ {0.5, 1, 2, 5, 10, 20, 50, 100} ms
  - At each dt, re-discretize the plant, re-synthesize both controllers,
    deploy on the §6.6 round-trip V maneuver, measure peak |y|, peak |u|,
    and stability.
  - Find the threshold dt where each method fails.

Output:
  paper_figures/sample_rate_sym.png    (synthesis time + sample-rate two-panel)
  results/sample_rate_sym.npz

Run from project root:
    python -m examples.lqg_flipased_demo.experiment_pqrnpo_online_cost
"""

from __future__ import annotations

from pathlib import Path
import time

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

# Part 1: synthesis time vs horizon
N_HORIZONS = [50, 100, 200, 400, 800]
N_REPEATS = 3

# Part 2: sample-rate sweep
DT_GRID_MS = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

# Synthesis settings
INIT_PERTURB = 0.3
MAX_ITER_PQRNPO = 15
LMI_MARGIN_LOOSE = 0.001

STAB_Y_THRESHOLD = 200.0

RNG_SEED = 42

COLORS = {
    "pqrnpo": "#0072B2",
    "lmi":    "#D55E00",
    "VF":     "#CC79A7",
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
# Synthesis helpers (timing-aware versions)
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


def time_pqrnpo_synthesis(tp_d, n, p, Q, R, W_proc, V_meas, Sigma_0,
                              A_grid, B_grid, alpha, weights, C_mid,
                              horizon):
    """Time a full P-QRNPO regulator + observer synthesis. Returns wallclock seconds."""
    t0 = time.time()
    kappa_seed, _ = reference_optimum(tp_d, n, Q, R, Sigma_0, horizon)
    kappa_init = (1.0 - INIT_PERTURB) * kappa_seed
    kappa, _ = p_qrnpo(kappa_init, A_grid, B_grid, alpha, weights, Q, R,
                          Sigma_0, max_iter=MAX_ITER_PQRNPO, verbose=False)
    obs = ObserverPRHPGProblem(tp=tp_d, C=C_mid, n=n)
    obs.set_noise_covariances(W=W_proc, V=V_meas)
    obs.set_horizon(N=horizon)
    lam, _ = obs.solve()
    return time.time() - t0, kappa, lam


def time_lmi_synthesis(tp_d, n, A_v_d, B_v_d, R_total, C_mid, margin):
    """Time a full LMI regulator + observer synthesis. Returns wallclock seconds."""
    t0 = time.time()
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
    return time.time() - t0, K_lmi, L_lmi


# --------------------------------------------------------------------------
# Simulation for sample-rate sweep
# --------------------------------------------------------------------------


def simulate_lqg_lpv(A_t, B_t, C, K_t, L_t, x0, xhat0, T_steps,
                          W_proc, V_meas, rng):
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
    print(f"§6.10 Online cost & sample-rate (channel={channel}, hull={HULL})")
    print("=" * 72)

    # ======================================================================
    # Part 1: synthesis time vs horizon (at nominal dt = 5 ms)
    # ======================================================================
    print("\n" + "=" * 72)
    print("Part 1: Synthesis time vs horizon N")
    print("=" * 72)

    poly_red = flipased_polytope_reduced(channel)
    poly_red_d_nom = discretize_polytope(poly_red, dt=DEFAULT_DT)
    n = poly_red_d_nom.n
    p = poly_red_d_nom.p
    tp_d_nom = flipased_tpform(poly_red_d_nom, hull=HULL,
                                  gridsize=DEFAULT_GRIDSIZE, verbose=False)

    Q = _default_Q(n, channel)
    R_ctrl = DEFAULT_R_CTRL.copy()
    W_proc = W_PROC * np.eye(n)
    V_meas = V_MEAS * np.eye(p)
    Sigma_0 = np.eye(n)

    A_v_d_nom, B_v_d_nom, R_total, m = _extract_vertices(tp_d_nom.S, n)
    alpha_synth = _evaluate_weights_on_grid(tp_d_nom.U)
    A_grid_synth, B_grid_synth = _evaluate_AB_on_grid(A_v_d_nom, B_v_d_nom,
                                                            alpha_synth)
    weights_synth = _build_quadrature_weights(tp_d_nom.U, tp_d_nom.domain)
    C_mid = poly_red_d_nom.C[poly_red_d_nom.R // 2]
    print(f"\nn={n}, m={m}, p={p}, R̃={R_total}")

    pq_times = np.zeros((len(N_HORIZONS), N_REPEATS))
    lmi_times = np.zeros(N_REPEATS)

    # LMI time is independent of N
    print(f"\n  Timing LMI synthesis ({N_REPEATS} repeats) ...")
    for k in range(N_REPEATS):
        dt_lmi, K_lmi, L_lmi = time_lmi_synthesis(tp_d_nom, n, A_v_d_nom,
                                                         B_v_d_nom, R_total, C_mid,
                                                         LMI_MARGIN_LOOSE)
        lmi_times[k] = dt_lmi
        print(f"    repeat {k+1}: {dt_lmi:.3f}s")
    lmi_med = float(np.median(lmi_times))
    print(f"  LMI median: {lmi_med:.3f}s")

    for j, hor in enumerate(N_HORIZONS):
        print(f"\n  Timing P-QRNPO synthesis at N={hor} ({N_REPEATS} repeats) ...")
        for k in range(N_REPEATS):
            dt_pq, kappa, lam = time_pqrnpo_synthesis(tp_d_nom, n, p, Q, R_ctrl,
                                                            W_proc, V_meas, Sigma_0,
                                                            A_grid_synth, B_grid_synth,
                                                            alpha_synth, weights_synth,
                                                            C_mid, hor)
            pq_times[j, k] = dt_pq
            print(f"    repeat {k+1}: {dt_pq:.3f}s")
        med = float(np.median(pq_times[j]))
        print(f"  N={hor}: median P-QRNPO = {med:.3f}s")

    pq_times_med = np.median(pq_times, axis=1)

    # ======================================================================
    # Part 2: sample-rate sensitivity
    # ======================================================================
    print("\n" + "=" * 72)
    print("Part 2: Sample-rate sensitivity")
    print("=" * 72)

    peak_y_pq_dt = np.full(len(DT_GRID_MS), np.nan)
    peak_u_pq_dt = np.full(len(DT_GRID_MS), np.nan)
    stable_pq_dt = np.zeros(len(DT_GRID_MS), dtype=bool)
    pq_synth_ok = np.zeros(len(DT_GRID_MS), dtype=bool)

    peak_y_lmi_dt = np.full(len(DT_GRID_MS), np.nan)
    peak_u_lmi_dt = np.full(len(DT_GRID_MS), np.nan)
    stable_lmi_dt = np.zeros(len(DT_GRID_MS), dtype=bool)
    lmi_synth_ok = np.zeros(len(DT_GRID_MS), dtype=bool)

    for i, dt_ms in enumerate(DT_GRID_MS):
        dt = dt_ms * 1e-3
        print(f"\n  dt = {dt_ms:5.1f} ms ...")

        # Re-discretize
        try:
            poly_red_d = discretize_polytope(poly_red, dt=dt)
            tp_d = flipased_tpform(poly_red_d, hull=HULL,
                                      gridsize=DEFAULT_GRIDSIZE, verbose=False)
            A_v_d, B_v_d, R_total_local, _ = _extract_vertices(tp_d.S, n)
            alpha_s = _evaluate_weights_on_grid(tp_d.U)
            A_grid_s, B_grid_s = _evaluate_AB_on_grid(A_v_d, B_v_d, alpha_s)
            weights_s = _build_quadrature_weights(tp_d.U, tp_d.domain)
            C_mid_local = poly_red_d.C[poly_red_d.R // 2]
        except Exception as e:
            print(f"    re-discretization failed: {e}")
            continue

        # Synthesize P-QRNPO
        try:
            _, kappa_pq, lambda_pq = time_pqrnpo_synthesis(tp_d, n, p, Q,
                                                                 R_ctrl, W_proc,
                                                                 V_meas, Sigma_0,
                                                                 A_grid_s, B_grid_s,
                                                                 alpha_s, weights_s,
                                                                 C_mid_local, 200)
            pq_synth_ok[i] = True
            print(f"    P-QRNPO synth ok")
        except Exception as e:
            print(f"    P-QRNPO synth FAILED: {e}")

        # Synthesize LMI
        try:
            _, K_lmi_dt, L_lmi_dt = time_lmi_synthesis(tp_d, n, A_v_d, B_v_d,
                                                             R_total_local,
                                                             C_mid_local,
                                                             LMI_MARGIN_LOOSE)
            lmi_synth_ok[i] = True
            print(f"    LMI synth ok")
        except Exception as e:
            print(f"    LMI synth FAILED: {e}")

        # If neither synthesizes, skip
        if not pq_synth_ok[i] and not lmi_synth_ok[i]:
            continue

        # Build maneuver at this dt
        T_steps = int(T_FINAL / dt)
        t_grid_dt = np.arange(T_steps + 1) * dt
        V_t_dt = airspeed_profile(t_grid_dt[:-1])
        weights_t_dt = evaluate_weights_at_V(tp_d, V_t_dt)
        A_t_dt = np.einsum("ti,iab->tab", weights_t_dt, A_v_d)
        B_t_dt = np.einsum("ti,iab->tab", weights_t_dt, B_v_d)

        x0 = np.zeros(n)
        xhat0 = np.zeros(n)

        # P-QRNPO simulation
        if pq_synth_ok[i]:
            K_pq_t_dt = np.einsum("ti,iab->tab", weights_t_dt, kappa_pq)
            L_pq_t_dt = np.einsum("ti,iab->tab", weights_t_dt, lambda_pq)
            rng = np.random.default_rng(RNG_SEED)
            x_pq, _, u_pq = simulate_lqg_lpv(A_t_dt, B_t_dt, C_mid_local,
                                                  K_pq_t_dt, L_pq_t_dt,
                                                  x0, xhat0, T_steps, W_proc,
                                                  V_meas, rng)
            y_pq = (C_mid_local @ x_pq.T).T[:, 0]
            peak_y_pq_dt[i] = float(np.max(np.abs(y_pq)))
            peak_u_pq_dt[i] = float(np.max(np.abs(u_pq)))
            stable_pq_dt[i] = peak_y_pq_dt[i] < STAB_Y_THRESHOLD
            marker = "✓" if stable_pq_dt[i] else "✗"
            print(f"    P-QRNPO {marker} |y|={peak_y_pq_dt[i]:.3f} "
                  f"|u|={peak_u_pq_dt[i]:.3f}")

        # LMI simulation
        if lmi_synth_ok[i]:
            K_lmi_t_dt = np.einsum("ti,iab->tab", weights_t_dt, K_lmi_dt)
            L_lmi_t_dt = np.einsum("ti,iab->tab", weights_t_dt, L_lmi_dt)
            rng = np.random.default_rng(RNG_SEED)
            x_lmi, _, u_lmi = simulate_lqg_lpv(A_t_dt, B_t_dt, C_mid_local,
                                                    K_lmi_t_dt, L_lmi_t_dt,
                                                    x0, xhat0, T_steps, W_proc,
                                                    V_meas, rng)
            y_lmi = (C_mid_local @ x_lmi.T).T[:, 0]
            peak_y_lmi_dt[i] = float(np.max(np.abs(y_lmi)))
            peak_u_lmi_dt[i] = float(np.max(np.abs(u_lmi)))
            stable_lmi_dt[i] = peak_y_lmi_dt[i] < STAB_Y_THRESHOLD
            marker = "✓" if stable_lmi_dt[i] else "✗"
            print(f"    LMI {marker} |y|={peak_y_lmi_dt[i]:.3f} "
                  f"|u|={peak_u_lmi_dt[i]:.3f}")

    # ---- SAVE FIRST ----
    save_dict = {
        "N_HORIZONS": np.array(N_HORIZONS),
        "pq_times": pq_times,
        "pq_times_med": pq_times_med,
        "lmi_times": lmi_times,
        "lmi_med": lmi_med,
        "DT_GRID_MS": np.array(DT_GRID_MS),
        "peak_y_pq_dt": peak_y_pq_dt, "peak_u_pq_dt": peak_u_pq_dt,
        "stable_pq_dt": stable_pq_dt, "pq_synth_ok": pq_synth_ok,
        "peak_y_lmi_dt": peak_y_lmi_dt, "peak_u_lmi_dt": peak_u_lmi_dt,
        "stable_lmi_dt": stable_lmi_dt, "lmi_synth_ok": lmi_synth_ok,
    }
    np.savez(res_dir / f"online_cost_{channel}.npz", **save_dict)
    print(f"\nsaved results/online_cost_{channel}.npz")

    # ---- plot two-panel ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2),
                                gridspec_kw={"wspace": 0.30})

    # Panel (a): synthesis time vs horizon
    ax = axes[0]
    ax.semilogy(N_HORIZONS, pq_times_med, "o-", color=COLORS["pqrnpo"],
                  lw=1.6, ms=6, label="P-QRNPO synthesis")
    ax.axhline(lmi_med, color=COLORS["lmi"], lw=1.4, ls="--",
                 label=rf"LMI synthesis (constant, {lmi_med:.2f}s)")
    ax.set_xlabel(r"horizon $N$ (P-QRNPO only)")
    ax.set_ylabel("wallclock time (s)")
    ax.set_title("(a) Synthesis time vs horizon $N$", fontsize=10)
    ax.legend(frameon=False, loc="best", fontsize=9)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in")

    # Panel (b): peak |y| vs dt (the meaningful sample-rate plot)
    ax = axes[1]
    dt_arr = np.array(DT_GRID_MS)
    # plot stable points and unstable points differently
    mask_pq_stab = stable_pq_dt & pq_synth_ok
    mask_pq_unstab = (~stable_pq_dt) & pq_synth_ok
    mask_lmi_stab = stable_lmi_dt & lmi_synth_ok
    mask_lmi_unstab = (~stable_lmi_dt) & lmi_synth_ok

    if np.any(mask_pq_stab):
        ax.loglog(dt_arr[mask_pq_stab], peak_y_pq_dt[mask_pq_stab],
                    "o-", color=COLORS["pqrnpo"], lw=1.6, ms=6,
                    label="P-QRNPO (stable)")
    if np.any(mask_pq_unstab):
        ax.loglog(dt_arr[mask_pq_unstab], peak_y_pq_dt[mask_pq_unstab],
                    "x", color=COLORS["pqrnpo"], ms=10, mew=2.5,
                    label="P-QRNPO (unstable)")
    if np.any(mask_lmi_stab):
        ax.loglog(dt_arr[mask_lmi_stab], peak_y_lmi_dt[mask_lmi_stab],
                    "s--", color=COLORS["lmi"], lw=1.4, ms=5,
                    label="LMI (stable)")
    if np.any(mask_lmi_unstab):
        ax.loglog(dt_arr[mask_lmi_unstab], peak_y_lmi_dt[mask_lmi_unstab],
                    "x", color=COLORS["lmi"], ms=10, mew=2.5,
                    label="LMI (unstable)")
    ax.axhline(STAB_Y_THRESHOLD, color="k", lw=0.5, ls=":", alpha=0.5)
    ax.text(dt_arr[0] * 1.2, STAB_Y_THRESHOLD * 1.1,
              rf"stability threshold $\|y\|_\infty < {STAB_Y_THRESHOLD:.0f}$",
              fontsize=8, color="#666666")
    ax.set_xlabel("sample time $dt$ (ms)")
    ax.set_ylabel(r"peak $\|y\|_\infty$ (rad/s)")
    ax.set_title("(b) Closed-loop peak output vs sample time", fontsize=10)
    ax.legend(frameon=False, loc="best", fontsize=9, ncol=2)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in")

    fig.suptitle(rf"FliPASED [{channel}, hull = {HULL}] online cost & sample-rate "
                 rf"sensitivity",
                 fontsize=10, y=1.02)

    out_path = fig_dir / f"online_cost_{channel}.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path.name}")

    # ---- summary ----
    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    print("\nPart 1 — Synthesis time scaling:")
    print(f"  LMI:     {lmi_med:.3f}s (constant in N)")
    for j, hor in enumerate(N_HORIZONS):
        print(f"  P-QRNPO at N={hor}: {pq_times_med[j]:.3f}s "
              f"(ratio to LMI: {pq_times_med[j] / lmi_med:.1f}×)")

    print("\nPart 2 — Sample-rate sensitivity:")
    print(f"  {'dt (ms)':>8} | {'P-QRNPO synth/sim':>20} | {'LMI synth/sim':>20}")
    print("  " + "-" * 60)
    for i, dt_ms in enumerate(DT_GRID_MS):
        pq_status = "fail" if not pq_synth_ok[i] else (
            "stable" if stable_pq_dt[i] else "unstable")
        lmi_status = "fail" if not lmi_synth_ok[i] else (
            "stable" if stable_lmi_dt[i] else "unstable")
        pq_y = f"|y|={peak_y_pq_dt[i]:.2f}" if pq_synth_ok[i] else ""
        lmi_y = f"|y|={peak_y_lmi_dt[i]:.2f}" if lmi_synth_ok[i] else ""
        print(f"  {dt_ms:>8.1f} | {pq_status:>10} {pq_y:>9} | "
              f"{lmi_status:>10} {lmi_y:>9}")


if __name__ == "__main__":
    main()
