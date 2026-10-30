"""
§6.4 Initial-condition transient response.

Honest signals-and-systems test of LQG closed-loop performance:
  - Perturb the wing in the flutter-mode direction at t=0
  - Observer starts cold (x̂_0 = 0)
  - Run closed-loop simulation with process and measurement noise
  - Compare three conditions: open-loop, P-QRNPO LQG, LMI LQG
  - Three airspeeds: V = 45 m/s (subcritical), V_F = 51.42 (marginal),
                     V = 58 m/s (supercritical)

This experiment does NOT use a gust input channel (which the public FliPASED
archive does not expose). Instead, the perturbation is introduced via x_0 in
the dangerous flutter-mode direction, then the controller's job is to restore
trim. This tests:
  - Transient regulation under cold-observer conditions
  - Transient response in the worst eigenvector direction
  - Behavior across the airspeed envelope including past V_F

Layout: 2 rows × 3 cols
  Row 1: y(t) (IMU rotation rate, sym channel) at each V
  Row 2: u(t) (actuator deflection) at each V

Run from project root:
    python -m examples.lqg_flipased_demo.experiment_pqrnpo_initial_condition
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


# Airspeeds
V_F_NOMINAL = 51.42
V_TEST = [45.0, V_F_NOMINAL, 58.0]
V_LABELS = [r"$V = 45$ m/s (subcritical)",
              r"$V = V_F = 51.42$ m/s (marginal)",
              r"$V = 58$ m/s (supercritical)"]

# Sim
T_FINAL = 1.0          # seconds
PERTURB_AMPLITUDE = 1.0  # initial perturbation magnitude in flutter direction
RNG_SEED = 42

# Synthesis settings
HORIZON_FOR_REF = 200
INIT_PERTURB = 0.3
MAX_ITER_PQRNPO = 15

# LMI
LMI_REG_MARGIN = 0.05
LMI_OBS_MARGIN = 0.05
LMI_UMAX = 0.35
LMI_PHI = 0.05

# Bang Wong colorblind-safe palette
COLORS = {
    "open":      "#999999",
    "pqrnpo":    "#0072B2",
    "lmi":       "#D55E00",
    "VF_marker": "#CC79A7",
}


# --------------------------------------------------------------------------
# Synthesis helpers (same as §6.3)
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


def synthesize_pqrnpo_lqg(tp_d, n, p, Q, R, W_proc, V_meas, Sigma_0,
                            A_grid, B_grid, alpha, weights, C_mid,
                            horizon_seed, init_perturb, max_iter):
    kappa_seed, _ = reference_optimum(tp_d, n, Q, R, Sigma_0, horizon_seed)
    kappa_init = (1.0 - init_perturb) * kappa_seed
    kappa, _ = p_qrnpo(kappa_init, A_grid, B_grid, alpha, weights, Q, R,
                          Sigma_0, max_iter=max_iter, verbose=False)
    obs = ObserverPRHPGProblem(tp=tp_d, C=C_mid, n=n)
    obs.set_noise_covariances(W=W_proc, V=V_meas)
    obs.set_horizon(N=horizon_seed)
    lam, _ = obs.solve()
    return kappa, lam


def synthesize_lmi_lqg(tp_d, n, C_mid, A_v_d, B_v_d, R_total):
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

    # sign convention check (LMI returns K such that A+BK stable; we want u=-Kx)
    v_mid_idx = R_total // 2
    rho_minus = float(np.max(np.abs(np.linalg.eigvals(
        A_v_d[v_mid_idx] - B_v_d[v_mid_idx] @ K_lmi[v_mid_idx]))))
    rho_plus = float(np.max(np.abs(np.linalg.eigvals(
        A_v_d[v_mid_idx] + B_v_d[v_mid_idx] @ K_lmi[v_mid_idx]))))
    if rho_plus < 1.0 and rho_minus >= 1.0:
        K_lmi = -K_lmi
    return K_lmi, L_lmi


# --------------------------------------------------------------------------
# Find flutter eigenvector at given V
# --------------------------------------------------------------------------


def flutter_mode_at_V(A_v, weights_at_V, V_F_guess_idx=None):
    """Return the (real) eigenvector of the most unstable / least-damped mode
    of A(V) at every V in the grid.

    Returns:
      v_flutter : (n_V, n) real vector (largest unstable real direction)
    """
    n_V = weights_at_V.shape[0]
    n = A_v.shape[1]
    A_grid = np.einsum("vi,iab->vab", weights_at_V, A_v)
    v_flutter = np.zeros((n_V, n))
    for k in range(n_V):
        w, V = np.linalg.eig(A_grid[k])
        # pick mode with largest |w| if all stable, or with |w| > 1 if unstable
        idx = int(np.argmax(np.abs(w)))
        v = V[:, idx]
        # take real part; if complex pair, the real part of either eigenvector is fine
        v_real = np.real(v)
        v_real = v_real / (np.linalg.norm(v_real) + 1e-14)
        v_flutter[k] = v_real
    return v_flutter, A_grid


# --------------------------------------------------------------------------
# LQG closed-loop simulation
# --------------------------------------------------------------------------


def simulate_lqg(A_grid_t, B_grid_t, C, K_grid_t, L_grid_t,
                   x0, xhat0, T_steps, W_proc, V_meas, rng):
    """Simulate discrete-time LQG closed loop with cold observer.

    A_grid_t : (T, n, n) plant A matrix at each timestep (V is constant here
               since we're at fixed airspeed, but kept general)
    B_grid_t : (T, n, m)
    K_grid_t : (T, m, n)  regulator gain (u = -K x̂)
    L_grid_t : (T, n, p)  observer gain (x̂_+ = A x̂ + B u + L (y - C x̂))
    x0       : (n,)       true initial state
    xhat0    : (n,)       initial estimate
    """
    n = A_grid_t.shape[1]
    m = B_grid_t.shape[2]
    p = C.shape[0]

    x = np.zeros((T_steps + 1, n))
    xhat = np.zeros((T_steps + 1, n))
    u = np.zeros((T_steps, m))
    y = np.zeros((T_steps, p))
    x[0] = x0
    xhat[0] = xhat0

    sqrt_W = np.linalg.cholesky(W_proc + 1e-12 * np.eye(n))
    sqrt_V = np.linalg.cholesky(V_meas + 1e-12 * np.eye(p))

    for t in range(T_steps):
        # measurement
        v_meas = sqrt_V @ rng.standard_normal(p)
        y[t] = C @ x[t] + v_meas
        # control
        u[t] = -K_grid_t[t] @ xhat[t]
        # plant update
        w_proc = sqrt_W @ rng.standard_normal(n)
        x[t + 1] = A_grid_t[t] @ x[t] + B_grid_t[t] @ u[t] + w_proc
        # observer update (innovation form)
        innov = y[t] - C @ xhat[t]
        xhat[t + 1] = A_grid_t[t] @ xhat[t] + B_grid_t[t] @ u[t] + L_grid_t[t] @ innov

    return x, xhat, u, y


def simulate_open_loop(A_grid_t, B_grid_t, x0, T_steps, W_proc, rng):
    """Simulate open-loop with no control, just process noise."""
    n = A_grid_t.shape[1]
    m = B_grid_t.shape[2]
    x = np.zeros((T_steps + 1, n))
    u = np.zeros((T_steps, m))
    x[0] = x0
    sqrt_W = np.linalg.cholesky(W_proc + 1e-12 * np.eye(n))
    for t in range(T_steps):
        w_proc = sqrt_W @ rng.standard_normal(n)
        x[t + 1] = A_grid_t[t] @ x[t] + w_proc   # u=0
    return x, u


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
    hull = "cno"

    print("=" * 72)
    print(f"§6.4 Initial-condition transient response (channel={channel})")
    print("=" * 72)

    # ---- discrete-time TP form ----
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

    # ---- synthesize ----
    print(f"\n[synth] P-QRNPO LQG ...")
    kappa_pqrnpo, lambda_pqrnpo = synthesize_pqrnpo_lqg(
        tp_d, n, p, Q, R, W_proc, V_meas, Sigma_0,
        A_grid_synth, B_grid_synth, alpha_synth, weights_synth, C_mid,
        HORIZON_FOR_REF, INIT_PERTURB, MAX_ITER_PQRNPO)
    print(f"  ||K_P-QRNPO|| = {np.linalg.norm(kappa_pqrnpo):.4e}")
    print(f"  ||L_P-QRNPO|| = {np.linalg.norm(lambda_pqrnpo):.4e}")

    print(f"\n[synth] LMI LQG ...")
    K_lmi, L_lmi = synthesize_lmi_lqg(tp_d, n, C_mid, A_v_d, B_v_d, R_total)
    print(f"  ||K_LMI||     = {np.linalg.norm(K_lmi):.4e}")
    print(f"  ||L_LMI||     = {np.linalg.norm(L_lmi):.4e}")

    # ---- evaluate at the test airspeeds ----
    weights_test = evaluate_weights_at_V(tp_d, np.array(V_TEST))
    # plant at each V
    A_test = np.einsum("vi,iab->vab", weights_test, A_v_d)
    B_test = np.einsum("vi,iab->vab", weights_test, B_v_d)
    # gains at each V
    K_pqrnpo_test = np.einsum("vi,iab->vab", weights_test, kappa_pqrnpo)
    L_pqrnpo_test = np.einsum("vi,iab->vab", weights_test, lambda_pqrnpo)
    K_lmi_test = np.einsum("vi,iab->vab", weights_test, K_lmi)
    L_lmi_test = np.einsum("vi,iab->vab", weights_test, L_lmi)

    # ---- find flutter direction at each V (= eigenvector of largest |eig| of A(V)) ----
    flutter_dirs = np.zeros((len(V_TEST), n))
    for v_idx in range(len(V_TEST)):
        w, V_eigs = np.linalg.eig(A_test[v_idx])
        worst_idx = int(np.argmax(np.abs(w)))
        v_real = np.real(V_eigs[:, worst_idx])
        flutter_dirs[v_idx] = v_real / (np.linalg.norm(v_real) + 1e-14)
        print(f"  V = {V_TEST[v_idx]:.2f} m/s: largest |λ| = {np.abs(w[worst_idx]):.4f}")

    # ---- simulate ----
    T_steps = int(T_FINAL / DEFAULT_DT)
    t_grid = np.arange(T_steps + 1) * DEFAULT_DT
    rng = np.random.default_rng(RNG_SEED)

    sim_data = {}  # sim_data[V_idx] = {'open': (x, u), 'pqrnpo': ..., 'lmi': ...}
    for v_idx, V_val in enumerate(V_TEST):
        print(f"\n[sim] V = {V_val} m/s ...")
        x0 = PERTURB_AMPLITUDE * flutter_dirs[v_idx]
        xhat0 = np.zeros(n)

        # tile the time-invariant gains across timesteps (constant V)
        A_t = np.tile(A_test[v_idx], (T_steps, 1, 1))
        B_t = np.tile(B_test[v_idx], (T_steps, 1, 1))

        # use a fresh RNG per V to get same noise sequence across controllers
        rng_open = np.random.default_rng(RNG_SEED + v_idx)
        rng_pq   = np.random.default_rng(RNG_SEED + v_idx)
        rng_lmi  = np.random.default_rng(RNG_SEED + v_idx)

        x_open, u_open = simulate_open_loop(A_t, B_t, x0, T_steps, W_proc, rng_open)

        K_pq_t = np.tile(K_pqrnpo_test[v_idx], (T_steps, 1, 1))
        L_pq_t = np.tile(L_pqrnpo_test[v_idx], (T_steps, 1, 1))
        x_pq, _, u_pq, _ = simulate_lqg(A_t, B_t, C_mid, K_pq_t, L_pq_t,
                                          x0, xhat0, T_steps, W_proc, V_meas, rng_pq)

        K_lm_t = np.tile(K_lmi_test[v_idx], (T_steps, 1, 1))
        L_lm_t = np.tile(L_lmi_test[v_idx], (T_steps, 1, 1))
        x_lm, _, u_lm, _ = simulate_lqg(A_t, B_t, C_mid, K_lm_t, L_lm_t,
                                          x0, xhat0, T_steps, W_proc, V_meas, rng_lmi)

        # output: y_sym (first row of C)
        y_open = (C_mid @ x_open.T).T   # (T+1, p)
        y_pq   = (C_mid @ x_pq.T).T
        y_lm   = (C_mid @ x_lm.T).T

        sim_data[v_idx] = {
            "open": (y_open[:, 0], u_open),
            "pqrnpo": (y_pq[:, 0], u_pq),
            "lmi": (y_lm[:, 0], u_lm),
        }

        # diagnostics
        print(f"  open-loop ||y_sym||_inf = {np.max(np.abs(y_open[:, 0])):.4e}")
        print(f"  P-QRNPO   ||y_sym||_inf = {np.max(np.abs(y_pq[:, 0])):.4e}, "
              f"||u||_inf = {np.max(np.abs(u_pq)):.4e}")
        print(f"  LMI       ||y_sym||_inf = {np.max(np.abs(y_lm[:, 0])):.4e}, "
              f"||u||_inf = {np.max(np.abs(u_lm)):.4e}")

    # ---- plot ----
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5),
                                gridspec_kw={"hspace": 0.35, "wspace": 0.25})

    for v_idx, V_val in enumerate(V_TEST):
        ax_y = axes[0, v_idx]
        ax_u = axes[1, v_idx]

        y_open, u_open = sim_data[v_idx]["open"]
        y_pq, u_pq     = sim_data[v_idx]["pqrnpo"]
        y_lm, u_lm     = sim_data[v_idx]["lmi"]

        # output panel
        ax_y.plot(t_grid, y_open, "-", color=COLORS["open"], lw=1.2,
                    label="open loop", alpha=0.9)
        ax_y.plot(t_grid, y_pq, "-", color=COLORS["pqrnpo"], lw=1.5,
                    label="P-QRNPO LQG")
        ax_y.plot(t_grid, y_lm, "--", color=COLORS["lmi"], lw=1.3,
                    label="LMI LQG")
        ax_y.axhline(0, color="k", lw=0.5, alpha=0.4)
        ax_y.set_title(V_LABELS[v_idx], fontsize=10)
        if v_idx == 0:
            ax_y.set_ylabel(r"output $y_\mathrm{sym}$ (rad/s)")
        ax_y.grid(True, lw=0.3, alpha=0.5)
        ax_y.spines["top"].set_visible(False)
        ax_y.spines["right"].set_visible(False)
        ax_y.tick_params(direction="in")
        # auto y-limit; for unstable open loop, limit so closed-loop curves are visible
        if v_idx == 2:  # supercritical; open loop diverges
            cl_max = max(np.max(np.abs(y_pq)), np.max(np.abs(y_lm)))
            ax_y.set_ylim(-3 * cl_max, 3 * cl_max)
        if v_idx == 0:
            ax_y.legend(frameon=False, loc="upper right", fontsize=8.5)

        # control panel
        ax_u.plot(t_grid[:-1], u_pq[:, 0], "-", color=COLORS["pqrnpo"], lw=1.5,
                    label="P-QRNPO LQG")
        ax_u.plot(t_grid[:-1], u_lm[:, 0], "--", color=COLORS["lmi"], lw=1.3,
                    label="LMI LQG")
        ax_u.axhline(0, color="k", lw=0.5, alpha=0.4)
        ax_u.set_xlabel(r"time $t$ (s)")
        if v_idx == 0:
            ax_u.set_ylabel(r"control $u$ (rad)")
        ax_u.grid(True, lw=0.3, alpha=0.5)
        ax_u.spines["top"].set_visible(False)
        ax_u.spines["right"].set_visible(False)
        ax_u.tick_params(direction="in")

    fig.suptitle(rf"FliPASED [{channel}] initial-condition transient response: "
                  rf"perturbation in flutter-mode direction at $t = 0$, "
                  rf"cold observer ($\hat{{x}}_0 = 0$)",
                  fontsize=10, y=0.99)

    out_path = fig_dir / f"initial_condition_{channel}.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out_path.name}")

    # ---- save data ----
    save_dict = {"V_TEST": np.array(V_TEST), "t_grid": t_grid,
                  "kappa_pqrnpo": kappa_pqrnpo, "lambda_pqrnpo": lambda_pqrnpo,
                  "K_lmi": K_lmi, "L_lmi": L_lmi}
    for v_idx in range(len(V_TEST)):
        for ctrl in ("open", "pqrnpo", "lmi"):
            y_arr, u_arr = sim_data[v_idx][ctrl]
            save_dict[f"y_{ctrl}_V{v_idx}"] = y_arr
            save_dict[f"u_{ctrl}_V{v_idx}"] = u_arr
    np.savez(res_dir / f"initial_condition_{channel}.npz", **save_dict)
    print(f"saved results/initial_condition_{channel}.npz")


if __name__ == "__main__":
    main()
