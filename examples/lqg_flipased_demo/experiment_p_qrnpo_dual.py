"""
Dual Polytopic Quasi-Riemannian Newton PO: polytopic observer gain.

Uses the Kalman-LQ duality (tptool/prhpg/observer.py exploits the same):
  Primal LQR for (A^T, C^T, W, V) with control K_d gives K_d such that
  A^T - C^T K_d is stable. Setting L = K_d^T makes A - LC stable
  (observer error dynamics).

So we transpose the plant, run the primal P-QRNPO from
experiment_p_qrnpo.py, then transpose the result.

Comparison: against the dual P-RHPG (`ObserverPRHPGProblem`) at large
horizon, evaluated against the same infinite-horizon polytopic dual
LQR cost.

Run from project root:
    python -m examples.lqg_flipased_demo.experiment_p_qrnpo_dual
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from examples.lqg_flipased_demo.dynamics import (
    discretize_polytope, flipased_polytope, flipased_polytope_reduced,
    flipased_tpform,
)
from examples.lqg_flipased_demo.experiment_p_qrnpo import (
    DAMPING, HESSIAN_FD_EPS, MAX_ITER,
    cost_and_gradient, is_stabilizing, p_qrnpo,
)
from examples.lqg_flipased_demo.pipeline import DEFAULT_DT, DEFAULT_GRIDSIZE
from tptool.prhpg import ObserverPRHPGProblem
from tptool.prhpg.synthesis import (
    _build_quadrature_weights,
    _evaluate_AB_on_grid,
    _evaluate_weights_on_grid,
    _extract_vertices,
)


CHANNEL = "sym"
HULL = "cno"
HORIZON_FOR_REF = 200

# Noise covariances for the dual problem.
# Use the same defaults as pipeline.py's V_meas_scale=100 to match the
# baseline you've been deploying.
W_PROC = 1.0     # process noise covariance scale, W = W_PROC * I
V_MEAS = 100.0   # measurement noise covariance scale, V = V_MEAS * I

# Initialization: perturbed-from-optimum (open-loop FliPASED is unstable
# at high V, so λ=0 doesn't give A - LC stable in general)
INIT_PERTURB = 0.3
SEED = 0

COLORS = {
    "PO":  "#0072B2",
    "opt": "#D55E00",
    "ref": "#999999",
}


def main():
    out_dir = Path("examples/lqg_flipased_demo")
    fig_dir = out_dir / "figures"
    res_dir = out_dir / "results"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"Dual P-QRNPO sanity test (channel={CHANNEL})")
    print("=" * 72)

    # build TP form
    poly_full = flipased_polytope(CHANNEL)
    poly_red = flipased_polytope_reduced(CHANNEL)
    poly_red_d = discretize_polytope(poly_red, dt=DEFAULT_DT)
    n = poly_red_d.n
    p = poly_red_d.p

    tp = flipased_tpform(poly_red_d, hull=HULL,
                          gridsize=DEFAULT_GRIDSIZE, verbose=True)

    W = W_PROC * np.eye(n)         # process noise covariance ("Q" of dual)
    V = V_MEAS * np.eye(p)         # measurement noise covariance ("R" of dual)
    Sigma_0_dual = np.eye(n)       # dual "initial covariance"

    # ---- extract grid quantities ----
    A_v, B_v, R_total, m = _extract_vertices(tp.S, n)
    alpha = _evaluate_weights_on_grid(tp.U)
    A_grid, B_grid = _evaluate_AB_on_grid(A_v, B_v, alpha)
    weights = _build_quadrature_weights(tp.U, tp.domain)

    # build C(V_q) on the same grid
    # C is parameter-INDEPENDENT in FliPASED at the modal level (same C across V),
    # so C_mid suffices, but to be general we replicate it on the grid.
    C_mid = poly_red_d.C[poly_red_d.R // 2]                     # (p, n)
    C_grid = np.tile(C_mid[None, :, :], (alpha.shape[0], 1, 1))  # (Nq, p, n)

    # ---- dual plant on the grid: (A_d, B_d) = (A^T, C^T), state dim n ----
    A_d_grid = np.transpose(A_grid, (0, 2, 1))                   # (Nq, n, n)
    B_d_grid = np.transpose(C_grid, (0, 2, 1))                   # (Nq, n, p)

    # The dual "input dimension" is p (number of outputs).
    # The dual "Q" is W (process noise), the dual "R" is V (measurement noise).

    print(f"\nn={n}, p={p}, R̃={R_total}, N_q={alpha.shape[0]}")
    print(f"Hessian size: {R_total*p*n} × {R_total*p*n} "
          f"({R_total*p*n} FD calls per Newton iter)")

    # ---- reference: dual P-RHPG at large horizon ----
    print(f"\n[reference] dual P-RHPG at N={HORIZON_FOR_REF} ...")
    obs = ObserverPRHPGProblem(tp=tp, C=C_mid, n=n)
    obs.set_noise_covariances(W=W, V=V)
    obs.set_horizon(N=HORIZON_FOR_REF)
    L_flat, J_L_finite = obs.solve()
    # observer.py returns L_stages[0] of shape (R̃, n, p) where L convention
    # is the deployed observer gain (A - LC stable). The dual P-RHPG
    # internally negates to convert from K_dual to L = -K_dual^T.
    lambda_star = L_flat                                          # (R̃, n, p)
    print(f"  dual P-RHPG J (finite horizon, accumulated): {J_L_finite:.6e}")
    print(f"  ||λ*||_F = {np.linalg.norm(lambda_star):.4e}")

    # convert λ* to the dual-LQR convention (K_d such that A^T - C^T K_d stable)
    # L = K_d^T  =>  K_d = L^T.   But also L returned by observer.py uses the
    # u = -Kx observer convention internally (A - LC), and we want
    # A^T - C^T K_d stable.   A - LC = (A^T - C^T L^T)^T, so K_d = L^T works.
    K_d_star = np.transpose(lambda_star, (0, 2, 1))               # (R̃, p, n)

    # check that K_d_star stabilizes (A_d, B_d) on the grid
    ok, rho = is_stabilizing(K_d_star, A_d_grid, B_d_grid, alpha)
    print(f"  dual-LQR check: max ρ(A_d - B_d K_d) = {rho:.4f}, "
          f"stabilizing = {ok}")

    # evaluate our infinite-horizon dual LQR cost at K_d_star
    J_star_ih, _, _ = cost_and_gradient(K_d_star, A_d_grid, B_d_grid, alpha,
                                           weights, W, V, Sigma_0_dual)
    print(f"  Our J (inf-horizon dual LQR at K_d*):       {J_star_ih:.6e}")

    # ---- initialize: 0.7 K_d* (stabilizing perturbation) ----
    K_d_0 = (1.0 - INIT_PERTURB) * K_d_star
    ok_init, rho_init = is_stabilizing(K_d_0, A_d_grid, B_d_grid, alpha)
    print(f"\n[init] (1-{INIT_PERTURB})·K_d*, max ρ = {rho_init:.4f}, "
          f"stabilizing = {ok_init}")
    if not ok_init:
        raise ValueError("init not stabilizing; reduce INIT_PERTURB")

    # ---- run primal P-QRNPO on the dual plant ----
    print(f"\n[P-QRNPO on dual plant] running {MAX_ITER} iterations ...")
    K_d_final, history = p_qrnpo(K_d_0, A_d_grid, B_d_grid, alpha, weights,
                                   W, V, Sigma_0_dual, max_iter=MAX_ITER,
                                   fd_eps=HESSIAN_FD_EPS, damping=DAMPING)

    # convert back to observer-gain convention λ = K_d^T
    lambda_final = np.transpose(K_d_final, (0, 2, 1))             # (R̃, n, p)

    # final stats
    J_final = history["J"][-1]
    rel_J = J_final / J_star_ih - 1.0
    rel_dist_lambda = float(np.linalg.norm(lambda_final - lambda_star)
                              / np.linalg.norm(lambda_star))
    rel_dist_Kd = float(np.linalg.norm(K_d_final - K_d_star)
                         / np.linalg.norm(K_d_star))
    print(f"\nfinal:")
    print(f"  J = {J_final:.6e}")
    print(f"  J/J* - 1 = {rel_J:.4e}")
    print(f"  ||K_d - K_d*|| / ||K_d*||   = {rel_dist_Kd:.4e}")
    print(f"  ||λ  - λ*||  / ||λ*||      = {rel_dist_lambda:.4e}")

    # plot
    iters = np.arange(len(history["J"]))
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    ax = axes[0]
    ax.semilogy(iters, np.abs(np.array(history["J"]) - J_star_ih), "-o",
                color=COLORS["PO"], lw=1.2, ms=4, label=r"$|J - J^*|$")
    ax.set_ylabel("suboptimality")
    ax.set_title(rf"FliPASED [{CHANNEL}] dual P-QRNPO via duality "
                 rf"(LQ$\to$Kalman)")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in")

    ax = axes[1]
    ax.semilogy(iters, history["grad_norm"], "-o",
                color=COLORS["PO"], lw=1.2, ms=4)
    ax.set_ylabel(r"$\|\nabla J\|_F$")
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in")

    ax = axes[2]
    if len(history["step_size"]) > 0:
        n_steps = len(history["step_size"])
        ax.plot(iters[:n_steps], history["step_size"], "-o",
                 color=COLORS["opt"], lw=1.2, ms=4, label=r"step size $\eta$")
    ax.axhline(1.0, color=COLORS["ref"], lw=0.8, ls=":", label=r"$\eta = 1$")
    ax.set_ylabel("step size")
    ax.set_xlabel("Newton iteration")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in")

    fig.tight_layout()
    out_path = fig_dir / f"p_qrnpo_dual_{CHANNEL}.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out_path.name}")

    np.savez(res_dir / f"p_qrnpo_dual_{CHANNEL}.npz",
              J_hist=np.array(history["J"]),
              grad_norm=np.array(history["grad_norm"]),
              step_size=np.array(history["step_size"]),
              rho_max=np.array(history["rho_max"]),
              lambda_star=lambda_star,
              lambda_final=lambda_final,
              K_d_star=K_d_star, K_d_final=K_d_final,
              J_star=J_star_ih)
    print(f"saved results/p_qrnpo_dual_{CHANNEL}.npz")


if __name__ == "__main__":
    main()
