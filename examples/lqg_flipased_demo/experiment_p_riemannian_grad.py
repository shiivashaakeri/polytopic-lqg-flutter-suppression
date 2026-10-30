"""
Phase 1 of polytopic Riemannian PO: natural gradient with the
polytopic-Lyapunov metric.

The Riemannian gradient is grad J = G_κ^{-1} · ∇J, where G_κ is the
metric tensor matrix:

  G_κ[(r,i,j),(s,k,ℓ)] = δ_{i,k} · ∫ w_r(V) w_s(V) [Y_V(κ)]_{j,ℓ} dμ(V)

with Y_V(κ) the solution to the Lyapunov equation
  Y_V = A_cl(V) Y_V A_cl(V)^T + Σ_V(κ)
and Σ_V(κ) the closed-loop state covariance at parameter V.

This is a Phase 1 sanity test: we use the Riemannian gradient with a
vanilla gradient step (no Newton, no connection), and check that
convergence is faster than vanilla Euclidean gradient descent. If
the metric is doing the right preconditioning, we should see a clean
linear rate without needing the full Newton machinery.

Comparison: same FliPASED setup as experiment_p_qrnpo.py.
Reference: closed-form κ* from existing P-RHPG.

Run from project root:
    python -m examples.lqg_flipased_demo.experiment_p_riemannian_grad
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
    cost_and_gradient, deploy_gain, is_stabilizing, reference_optimum,
)
from examples.lqg_flipased_demo.pipeline import (
    DEFAULT_DT, DEFAULT_GRIDSIZE, DEFAULT_R_CTRL, _default_Q,
)
from tptool.prhpg.synthesis import (
    _build_quadrature_weights,
    _evaluate_AB_on_grid,
    _evaluate_weights_on_grid,
    _extract_vertices,
)


CHANNEL = "sym"
HULL = "cno"
HORIZON_FOR_REF = 200

INIT_PERTURB = 0.3
SEED = 0

# Phase 1: natural gradient descent with polytopic-Lyapunov metric
ETA_NATURAL = 1.0           # stepsize for natural gradient
ETA_EUCLIDEAN = 1e-8        # stepsize for vanilla Euclidean GD baseline
MAX_ITER = 50

COLORS = {
    "natural":   "#0072B2",
    "euclidean": "#E69F00",
    "newton":    "#009E73",
    "ref":       "#999999",
}


# --------------------------------------------------------------------------
# Metric tensor: build G_κ matrix from polytopic-Lyapunov metric
# --------------------------------------------------------------------------


def lyap_y(kappa, A_grid, B_grid, alpha, Sigma_0):
    """Compute Y_V(κ) at every grid point.

    Y_V satisfies: Y_V = A_cl(V) Y_V A_cl(V)^T + Σ_V(κ)
    where Σ_V(κ) solves: Σ_V = A_cl(V) Σ_V A_cl(V)^T + Σ_0.

    Returns: (N_q, n, n)
    """
    K_grid = deploy_gain(kappa, alpha)
    A_cl = A_grid - np.einsum("qij,qjk->qik", B_grid, K_grid)
    Nq = A_cl.shape[0]
    n = A_cl.shape[1]
    Y_grid = np.zeros((Nq, n, n))
    for q in range(Nq):
        # Σ_V step
        Sigma_V = solve_discrete_lyapunov(A_cl[q], Sigma_0)
        # Y_V step
        Y_grid[q] = solve_discrete_lyapunov(A_cl[q], Sigma_V)
    return Y_grid


def metric_matrix(Y_grid, alpha, weights, m):
    """Build the metric tensor matrix G_κ of shape (R̃·m·n, R̃·m·n).

    g_{(r,i,j),(s,k,ℓ)} = δ_{i,k} · ∫ w_r(V) w_s(V) [Y_V]_{j,ℓ} dμ(V)

    Block structure: pick coordinate ordering (r, i, j) → flat index
    f(r, i, j) = r * (m * n) + i * n + j. Then for fixed i = k, the
    (r,j),(s,ℓ) entry is determined; for i ≠ k, entry is 0.
    """
    Nq = Y_grid.shape[0]
    n = Y_grid.shape[1]
    R_tilde = alpha.shape[1]
    d = R_tilde * m * n

    # build the per-(r,s) matrix block: A_{rs} = ∫ w_r w_s Y_V dμ ∈ R^{n×n}
    # alpha shape: (Nq, R̃)
    # Y_grid shape: (Nq, n, n)
    # weights shape: (Nq,)
    # A[r, s, j, ℓ] = Σ_q weights[q] alpha[q,r] alpha[q,s] Y[q,j,ℓ]
    A_rs = np.einsum("q,qr,qs,qjl->rsjl", weights, alpha, alpha, Y_grid)
    # shape: (R̃, R̃, n, n)

    G = np.zeros((d, d))
    for r in range(R_tilde):
        for s in range(R_tilde):
            for i in range(m):
                # block at (r, i, *) × (s, i, *) is the n×n A_rs[r,s]
                # other (i ≠ k) blocks are zero
                row_start = r * (m * n) + i * n
                col_start = s * (m * n) + i * n
                G[row_start:row_start + n, col_start:col_start + n] = A_rs[r, s]
    return G


def riemannian_gradient(grad_eucl, G):
    """Solve G · vec(grad_riem) = vec(grad_eucl) for the Riemannian gradient.

    grad_eucl : (R̃, m, n)
    G         : (R̃ m n, R̃ m n) metric matrix
    Returns: grad_riem (R̃, m, n)
    """
    R_tilde, m, n = grad_eucl.shape
    g_flat = grad_eucl.reshape(-1)
    # Solve G · x = g  (G is symmetric PD, so use cholesky-based solve)
    riem_flat = np.linalg.solve(G, g_flat)
    return riem_flat.reshape(R_tilde, m, n)


# --------------------------------------------------------------------------
# Optimizers
# --------------------------------------------------------------------------


def euclidean_gd(kappa_init, A_grid, B_grid, alpha, weights, Q, R, Sigma_0,
                  eta=ETA_EUCLIDEAN, max_iter=MAX_ITER, verbose=True):
    """Vanilla Euclidean gradient descent baseline."""
    kappa = kappa_init.copy()
    history = {"J": [], "grad_norm": [], "rho_max": []}

    for t in range(max_iter):
        try:
            J, grad, diag = cost_and_gradient(kappa, A_grid, B_grid, alpha,
                                                weights, Q, R, Sigma_0)
        except ValueError as e:
            print(f"[euclidean] iter {t}: stability violated, {e}")
            break

        history["J"].append(J)
        history["grad_norm"].append(float(np.linalg.norm(grad)))
        history["rho_max"].append(diag["rho_max"])

        if verbose and (t % 5 == 0 or t == max_iter - 1):
            print(f"  [euclidean] iter {t:>3}: J = {J:.6e}, "
                  f"||∇J|| = {history['grad_norm'][-1]:.4e}")

        kappa = kappa - eta * grad

    return kappa, history


def natural_gd(kappa_init, A_grid, B_grid, alpha, weights, Q, R, Sigma_0,
                eta=ETA_NATURAL, max_iter=MAX_ITER, verbose=True,
                use_armijo=True, c1=1e-4):
    """Natural gradient descent with polytopic-Lyapunov metric.

    If use_armijo=True, perform Armijo backtracking line search:
        accept step if J(κ - η g_riem) ≤ J(κ) - c1 · η · <g_eucl, g_riem>
    """
    kappa = kappa_init.copy()
    R_tilde, m, n = kappa.shape
    history = {"J": [], "grad_eucl_norm": [], "grad_riem_norm": [],
                "rho_max": [], "step_size": []}

    for t in range(max_iter):
        try:
            J, grad_eucl, diag = cost_and_gradient(kappa, A_grid, B_grid, alpha,
                                                     weights, Q, R, Sigma_0)
        except ValueError as e:
            print(f"[natural] iter {t}: stability violated, {e}")
            break

        history["J"].append(J)
        history["grad_eucl_norm"].append(float(np.linalg.norm(grad_eucl)))
        history["rho_max"].append(diag["rho_max"])

        # build metric tensor and Riemannian gradient
        Y_grid = lyap_y(kappa, A_grid, B_grid, alpha, Sigma_0)
        G = metric_matrix(Y_grid, alpha, weights, m)
        grad_riem = riemannian_gradient(grad_eucl, G)
        history["grad_riem_norm"].append(float(np.linalg.norm(grad_riem)))

        # descent slope: <∇J, g_riem> = g_eucl · g_riem (in Euclidean inner product)
        slope = float(np.sum(grad_eucl * grad_riem))

        if verbose and (t % 5 == 0 or t == max_iter - 1):
            print(f"  [natural]   iter {t:>3}: J = {J:.6e}, "
                  f"||∇J||={history['grad_eucl_norm'][-1]:.4e}, "
                  f"||grad_riem||={history['grad_riem_norm'][-1]:.4e}")

        # try unit step, backtrack on instability OR Armijo failure
        step = eta
        for bt in range(40):
            kappa_new = kappa - step * grad_riem
            ok, rho = is_stabilizing(kappa_new, A_grid, B_grid, alpha)
            if not ok:
                step *= 0.5
                continue
            if use_armijo:
                try:
                    J_new, _, _ = cost_and_gradient(kappa_new, A_grid, B_grid,
                                                       alpha, weights, Q, R,
                                                       Sigma_0)
                except ValueError:
                    step *= 0.5
                    continue
                # Armijo: J_new ≤ J - c1 step · slope
                if J_new <= J - c1 * step * slope:
                    break
                else:
                    step *= 0.5
            else:
                break
        else:
            print(f"[natural] iter {t}: line search failed")
            break

        history["step_size"].append(step)
        kappa = kappa_new

        if history["grad_eucl_norm"][-1] < 1e-10:
            print(f"[natural] converged at iter {t}")
            break

    return kappa, history


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main():
    out_dir = Path("examples/lqg_flipased_demo")
    fig_dir = out_dir / "figures"
    res_dir = out_dir / "results"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Phase 1 Riemannian: natural gradient (no Newton, no connection)")
    print("=" * 72)

    poly_full = flipased_polytope(CHANNEL)
    poly_red = flipased_polytope_reduced(CHANNEL)
    poly_red_d = discretize_polytope(poly_red, dt=DEFAULT_DT)
    n = poly_red_d.n

    tp = flipased_tpform(poly_red_d, hull=HULL, gridsize=DEFAULT_GRIDSIZE,
                          verbose=True)

    Q = _default_Q(n, CHANNEL)
    R = DEFAULT_R_CTRL.copy()
    Sigma_0 = np.eye(n)

    A_v, B_v, R_total, m = _extract_vertices(tp.S, n)
    alpha = _evaluate_weights_on_grid(tp.U)
    A_grid, B_grid = _evaluate_AB_on_grid(A_v, B_v, alpha)
    weights = _build_quadrature_weights(tp.U, tp.domain)

    print(f"\nn={n}, m={m}, R̃={R_total}, N_q={alpha.shape[0]}")
    print(f"metric tensor size: {R_total*m*n}×{R_total*m*n}")

    # reference
    print(f"\n[reference] P-RHPG at N={HORIZON_FOR_REF} ...")
    kappa_star, _ = reference_optimum(tp, n, Q, R, Sigma_0, HORIZON_FOR_REF)
    J_star, _, _ = cost_and_gradient(kappa_star, A_grid, B_grid, alpha,
                                        weights, Q, R, Sigma_0)
    print(f"  J* = {J_star:.6e}")

    # init
    kappa_0 = (1.0 - INIT_PERTURB) * kappa_star
    ok, rho = is_stabilizing(kappa_0, A_grid, B_grid, alpha)
    print(f"\n[init] (1-{INIT_PERTURB})·κ*, max ρ = {rho:.4f}, stab = {ok}")

    # run both: vanilla Euclidean GD vs natural gradient
    print(f"\n[euclidean GD] η = {ETA_EUCLIDEAN}, {MAX_ITER} iters")
    _, hist_eucl = euclidean_gd(kappa_0, A_grid, B_grid, alpha, weights,
                                  Q, R, Sigma_0,
                                  eta=ETA_EUCLIDEAN, max_iter=MAX_ITER)

    print(f"\n[natural GD]   η = {ETA_NATURAL}, {MAX_ITER} iters")
    _, hist_nat = natural_gd(kappa_0, A_grid, B_grid, alpha, weights,
                               Q, R, Sigma_0,
                               eta=ETA_NATURAL, max_iter=MAX_ITER)

    # plot
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax = axes[0]
    ax.semilogy(np.abs(np.array(hist_eucl["J"]) - J_star), "-o",
                color=COLORS["euclidean"], lw=1.2, ms=4,
                label=r"Euclidean GD ($\eta = " + f"{ETA_EUCLIDEAN:.0e}" + "$)")
    ax.semilogy(np.abs(np.array(hist_nat["J"]) - J_star), "-o",
                color=COLORS["natural"], lw=1.2, ms=4,
                label=r"Natural GD ($\eta = " + f"{ETA_NATURAL:.1f}" + "$)")
    ax.set_ylabel("suboptimality $|J - J^*|$")
    ax.set_title(rf"FliPASED [{CHANNEL}] Phase 1: natural gradient "
                 rf"vs Euclidean GD")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in")

    ax = axes[1]
    ax.semilogy(hist_eucl["grad_norm"], "-o",
                color=COLORS["euclidean"], lw=1.2, ms=4,
                label="Euclidean: $\\|\\nabla J\\|$")
    ax.semilogy(hist_nat["grad_eucl_norm"], "-o",
                color=COLORS["natural"], lw=1.2, ms=4,
                label="Natural: $\\|\\nabla J\\|$")
    ax.semilogy(hist_nat["grad_riem_norm"], "--s",
                color=COLORS["natural"], lw=0.8, ms=3, alpha=0.6,
                label="Natural: $\\|\\mathrm{grad}\\, J\\|_g$")
    ax.set_ylabel("gradient norm")
    ax.set_xlabel("iteration")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in")

    fig.tight_layout()
    out_path = fig_dir / f"riemannian_phase1_{CHANNEL}.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out_path.name}")

    np.savez(res_dir / f"riemannian_phase1_{CHANNEL}.npz",
              J_eucl=np.array(hist_eucl["J"]),
              J_nat=np.array(hist_nat["J"]),
              grad_norm_eucl=np.array(hist_eucl["grad_norm"]),
              grad_norm_nat=np.array(hist_nat["grad_eucl_norm"]),
              grad_riem_norm=np.array(hist_nat["grad_riem_norm"]),
              J_star=J_star)
    print(f"saved results/riemannian_phase1_{CHANNEL}.npz")


if __name__ == "__main__":
    main()
