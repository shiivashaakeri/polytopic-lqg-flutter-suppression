"""
Phase 2: Full Riemannian Newton with Christoffel symbols.

Compares three optimizers on FliPASED primal:
  1. Euclidean Newton (P-QRNPO, what we already have) - uses Euclidean Hessian
  2. Riemannian Newton (this script) - uses Riemannian Hessian
                                         = Euclidean Hessian - Γ correction
  3. Natural gradient (Phase 1) - first order, Riemannian gradient

The Riemannian Hessian uses the Christoffel symbols of the
polytopic-Lyapunov metric:
  g_{(r,i,j),(s,k,ℓ)} = δ_{i,k} · ∫ w_r w_s [Y_V(κ)]_{j,ℓ} dμ(V)

Christoffel symbols (Koszul):
  Γ^t_{r,s} = (1/2) g^{tu} [∂_r g_{s,u} + ∂_s g_{r,u} - ∂_u g_{r,s}]

Riemannian Hessian:
  Hess J [V, W] = Euclidean Hess J [V, W] - Σ_t Γ^t[V, W] · ∂_t J

Newton direction G satisfies:
  Hess J [G] = -grad J   (Riemannian gradient on RHS!)

Stress test: initialize from 0.5·κ* (further from optimum than 0.7·κ*)
to see whether Riemannian Newton helps when Euclidean might struggle.

Run from project root:
    python -m examples.lqg_flipased_demo.experiment_p_riemannian_newton
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
    cost_and_gradient, deploy_gain, is_stabilizing, numerical_hessian,
    p_qrnpo, reference_optimum,
)
from examples.lqg_flipased_demo.experiment_p_riemannian_grad import (
    lyap_y, metric_matrix,
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

# Test inside Euclidean Newton's basin of attraction. From 0.5·κ* the
# polytopic-Lyapunov metric has huge curvature near flutter (Y_V scales
# like 1/(1-ρ)^2 with ρ → 1), making the Christoffel correction enormous
# and Riemannian Newton diverge — consistent with Talebi-Mesbahi Thm 4.3
# guaranteeing only LOCAL convergence within a basin around the optimum.
INIT_PERTURB = 0.3
SEED = 0

MAX_ITER = 20
HESSIAN_FD_EPS = 1e-6
METRIC_FD_EPS = 1e-7        # tightened from 1e-5 to reduce truncation error
DAMPING = 1e-6

COLORS = {
    "newton_eucl": "#0072B2",
    "newton_riem": "#009E73",
    "natural":     "#E69F00",
    "ref":         "#999999",
}


# --------------------------------------------------------------------------
# Christoffel symbols via FD on the metric
# --------------------------------------------------------------------------


def metric_at(kappa, A_grid, B_grid, alpha, weights, Sigma_0, m):
    """Build metric tensor G(κ)."""
    Y_grid = lyap_y(kappa, A_grid, B_grid, alpha, Sigma_0)
    return metric_matrix(Y_grid, alpha, weights, m)


def metric_partials_analytic(kappa, A_grid, B_grid, alpha, weights, Sigma_0, m):
    """Compute ∂G/∂κ_{(r,i,j)} for all coordinates via analytic Lyapunov
    sensitivity (clean numerics, no FD subtraction artifacts).

    For each direction δκ_r = E_{ij} (zero elsewhere):
      δK(V) = w_r(V) · E_{ij}
      δA_cl(V) = -B(V) · w_r(V) · E_{ij}
      δΣ_V solves:  δΣ_V - A_cl δΣ_V A_cl^T
                       = δA_cl Σ_V A_cl^T + A_cl Σ_V δA_cl^T
      δY_V solves:  δY_V - A_cl δY_V A_cl^T
                       = δA_cl Y_V A_cl^T + A_cl Y_V δA_cl^T + δΣ_V
    Then dG[t, ·, ·] is built from δY_V at each grid point, the same way
    G is built from Y_V.

    Returns:
        dG : ndarray of shape (d, d, d) where dG[t, :, :] = ∂G/∂κ^t
              with t flat-indexed by (r, i, j) -> r*(m*n) + i*n + j
        G  : current metric tensor at κ
    """
    R_tilde, m_dim, n = kappa.shape
    d = R_tilde * m_dim * n

    K_grid = deploy_gain(kappa, alpha)
    A_cl = A_grid - np.einsum("qij,qjk->qik", B_grid, K_grid)
    Nq = A_cl.shape[0]

    # current Σ_V and Y_V at each grid point
    Sigma_V = np.zeros((Nq, n, n))
    Y_V = np.zeros((Nq, n, n))
    for q in range(Nq):
        Sigma_V[q] = solve_discrete_lyapunov(A_cl[q], Sigma_0)
        Y_V[q] = solve_discrete_lyapunov(A_cl[q], Sigma_V[q])

    # current G
    G = metric_matrix(Y_V, alpha, weights, m_dim)

    # for each direction t = (r, i, j), compute δY_V at every grid point
    dG = np.zeros((d, d, d))

    for r in range(R_tilde):
        for i in range(m_dim):
            for j in range(n):
                t = r * (m_dim * n) + i * n + j

                # δK(V_q) = w_r(V_q) · E_{ij} ∈ R^{m × n}
                # δA_cl(V_q) = -B(V_q) δK = -w_r(V_q) B(V_q) E_{ij}
                # E_{ij} is the m×n matrix with 1 at (i,j)
                # B(V_q) E_{ij} is the n×n matrix with B(V_q)[:, i] in column j
                # So δA_cl(V_q) is rank-1: -w_r(V_q) · B(V_q)[:, i] e_j^T (n×n)

                w_r_grid = alpha[:, r]          # (Nq,)

                # δY_V at each grid point
                dY_q = np.zeros((Nq, n, n))
                for q in range(Nq):
                    if abs(w_r_grid[q]) < 1e-14:
                        continue
                    A_cl_q = A_cl[q]
                    Sigma_q = Sigma_V[q]
                    Y_q = Y_V[q]

                    # δA_cl is rank-1: u v^T with u = -w_r * B[:,i], v = e_j
                    u = -w_r_grid[q] * B_grid[q][:, i]   # (n,)
                    # δA_cl Σ_V A_cl^T = u (v^T Σ_V) A_cl^T = u (Sigma_q[j,:] @ A_cl_q.T) -> outer
                    # actually δA_cl Σ_V = u v^T Σ_V = u (Σ_V[j,:])
                    uv_Sigma = np.outer(u, Sigma_q[j, :])           # (n, n)
                    rhs_Sigma = uv_Sigma @ A_cl_q.T \
                                  + (uv_Sigma @ A_cl_q.T).T
                    # this is δA_cl Σ_V A_cl^T + (δA_cl Σ_V A_cl^T)^T = symmetric

                    # δΣ_V = Lyap(A_cl, rhs_Sigma)
                    dSigma_q = solve_discrete_lyapunov(A_cl_q, rhs_Sigma)

                    # rhs for δY: δA_cl Y_V A_cl^T + A_cl Y_V δA_cl^T + δΣ_V
                    uv_Y = np.outer(u, Y_q[j, :])                   # (n, n)
                    rhs_Y = uv_Y @ A_cl_q.T \
                             + (uv_Y @ A_cl_q.T).T \
                             + dSigma_q

                    # δY_V = Lyap(A_cl, rhs_Y)
                    dY_q[q] = solve_discrete_lyapunov(A_cl_q, rhs_Y)

                # build dG[t, :, :] = the metric tensor with Y_V replaced by δY_V
                dG[t] = metric_matrix(dY_q, alpha, weights, m_dim)

    return dG, G


def christoffel_correction(dG, G_inv, grad_eucl_flat):
    """Compute the Christoffel correction term:

       (Γ correction)_{rs} = Σ_t ∂_t J · Γ^t_{rs}
                           = (1/2) Σ_{t,u} ∂_t J · g^{tu} · (∂_r g_{su} + ∂_s g_{ru} - ∂_u g_{rs})

    Returns: (d, d) matrix to subtract from the Euclidean Hessian.

    Implementation note: dG[t, r, s] = ∂g_{rs}/∂κ^t  (Christoffel uses
    derivatives with first index = derivative-direction index).

    Working out the index gymnastics:
      Σ_{t,u} (∂_t J) · (G_inv)_{tu} · ∂_r g_{su}
        = Σ_u (G_inv ∂J)_u · dG[r, s, u]   (with dG[r, s, u] = ∂g_{su}/∂κ_r)

    But our dG storage convention is dG[t, r, s] = ∂g_{rs}/∂κ_t. So
    dG[r, s, u] in Koszul-form means dG_index1=r (the diff-direction), and
    g-indices are (s, u). With our convention that's dG[r, s, u].

    Result:
      correction[r, s] = (1/2) [
        (G_inv ∂J)_u · dG[r, s, u]       (from ∂_r g_{su})
        + (G_inv ∂J)_u · dG[s, r, u]     (from ∂_s g_{ru})
        - (G_inv ∂J)_u · dG[u, r, s]     (from ∂_u g_{rs})
      ]
    summed over u.
    """
    a = G_inv @ grad_eucl_flat   # shape (d,)
    # term1: Σ_u dG[r, s, u] a_u
    term1 = np.einsum("rsu,u->rs", dG, a)
    # term2: Σ_u dG[s, r, u] a_u
    term2 = np.einsum("sru,u->rs", dG, a)
    # term3: Σ_u dG[u, r, s] a_u
    term3 = np.einsum("urs,u->rs", dG, a)
    return 0.5 * (term1 + term2 - term3)


# --------------------------------------------------------------------------
# Riemannian Newton solver
# --------------------------------------------------------------------------


def riemannian_newton(kappa_init, A_grid, B_grid, alpha, weights, Q, R, Sigma_0,
                        max_iter=MAX_ITER, fd_eps=HESSIAN_FD_EPS,
                        damping=DAMPING, verbose=True,
                        lm_lambda_init=1.0, lm_grow=4.0, lm_shrink=0.5,
                        lm_max=1e10, c1=1e-4):
    """Riemannian Newton with adaptive Levenberg-Marquardt damping.

    The Riemannian Hessian H_R = H_E - Γ has Γ that can dominate H_E for
    high-curvature regions of the polytopic-Lyapunov metric (e.g. near
    flutter). Pure Newton with such an H_R explodes. LM damping adds
    λ·diag(H_R) to keep the inversion well-conditioned, with λ adapted
    based on whether each step actually decreases the cost (Armijo).

    Parameters
    ----------
    lm_lambda_init : initial damping
    lm_grow        : multiplier when step fails (cost increases or unstable)
    lm_shrink      : multiplier when step succeeds (cost decreases)
    lm_max         : cap on damping
    c1             : Armijo constant
    """
    kappa = kappa_init.copy()
    R_tilde, m, n = kappa.shape
    d = R_tilde * m * n
    lm_lambda = lm_lambda_init

    history = {"J": [], "grad_eucl_norm": [], "grad_riem_norm": [],
                "step_size": [], "rho_max": [], "lm_lambda": []}

    for t in range(max_iter):
        try:
            J, grad_eucl, diag = cost_and_gradient(kappa, A_grid, B_grid, alpha,
                                                     weights, Q, R, Sigma_0)
        except ValueError as e:
            print(f"iter {t}: stab violated, {e}")
            break

        history["J"].append(J)
        history["grad_eucl_norm"].append(float(np.linalg.norm(grad_eucl)))
        history["rho_max"].append(diag["rho_max"])

        # Euclidean Hessian
        H_eucl = numerical_hessian(kappa, A_grid, B_grid, alpha, weights, Q, R,
                                      Sigma_0, eps=fd_eps)

        # Metric and its derivatives (analytic Lyapunov sensitivity)
        dG, G = metric_partials_analytic(kappa, A_grid, B_grid, alpha,
                                              weights, Sigma_0, m)
        G_inv = np.linalg.inv(G + 1e-10 * np.eye(d))

        # Riemannian gradient
        grad_eucl_flat = grad_eucl.reshape(-1)
        grad_riem_flat = G_inv @ grad_eucl_flat
        grad_riem = grad_riem_flat.reshape(R_tilde, m, n)
        history["grad_riem_norm"].append(float(np.linalg.norm(grad_riem)))

        # Christoffel correction
        chris_corr = christoffel_correction(dG, G_inv, grad_eucl_flat)

        # Riemannian Hessian
        H_riem = H_eucl - chris_corr
        H_riem = 0.5 * (H_riem + H_riem.T)

        # Use diagonal of H_riem for LM scaling (Marquardt's variant)
        diag_HR = np.abs(np.diag(H_riem)) + 1e-10
        # Slope for Armijo (descent direction check)
        # In Riemannian sense: slope = <∇J, grad_riem> = grad_eucl · grad_riem
        slope = float(np.sum(grad_eucl * grad_riem))

        if verbose:
            ratio = np.linalg.norm(chris_corr) / max(np.linalg.norm(H_eucl), 1e-10)
            print(f"  [riem-newton] iter {t:>3}: J = {J:.6e}, "
                  f"||∇J||={history['grad_eucl_norm'][-1]:.4e}, "
                  f"||Γ||/||H_E||={ratio:.2e}, λ={lm_lambda:.2e}")

        # Try damped Newton with current λ; if step fails, grow λ and retry.
        accepted = False
        for inner in range(20):
            H_damped = H_riem + lm_lambda * np.diag(diag_HR) \
                         + damping * np.eye(d)
            try:
                G_flat = -np.linalg.solve(H_damped, grad_riem_flat)
            except np.linalg.LinAlgError:
                lm_lambda = min(lm_lambda * lm_grow, lm_max)
                continue
            G_dir = G_flat.reshape(R_tilde, m, n)

            # check stability with eta = 1
            eta = 1.0
            kappa_new = kappa + eta * G_dir
            ok, rho = is_stabilizing(kappa_new, A_grid, B_grid, alpha)
            if not ok:
                lm_lambda = min(lm_lambda * lm_grow, lm_max)
                if lm_lambda >= lm_max:
                    print(f"iter {t}: λ saturated at {lm_max:.0e}, abort")
                    break
                continue

            # Armijo: J_new ≤ J - c1 · η · slope (slope > 0 since grad_riem
            # is descent direction in Riemannian sense)
            try:
                J_new, _, _ = cost_and_gradient(kappa_new, A_grid, B_grid,
                                                  alpha, weights, Q, R, Sigma_0)
            except ValueError:
                lm_lambda = min(lm_lambda * lm_grow, lm_max)
                continue

            if J_new <= J - c1 * eta * slope:
                accepted = True
                # success: shrink damping for next iteration
                lm_lambda = max(lm_lambda * lm_shrink, 1e-8)
                break
            else:
                lm_lambda = min(lm_lambda * lm_grow, lm_max)
                if lm_lambda >= lm_max:
                    print(f"iter {t}: λ saturated, accepting last step anyway")
                    accepted = True
                    break

        if not accepted:
            print(f"iter {t}: damping loop failed, abort")
            break

        history["step_size"].append(eta)
        history["lm_lambda"].append(lm_lambda)
        kappa = kappa_new

        if history["grad_eucl_norm"][-1] < 1e-10:
            print(f"converged at iter {t}")
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
    print(f"Phase 2: Riemannian Newton with Christoffel symbols (channel={CHANNEL})")
    print(f"INIT_PERTURB = {INIT_PERTURB} (further from optimum than usual 0.3)")
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
    d = R_total * m * n
    print(f"Hessian/metric size: {d}×{d}, Christoffel tensor: ({d}, {d}, {d})")
    print(f"≈ {d * d} cost evaluations for Hessian + {d} metric evals per iter")

    print(f"\n[reference] P-RHPG at N={HORIZON_FOR_REF} ...")
    kappa_star, _ = reference_optimum(tp, n, Q, R, Sigma_0, HORIZON_FOR_REF)
    J_star, _, _ = cost_and_gradient(kappa_star, A_grid, B_grid, alpha,
                                        weights, Q, R, Sigma_0)
    print(f"  J* = {J_star:.6e}")

    kappa_0 = (1.0 - INIT_PERTURB) * kappa_star
    ok, rho = is_stabilizing(kappa_0, A_grid, B_grid, alpha)
    print(f"\n[init] (1-{INIT_PERTURB})·κ*, max ρ = {rho:.4f}, stab = {ok}")
    if not ok:
        raise ValueError("init not stabilizing")

    # ---- Euclidean Newton (P-QRNPO) ----
    print(f"\n[Euclidean Newton] {MAX_ITER} iters")
    _, hist_eucl = p_qrnpo(kappa_0, A_grid, B_grid, alpha, weights, Q, R,
                              Sigma_0, max_iter=MAX_ITER)

    # ---- Riemannian Newton ----
    print(f"\n[Riemannian Newton] {MAX_ITER} iters")
    _, hist_riem = riemannian_newton(kappa_0, A_grid, B_grid, alpha, weights,
                                         Q, R, Sigma_0, max_iter=MAX_ITER)

    # ---- plot ----
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax = axes[0]
    sub_eucl = np.abs(np.array(hist_eucl["J"]) - J_star)
    sub_riem = np.abs(np.array(hist_riem["J"]) - J_star)
    ax.semilogy(sub_eucl, "-o", color=COLORS["newton_eucl"], lw=1.2, ms=4,
                 label=r"Euclidean Newton (P-QRNPO)")
    ax.semilogy(sub_riem, "-s", color=COLORS["newton_riem"], lw=1.2, ms=4,
                 label=r"Riemannian Newton")
    ax.set_ylabel(r"suboptimality $|J - J^*|$")
    ax.set_title(rf"FliPASED [{CHANNEL}] Phase 2: Riemannian Newton, "
                 rf"init = $({1 - INIT_PERTURB})\kappa^*$")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in")

    ax = axes[1]
    ax.semilogy(hist_eucl["grad_norm"], "-o", color=COLORS["newton_eucl"],
                 lw=1.2, ms=4, label=r"Euclidean: $\|\nabla J\|$")
    ax.semilogy(hist_riem["grad_eucl_norm"], "-s", color=COLORS["newton_riem"],
                 lw=1.2, ms=4, label=r"Riemannian: $\|\nabla J\|$")
    ax.set_ylabel("Euclidean gradient norm")
    ax.set_xlabel("Newton iteration")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in")

    fig.tight_layout()
    out_path = fig_dir / f"riemannian_newton_{CHANNEL}.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out_path.name}")

    np.savez(res_dir / f"riemannian_newton_{CHANNEL}.npz",
              J_eucl=np.array(hist_eucl["J"]),
              J_riem=np.array(hist_riem["J"]),
              grad_norm_eucl=np.array(hist_eucl["grad_norm"]),
              grad_norm_riem=np.array(hist_riem["grad_eucl_norm"]),
              grad_riem_norm=np.array(hist_riem["grad_riem_norm"]),
              J_star=J_star)
    print(f"saved results/riemannian_newton_{CHANNEL}.npz")


if __name__ == "__main__":
    main()
