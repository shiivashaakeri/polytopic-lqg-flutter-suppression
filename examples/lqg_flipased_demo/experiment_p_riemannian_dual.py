"""
Direct dual natural gradient: polytopic observer PO without the
LQ-Kalman duality trick.

Setup (no duality):
  - Variables: λ = (λ_1, ..., λ_R̃), each λ_r ∈ R^(n×p)
  - Deployed observer gain: L(V) = Σ_r w_r(V) λ_r
  - Closed-loop estimation error: A^e_cl(V) = A(V) - L(V) C(V)
  - Steady-state error covariance:
      Σ^e_V = A^e_cl Σ^e_V A^e_cl^T + W + L V_meas L^T
  - Cost: J^e(λ) = ∫ tr[S · Σ^e_V] dμ(V),  S = I

Gradient (LTI Kalman gradient, Lemma 3 routed):
  ∇_L J^e_V = -2 P^e_V A(V) Σ^e_V C(V)^T
              + 2 P^e_V L(V) (C(V) Σ^e_V C(V)^T + V_meas)
  where P^e_V solves P^e = A^e_cl^T P^e A^e_cl + S

Polytopic-Lyapunov metric (dual analog):
  <V, W>_λ = ∫ tr[Ṽ(V)^T W̃(V) Y^e_V(λ)] dμ(V)
  Y^e_V(λ) = Lyap(A^e_cl(V), Σ^e_V(λ))

Reference: ObserverPRHPGProblem at large horizon (which uses duality
internally; we compare its λ_star to ours).

Run from project root:
    python -m examples.lqg_flipased_demo.experiment_p_riemannian_dual
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
from examples.lqg_flipased_demo.experiment_p_qrnpo_dual import V_MEAS, W_PROC
from examples.lqg_flipased_demo.experiment_p_riemannian_grad import (
    metric_matrix, riemannian_gradient,
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

INIT_PERTURB = 0.3
SEED = 0

ETA_NATURAL = 1.0
ETA_EUCLIDEAN = 1e-6        # different scale than primal because dual cost is smaller
MAX_ITER = 50

COLORS = {
    "natural":   "#0072B2",
    "euclidean": "#E69F00",
    "ref":       "#999999",
}


# --------------------------------------------------------------------------
# Direct dual cost and gradient (no duality)
# --------------------------------------------------------------------------


def deploy_observer(lam, alpha):
    """L(V_q) = Σ_r w_r(V_q) λ_r at every grid point.

    lam   : (R̃, n, p)
    alpha : (N_q, R̃)
    Returns: (N_q, n, p)
    """
    return np.einsum("qi,inp->qnp", alpha, lam)


def is_observer_stabilizing(lam, A_grid, C_grid, alpha, tol=1.0 - 1e-8):
    """Check max ρ(A - LC) over the grid."""
    L_grid = deploy_observer(lam, alpha)
    A_e_cl = A_grid - np.einsum("qnp,qpk->qnk", L_grid, C_grid)
    rhos = [float(np.max(np.abs(np.linalg.eigvals(A_e_cl[q]))))
             for q in range(A_e_cl.shape[0])]
    return max(rhos) < tol, max(rhos)


def dual_cost_and_gradient(lam, A_grid, C_grid, alpha, weights, W, V_meas, S):
    """Polytopic dual cost J^e and its Euclidean gradient.

    Returns:
        J : float
        grad : (R̃, n, p)
        diag : dict with Sigma_e, P_e, A_e_cl, L_grid, rho_max
    """
    L_grid = deploy_observer(lam, alpha)                          # (Nq, n, p)
    # closed-loop error dynamics
    LC = np.einsum("qnp,qpk->qnk", L_grid, C_grid)                # (Nq, n, n)
    A_e_cl = A_grid - LC                                           # (Nq, n, n)

    # check stability
    rhos = np.array([float(np.max(np.abs(np.linalg.eigvals(A_e_cl[q]))))
                       for q in range(A_e_cl.shape[0])])
    if rhos.max() >= 1.0:
        raise ValueError(f"observer not stabilizing: max ρ = {rhos.max():.6f}")

    # noise input on error: ξ - L ω, covariance W + L V_meas L^T
    LV = np.einsum("qnp,pl->qnl", L_grid, V_meas)                 # (Nq, n, p)
    LVLt = np.einsum("qnl,qml->qnm", LV, L_grid)                  # (Nq, n, n)
    Q_eff = W[None, :, :] + LVLt                                   # (Nq, n, n)

    # forward error covariance: Σ^e = A_e_cl Σ^e A_e_cl^T + Q_eff
    Nq = A_e_cl.shape[0]
    n = A_e_cl.shape[1]
    Sigma_e = np.zeros((Nq, n, n))
    for q in range(Nq):
        Sigma_e[q] = solve_discrete_lyapunov(A_e_cl[q], Q_eff[q])

    # backward cost-to-go: P^e = A_e_cl^T P^e A_e_cl + S
    P_e = np.zeros((Nq, n, n))
    for q in range(Nq):
        P_e[q] = solve_discrete_lyapunov(A_e_cl[q].T, S)

    # cost: J^e = ∫ tr(S Σ^e_V) dμ(V)
    SS = np.einsum("ij,qjk->qik", S, Sigma_e)
    J = float(np.sum(weights * np.einsum("qii->q", SS)))

    # gradient w.r.t. L(V_q):
    #   ∇_L J^e_V = -2 P^e A Σ^e C^T + 2 P^e L (C Σ^e C^T + V_meas)
    PA = np.einsum("qij,qjk->qik", P_e, A_grid)                   # (Nq, n, n)
    PA_Sigma = np.einsum("qij,qjk->qik", PA, Sigma_e)             # (Nq, n, n)
    term_A = -2.0 * np.einsum("qij,qkj->qik", PA_Sigma, C_grid)   # (Nq, n, p)

    CSC = np.einsum("qij,qjk,qlk->qil", C_grid, Sigma_e, C_grid)  # (Nq, p, p)
    inner = CSC + V_meas[None, :, :]                               # (Nq, p, p)
    PL = np.einsum("qij,qjk->qik", P_e, L_grid)                   # (Nq, n, p)
    term_L = 2.0 * np.einsum("qij,qjk->qik", PL, inner)           # (Nq, n, p)

    G_lti = term_A + term_L                                        # (Nq, n, p)

    # polytopic gradient via Lemma 3
    grad = np.einsum("q,qi,qnp->inp", weights, alpha, G_lti)      # (R̃, n, p)

    diag = {"Sigma_e": Sigma_e, "P_e": P_e, "A_e_cl": A_e_cl,
            "L_grid": L_grid, "rho_max": float(rhos.max())}
    return J, grad, diag


# --------------------------------------------------------------------------
# Polytopic-Lyapunov metric for the dual
# --------------------------------------------------------------------------


def lyap_y_dual(lam, A_grid, C_grid, alpha, W, V_meas):
    """Compute Y^e_V(λ) = Lyap(A^e_cl(V), Σ^e_V(λ)) at every grid point."""
    L_grid = deploy_observer(lam, alpha)
    LC = np.einsum("qnp,qpk->qnk", L_grid, C_grid)
    A_e_cl = A_grid - LC

    LV = np.einsum("qnp,pl->qnl", L_grid, V_meas)
    LVLt = np.einsum("qnl,qml->qnm", LV, L_grid)
    Q_eff = W[None, :, :] + LVLt

    Nq = A_e_cl.shape[0]
    n = A_e_cl.shape[1]
    Y_grid = np.zeros((Nq, n, n))
    for q in range(Nq):
        # Σ^e step
        Sigma_e = solve_discrete_lyapunov(A_e_cl[q], Q_eff[q])
        # Y^e step
        Y_grid[q] = solve_discrete_lyapunov(A_e_cl[q], Sigma_e)
    return Y_grid


def metric_matrix_dual(Y_grid, alpha, weights, p):
    """Metric tensor for the dual: g_{(r,i,j),(s,k,ℓ)} = δ_{j,ℓ} ∫ w_r w_s [Y]_{i,k} dμ.

    For the dual, the "matrix" variable is L ∈ R^(n×p), so coordinates are
    (r, i, j) with i ∈ [n], j ∈ [p]. The metric block has the same structure
    as primal but transposed: the n-index is paired with i (output dim),
    the p-index is paired with j (input dim of the Lyapunov metric variant).

    Specifically: <V, W>_λ = ∫ tr(Ṽ^T W̃ Y) dμ. With Ṽ ∈ R^(n×p), Y ∈ R^(n×n),
    tr(Ṽ^T W̃ Y) = Σ_{i,j,k} Ṽ_{i,j} W̃_{k,j} Y_{ki}.
    Wait — that's not right. tr(A B) for A,B square is Σ_{i,j} A_{ij} B_{ji}.
    Here Ṽ^T ∈ R^(p×n), W̃ ∈ R^(n×p), so Ṽ^T W̃ ∈ R^(p×p). Then
    tr((Ṽ^T W̃) Y) doesn't typecheck since Y is n×n.

    Correct expression: for Ṽ, W̃ ∈ R^(n×p), the LTI Lyapunov metric
    inner product is tr(Ṽ^T W̃ · M) where M is some p×p matrix derived
    from the Lyap of the dual closed-loop. Or equivalently
    tr(Ṽ M' Ṽ^T) with M' an n×n matrix. The cleanest LTI dual-LQR
    Lyapunov metric (per Talebi-Mesbahi) for state-feedback with
    "input dimension" p and "state dimension" n is
        <V, W> = tr(V^T W · Y)
    where Y is n×n in the natural way. But the dimensions only work if
    V, W are p×n, not n×p.

    For the dual we have L ∈ R^(n×p) (output gain, n=state, p=output).
    So if we identify "control-side" with p (the input axis of the dual
    LQR-on-transposed-plant), the metric should be
        <V, W>_λ = tr(V^T W Y),  V, W ∈ R^(n×p), Y ∈ R^(p×p)
    BUT Y in our derivation was n×n (the dual Lyap with n×n A_e_cl).

    Let me re-derive the correct dual metric form below.
    """
    raise NotImplementedError("see comment - dual metric needs care; "
                                "use metric_matrix_dual_v2 instead")


def metric_matrix_dual_v2(Y_grid, alpha, weights, p):
    """Polytopic-Lyapunov metric for dual L ∈ R^(n×p).

    Inner product: for tangent vectors V, W ∈ R^(R̃, n, p), let
        Ṽ(q) = Σ_r w_r(V_q) V_r ∈ R^(n×p),
    and define
        <V, W>_λ = ∫ tr(Ṽ(V)^T Y^e_V(λ) W̃(V)) dμ(V).
    Here Ṽ^T ∈ R^(p×n), Y^e ∈ R^(n×n), W̃ ∈ R^(n×p), so the product
    is p×p and the trace is well-defined.

    In coordinates (r, i, j) (n-index i, p-index j):
        Ṽ_{i,j}(q) = Σ_r α_r(q) V_{r,i,j}
        tr(Ṽ^T Y W̃) = Σ_{i,k,j} Ṽ_{i,j} Y_{i,k} W̃_{k,j}
    so
        g_{(r,i,j),(s,k,ℓ)}(λ) = δ_{j,ℓ} · ∫ w_r w_s Y_{i,k} dμ
    """
    Nq = Y_grid.shape[0]
    n = Y_grid.shape[1]
    R_tilde = alpha.shape[1]
    d = R_tilde * n * p

    # block A_rs[r,s,i,k] = Σ_q weights[q] alpha[q,r] alpha[q,s] Y[q,i,k]
    A_rs = np.einsum("q,qr,qs,qik->rsik", weights, alpha, alpha, Y_grid)

    G = np.zeros((d, d))
    for r in range(R_tilde):
        for s in range(R_tilde):
            for j in range(p):
                # block at (r, *, j) × (s, *, j) is the n×n A_rs[r, s]
                row_start = r * (n * p) + j  # need careful indexing
                # Use vec convention: flat = r*(n*p) + i*p + j
                for i in range(n):
                    for k in range(n):
                        row = r * (n * p) + i * p + j
                        col = s * (n * p) + k * p + j
                        G[row, col] = A_rs[r, s, i, k]
    return G


# --------------------------------------------------------------------------
# Optimizers (mirror primal experiment)
# --------------------------------------------------------------------------


def euclidean_gd_dual(lam_init, A_grid, C_grid, alpha, weights, W, V_meas, S,
                       eta=ETA_EUCLIDEAN, max_iter=MAX_ITER, verbose=True):
    lam = lam_init.copy()
    history = {"J": [], "grad_norm": [], "rho_max": []}

    for t in range(max_iter):
        try:
            J, grad, diag = dual_cost_and_gradient(lam, A_grid, C_grid, alpha,
                                                       weights, W, V_meas, S)
        except ValueError:
            print(f"[eucl-dual] iter {t}: stab violated")
            break

        history["J"].append(J)
        history["grad_norm"].append(float(np.linalg.norm(grad)))
        history["rho_max"].append(diag["rho_max"])

        if verbose and (t % 5 == 0 or t == max_iter - 1):
            print(f"  [eucl-dual] iter {t:>3}: J = {J:.6e}, "
                  f"||∇J|| = {history['grad_norm'][-1]:.4e}")

        lam = lam - eta * grad

    return lam, history


def natural_gd_dual(lam_init, A_grid, C_grid, alpha, weights, W, V_meas, S,
                     eta=ETA_NATURAL, max_iter=MAX_ITER, verbose=True,
                     use_armijo=True, c1=1e-4):
    lam = lam_init.copy()
    R_tilde, n, p = lam.shape
    history = {"J": [], "grad_eucl_norm": [], "grad_riem_norm": [],
                "rho_max": [], "step_size": []}

    for t in range(max_iter):
        try:
            J, grad_eucl, diag = dual_cost_and_gradient(lam, A_grid, C_grid,
                                                            alpha, weights, W,
                                                            V_meas, S)
        except ValueError:
            print(f"[nat-dual] iter {t}: stab violated")
            break

        history["J"].append(J)
        history["grad_eucl_norm"].append(float(np.linalg.norm(grad_eucl)))
        history["rho_max"].append(diag["rho_max"])

        # build metric tensor and Riemannian gradient
        Y_grid = lyap_y_dual(lam, A_grid, C_grid, alpha, W, V_meas)
        G = metric_matrix_dual_v2(Y_grid, alpha, weights, p)
        # need to reshape grad_eucl to match G's flat layout (r, i, j)
        # with flat = r*(n*p) + i*p + j -- this is standard row-major reshape
        grad_eucl_flat = grad_eucl.reshape(-1)
        grad_riem_flat = np.linalg.solve(G, grad_eucl_flat)
        grad_riem = grad_riem_flat.reshape(R_tilde, n, p)
        history["grad_riem_norm"].append(float(np.linalg.norm(grad_riem)))

        slope = float(np.sum(grad_eucl * grad_riem))

        if verbose and (t % 5 == 0 or t == max_iter - 1):
            print(f"  [nat-dual]  iter {t:>3}: J = {J:.6e}, "
                  f"||∇J||={history['grad_eucl_norm'][-1]:.4e}, "
                  f"||grad_riem||={history['grad_riem_norm'][-1]:.4e}")

        # Armijo line search
        step = eta
        for _ in range(40):
            lam_new = lam - step * grad_riem
            ok, rho = is_observer_stabilizing(lam_new, A_grid, C_grid, alpha)
            if not ok:
                step *= 0.5
                continue
            if use_armijo:
                try:
                    J_new, _, _ = dual_cost_and_gradient(lam_new, A_grid, C_grid,
                                                            alpha, weights, W,
                                                            V_meas, S)
                except ValueError:
                    step *= 0.5
                    continue
                if J_new <= J - c1 * step * slope:
                    break
                else:
                    step *= 0.5
            else:
                break
        else:
            print(f"[nat-dual] iter {t}: line search failed")
            break

        history["step_size"].append(step)
        lam = lam_new

        if history["grad_eucl_norm"][-1] < 1e-10:
            print(f"[nat-dual] converged at iter {t}")
            break

    return lam, history


# --------------------------------------------------------------------------
# Reference: dual P-RHPG
# --------------------------------------------------------------------------


def reference_dual_optimum(tp, n, C_mid, W, V_meas, N_ref):
    obs = ObserverPRHPGProblem(tp=tp, C=C_mid, n=n)
    obs.set_noise_covariances(W=W, V=V_meas)
    obs.set_horizon(N=N_ref)
    L_flat, J_finite = obs.solve()
    return L_flat, J_finite      # L_flat shape (R̃, n, p)


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
    print("Direct dual natural gradient (no duality trick)")
    print("=" * 72)

    poly_full = flipased_polytope(CHANNEL)
    poly_red = flipased_polytope_reduced(CHANNEL)
    poly_red_d = discretize_polytope(poly_red, dt=DEFAULT_DT)
    n = poly_red_d.n
    p = poly_red_d.p

    tp = flipased_tpform(poly_red_d, hull=HULL, gridsize=DEFAULT_GRIDSIZE,
                          verbose=True)

    A_v, B_v, R_total, m = _extract_vertices(tp.S, n)
    alpha = _evaluate_weights_on_grid(tp.U)
    A_grid, B_grid = _evaluate_AB_on_grid(A_v, B_v, alpha)
    weights = _build_quadrature_weights(tp.U, tp.domain)

    C_mid = poly_red_d.C[poly_red_d.R // 2]
    C_grid = np.tile(C_mid[None, :, :], (alpha.shape[0], 1, 1))

    W = W_PROC * np.eye(n)
    V_meas = V_MEAS * np.eye(p)
    S = np.eye(n)

    print(f"\nn={n}, p={p}, R̃={R_total}, N_q={alpha.shape[0]}")

    # reference
    print(f"\n[reference] dual P-RHPG at N={HORIZON_FOR_REF} ...")
    lambda_star, J_finite = reference_dual_optimum(tp, n, C_mid, W, V_meas,
                                                      HORIZON_FOR_REF)
    J_star, _, _ = dual_cost_and_gradient(lambda_star, A_grid, C_grid, alpha,
                                              weights, W, V_meas, S)
    print(f"  dual P-RHPG J (finite, N={HORIZON_FOR_REF}): {J_finite:.6e}")
    print(f"  Direct J^e at λ_star (inf horizon):          {J_star:.6e}")
    print(f"  ||λ*||_F = {np.linalg.norm(lambda_star):.4e}")

    # init
    lam_0 = (1.0 - INIT_PERTURB) * lambda_star
    ok, rho = is_observer_stabilizing(lam_0, A_grid, C_grid, alpha)
    print(f"\n[init] (1-{INIT_PERTURB})·λ*, max ρ(A-LC) = {rho:.4f}, stab = {ok}")

    # run both
    print(f"\n[euclidean GD on dual] η = {ETA_EUCLIDEAN}, {MAX_ITER} iters")
    _, hist_eucl = euclidean_gd_dual(lam_0, A_grid, C_grid, alpha, weights,
                                          W, V_meas, S,
                                          eta=ETA_EUCLIDEAN, max_iter=MAX_ITER)

    print(f"\n[natural GD on dual]   η = {ETA_NATURAL}, {MAX_ITER} iters")
    _, hist_nat = natural_gd_dual(lam_0, A_grid, C_grid, alpha, weights,
                                       W, V_meas, S,
                                       eta=ETA_NATURAL, max_iter=MAX_ITER)

    # plot
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax = axes[0]
    ax.semilogy(np.abs(np.array(hist_eucl["J"]) - J_star), "-o",
                color=COLORS["euclidean"], lw=1.2, ms=4,
                label=r"Euclidean GD ($\eta = " + f"{ETA_EUCLIDEAN:.0e}" + "$)")
    ax.semilogy(np.abs(np.array(hist_nat["J"]) - J_star), "-o",
                color=COLORS["natural"], lw=1.2, ms=4,
                label=r"Natural GD ($\eta = " + f"{ETA_NATURAL:.1f}" + "$, Armijo)")
    ax.set_ylabel(r"suboptimality $|J^e - J^{e,*}|$")
    ax.set_title(rf"FliPASED [{CHANNEL}] dual: natural gradient (direct, "
                 rf"no duality trick)")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in")

    ax = axes[1]
    ax.semilogy(hist_eucl["grad_norm"], "-o",
                color=COLORS["euclidean"], lw=1.2, ms=4,
                label=r"Euclidean: $\|\nabla J^e\|$")
    ax.semilogy(hist_nat["grad_eucl_norm"], "-o",
                color=COLORS["natural"], lw=1.2, ms=4,
                label=r"Natural: $\|\nabla J^e\|$")
    ax.semilogy(hist_nat["grad_riem_norm"], "--s",
                color=COLORS["natural"], lw=0.8, ms=3, alpha=0.6,
                label=r"Natural: $\|\mathrm{grad}\, J^e\|_g$")
    ax.set_ylabel("gradient norm")
    ax.set_xlabel("iteration")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in")

    fig.tight_layout()
    out_path = fig_dir / f"riemannian_dual_{CHANNEL}.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out_path.name}")

    np.savez(res_dir / f"riemannian_dual_{CHANNEL}.npz",
              J_eucl=np.array(hist_eucl["J"]),
              J_nat=np.array(hist_nat["J"]),
              grad_norm_eucl=np.array(hist_eucl["grad_norm"]),
              grad_norm_nat=np.array(hist_nat["grad_eucl_norm"]),
              grad_riem_norm=np.array(hist_nat["grad_riem_norm"]),
              J_star=J_star, lambda_star=lambda_star)
    print(f"saved results/riemannian_dual_{CHANNEL}.npz")


if __name__ == "__main__":
    main()
