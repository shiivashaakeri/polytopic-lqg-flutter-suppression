"""
Polytopic Quasi-Riemannian Newton Policy Optimization (P-QRNPO).

Framing B: stationary polytopic gain stack {kappa_r}_{r=1..R̃}, infinite
horizon, on the polytopic stabilizing set. Adapts Talebi-Mesbahi 2024 TAC
"Policy Optimization Over Submanifolds" to the polytopic LPV case.

Geometry (in short):
  - Manifold: S_inf^κ = open subset of R^(R̃ mn) on which the deployed gain
    K(V) = Σ_r w_r(V) κ_r stabilizes A(V) - B(V)K(V) for all V ∈ P.
  - Riemannian metric (polytopic-Lyapunov, parameter-integrated):
      <V, W>_κ = ∫ tr[Ṽ(V)^T W̃(V) Y_V(κ)] dμ(V)
      where Ṽ(V) = Σ_r w_r(V) V_r and Y_V is the Lyapunov solution.
  - Newton direction G via Euclidean Hessian:
      H · vec(G) = -vec(∇J)
    (Newton dir doesn't need the metric; metric only enters stability cert.)
  - Stability certificate s_κ = min_V λ_min(Q_κ) / (2 λ_max(P_V) ||B(V)|| ||G̃(V)||)
  - Step: κ ← κ + min(s_κ, 1) · G

For this first test, the Euclidean Hessian is computed numerically via
finite differences on the analytic gradient (R̃ · mn columns per Newton
iteration). Closed-form Lyapunov-sensitivity expressions are TODO once
this version validates.

Test plant: FliPASED reduced sym (14 states, 4 vertices, dt=5ms).
Comparison: infinite-horizon stationary optimum recovered from your
existing P-RHPG backward sweep at N=200 (which is converged).

Run from project root:
    python -m examples.lqg_flipased_demo.experiment_p_qrnpo
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
from examples.lqg_flipased_demo.pipeline import (
    DEFAULT_DT, DEFAULT_GRIDSIZE, DEFAULT_R_CTRL, _default_Q,
)
from tptool.prhpg.synthesis import (
    PRHPGProblem,
    _build_quadrature_weights,
    _evaluate_AB_on_grid,
    _evaluate_weights_on_grid,
    _extract_vertices,
)


CHANNEL = "sym"
HULL = "cno"
HORIZON_FOR_REF = 200       # P-RHPG backward sweep horizon (large = infinite limit)

# P-QRNPO settings
MAX_ITER = 30
HESSIAN_FD_EPS = 1e-6       # finite-difference step for numerical Hessian
DAMPING = 1e-6              # regularize Newton system: H + DAMPING I

# Initialization
INIT = "perturb_optimum"    # 'zero' won't stabilize at high V (flutter)
INIT_PERTURB = 0.3          # multiplicative perturbation: κ_0 = κ* (1 - δ)
SEED = 0

COLORS = {
    "PO":     "#0072B2",
    "opt":    "#D55E00",
    "ref":    "#999999",
}


# --------------------------------------------------------------------------
# Helpers: closed-loop matrices and Lyapunov solves on the grid
# --------------------------------------------------------------------------


def deploy_gain(kappa, alpha):
    """K(V_q) = Σ_r w_r(V_q) κ_r at every grid point.

    kappa : (R̃, m, n)
    alpha : (N_q, R̃)
    Returns: (N_q, m, n)
    """
    return np.einsum("qi,imn->qmn", alpha, kappa)


def closed_loop_grid(A_grid, B_grid, K_grid):
    """A_cl(V_q) = A(V_q) - B(V_q) K(V_q)."""
    BK = np.einsum("qij,qjk->qik", B_grid, K_grid)
    return A_grid - BK


def solve_lyap_each(A_cl_grid, Q_or_W_grid):
    """Solve P = A_cl^T P A_cl + Q at each grid point.

    A_cl_grid    : (N_q, n, n)
    Q_or_W_grid  : (N_q, n, n)  (per-grid-point RHS)
    Returns: (N_q, n, n)
    """
    Nq = A_cl_grid.shape[0]
    n = A_cl_grid.shape[1]
    out = np.zeros((Nq, n, n))
    for q in range(Nq):
        # solve_discrete_lyapunov solves X = A X A^T + Q (note: A, not A^T)
        # We want P = A_cl^T P A_cl + Q -> use A = A_cl^T
        out[q] = solve_discrete_lyapunov(A_cl_grid[q].T, Q_or_W_grid[q])
    return out


def solve_lyap_dual_each(A_cl_grid, Sigma_0):
    """Solve Σ = A_cl Σ A_cl^T + Σ_0 at each grid point.

    A_cl_grid : (N_q, n, n)
    Sigma_0   : (n, n)
    Returns: (N_q, n, n)
    """
    Nq = A_cl_grid.shape[0]
    n = A_cl_grid.shape[1]
    out = np.zeros((Nq, n, n))
    for q in range(Nq):
        out[q] = solve_discrete_lyapunov(A_cl_grid[q], Sigma_0)
    return out


def is_stabilizing(kappa, A_grid, B_grid, alpha, tol=1.0 - 1e-8):
    """Check max spectral radius of A_cl(V_q) over the grid."""
    K_grid = deploy_gain(kappa, alpha)
    A_cl = closed_loop_grid(A_grid, B_grid, K_grid)
    rhos = [float(np.max(np.abs(np.linalg.eigvals(A_cl[q]))))
            for q in range(A_cl.shape[0])]
    return max(rhos) < tol, max(rhos)


# --------------------------------------------------------------------------
# Cost and gradient (analytic, infinite horizon)
# --------------------------------------------------------------------------


def cost_and_gradient(kappa, A_grid, B_grid, alpha, weights, Q, R, Sigma_0):
    """Polytopic infinite-horizon LQR cost and Euclidean gradient.

    Returns:
        J         : float
        grad      : (R̃, m, n)
        diagnostics : dict (P, Sigma, K_grid, A_cl, max spectral radius)
    """
    K_grid = deploy_gain(kappa, alpha)                           # (Nq, m, n)
    A_cl = closed_loop_grid(A_grid, B_grid, K_grid)              # (Nq, n, n)

    # check stability across grid
    rhos = np.array([float(np.max(np.abs(np.linalg.eigvals(A_cl[q]))))
                      for q in range(A_cl.shape[0])])
    if rhos.max() >= 1.0:
        raise ValueError(f"polytopic gain is not stabilizing: "
                          f"max ρ(A_cl) = {rhos.max():.6f} at grid point "
                          f"{int(np.argmax(rhos))}")

    # cost-to-go P_V at each grid point
    n = A_cl.shape[1]
    QKRK = (Q[None, :, :]
             + np.einsum("qji,jk,qkl->qil", K_grid, R, K_grid))   # (Nq, n, n)
    P_grid = solve_lyap_each(A_cl, QKRK)                          # (Nq, n, n)

    # state covariance Σ_V at each grid point
    Sigma_grid = solve_lyap_dual_each(A_cl, Sigma_0)              # (Nq, n, n)

    # cost: J = ∫ tr(P_V Σ_0) dμ(V)
    PS = np.einsum("qij,jk->qik", P_grid, Sigma_0)
    J = float(np.sum(weights * np.einsum("qii->q", PS)))

    # LTI gradient at each grid point
    # E_V = (R + B^T P_V B) K - B^T P_V A
    BPB = np.einsum("qji,qjk,qkl->qil", B_grid, P_grid, B_grid)   # (Nq, m, m)
    BPA = np.einsum("qji,qjk,qkl->qil", B_grid, P_grid, A_grid)   # (Nq, m, n)
    R_BPB = R[None, :, :] + BPB
    E_V = np.einsum("qij,qjk->qik", R_BPB, K_grid) - BPA          # (Nq, m, n)
    G_lti = 2.0 * np.einsum("qij,qjk->qik", E_V, Sigma_grid)      # (Nq, m, n)

    # polytopic gradient: aggregate via Lemma 3
    grad = np.einsum("q,qi,qmn->imn", weights, alpha, G_lti)      # (R̃, m, n)

    diag = {
        "P": P_grid, "Sigma": Sigma_grid, "K_grid": K_grid,
        "A_cl": A_cl, "rho_max": float(rhos.max()),
    }
    return J, grad, diag


# --------------------------------------------------------------------------
# Numerical Euclidean Hessian via finite differences on the gradient
# --------------------------------------------------------------------------


def numerical_hessian(kappa, A_grid, B_grid, alpha, weights, Q, R, Sigma_0,
                       eps=HESSIAN_FD_EPS):
    """Forward-difference Hessian: H[i, j] = (∇J(κ + ε e_j) - ∇J(κ))_i / ε.

    Symmetrize at the end for numerical stability.
    """
    R_tilde, m, n = kappa.shape
    d = R_tilde * m * n

    # gradient at the base point
    _, g0, _ = cost_and_gradient(kappa, A_grid, B_grid, alpha, weights, Q, R, Sigma_0)
    g0_flat = g0.reshape(-1)

    H = np.zeros((d, d))
    for j in range(d):
        kappa_p = kappa.copy().reshape(-1)
        kappa_p[j] += eps
        kappa_p = kappa_p.reshape(R_tilde, m, n)
        try:
            _, gp, _ = cost_and_gradient(kappa_p, A_grid, B_grid, alpha,
                                          weights, Q, R, Sigma_0)
            H[:, j] = (gp.reshape(-1) - g0_flat) / eps
        except ValueError as e:
            # perturbation pushed us out of stability region; fall back to backward diff
            kappa_m = kappa.copy().reshape(-1)
            kappa_m[j] -= eps
            kappa_m = kappa_m.reshape(R_tilde, m, n)
            _, gm, _ = cost_and_gradient(kappa_m, A_grid, B_grid, alpha,
                                          weights, Q, R, Sigma_0)
            H[:, j] = (g0_flat - gm.reshape(-1)) / eps

    H = 0.5 * (H + H.T)   # symmetrize
    return H


# --------------------------------------------------------------------------
# Stability certificate (Talebi-Mesbahi Lemma 4.1, polytopic generalization)
# --------------------------------------------------------------------------


def stability_certificate(kappa, G, A_grid, B_grid, alpha, Q_kappa_fn=None, Q=None, R=None):
    """Polytopic stability certificate for Newton step direction G.

    s_κ = min_q λ_min(Q_κ(V_q)) / (2 λ_max(P_q) ||B(V_q)|| ||G̃(V_q)||)

    Q_κ_fn is a smooth, possibly per-grid-point choice. Default: at each
    grid point, use Q_κ(V_q) = Q + K(V_q)^T R K(V_q), the actual stage
    cost the cost-to-go P_q solves with. This makes the certificate
    much less conservative than Q_κ = I, which is the choice that crushes
    the step size when the optimum has nonzero gain.
    """
    K_grid = deploy_gain(kappa, alpha)
    A_cl = closed_loop_grid(A_grid, B_grid, K_grid)
    G_grid = deploy_gain(G, alpha)

    n = A_cl.shape[1]
    Nq = A_cl.shape[0]
    s_per = np.zeros(Nq)
    for q in range(Nq):
        if Q_kappa_fn is None and Q is not None and R is not None:
            # Q_κ(V_q) = Q + K(V_q)^T R K(V_q)
            K_q = K_grid[q]
            Q_kappa = Q + K_q.T @ R @ K_q
        else:
            Q_kappa = Q_kappa_fn(q) if Q_kappa_fn is not None else np.eye(n)
        lam_Q = float(np.linalg.eigvalsh(Q_kappa).min())
        # P_q solves P = A_cl^T P A_cl + Q_κ
        P_q = solve_discrete_lyapunov(A_cl[q].T, Q_kappa)
        lam_P = float(np.linalg.eigvalsh(P_q).max())
        B_q_norm = float(np.linalg.norm(B_grid[q], 2))
        G_q_norm = float(np.linalg.norm(G_grid[q], 2))
        if G_q_norm < 1e-12 or B_q_norm < 1e-12:
            s_per[q] = np.inf
        else:
            s_per[q] = lam_Q / (2.0 * lam_P * B_q_norm * G_q_norm)
    return float(s_per.min())


# --------------------------------------------------------------------------
# P-QRNPO main loop
# --------------------------------------------------------------------------


def p_qrnpo(kappa_init, A_grid, B_grid, alpha, weights, Q, R, Sigma_0,
             max_iter=MAX_ITER, fd_eps=HESSIAN_FD_EPS, damping=DAMPING,
             verbose=True):
    """Polytopic Quasi-Riemannian Newton PO (Euclidean-connection variant)."""
    kappa = kappa_init.copy()
    R_tilde, m, n = kappa.shape

    history = {"J": [], "grad_norm": [], "step_size": [], "rho_max": [],
                "newton_residual": []}

    for t in range(max_iter):
        # cost + gradient
        try:
            J, grad, diag = cost_and_gradient(kappa, A_grid, B_grid, alpha,
                                                weights, Q, R, Sigma_0)
        except ValueError as e:
            print(f"iter {t}: stability violated, {e}")
            break

        history["J"].append(J)
        history["grad_norm"].append(float(np.linalg.norm(grad)))
        history["rho_max"].append(diag["rho_max"])

        if verbose:
            print(f"iter {t:>3}: J = {J:.6e}, ||∇J|| = "
                  f"{history['grad_norm'][-1]:.4e}, "
                  f"max ρ(A_cl) = {diag['rho_max']:.4f}")

        # numerical Hessian
        H = numerical_hessian(kappa, A_grid, B_grid, alpha, weights, Q, R,
                                Sigma_0, eps=fd_eps)

        # solve damped Newton system
        H_damped = H + damping * np.eye(H.shape[0])
        try:
            G_flat = -np.linalg.solve(H_damped, grad.reshape(-1))
        except np.linalg.LinAlgError:
            print(f"iter {t}: Hessian singular; aborting")
            break
        G = G_flat.reshape(R_tilde, m, n)

        # diagnostic: ||H G + ∇J||
        residual = float(np.linalg.norm(H @ G_flat + grad.reshape(-1)))
        history["newton_residual"].append(residual)

        # try unit step first; backtrack only if it violates stability
        eta = 1.0
        kappa_new = kappa + eta * G
        ok, rho = is_stabilizing(kappa_new, A_grid, B_grid, alpha)
        if not ok:
            # backtrack: halve step until stable, but no smaller than the
            # provable stability certificate (Talebi-Mesbahi Lemma 4.1)
            s_kappa = stability_certificate(kappa, G, A_grid, B_grid, alpha,
                                              Q=Q, R=R)
            for _ in range(20):
                eta *= 0.5
                kappa_new = kappa + eta * G
                ok, rho = is_stabilizing(kappa_new, A_grid, B_grid, alpha)
                if ok:
                    break
                if eta < s_kappa:
                    eta = s_kappa
                    kappa_new = kappa + eta * G
                    ok, rho = is_stabilizing(kappa_new, A_grid, B_grid, alpha)
                    break
            if not ok:
                print(f"iter {t}: backtracking failed, max ρ = {rho:.4f}")
                break

        history["step_size"].append(eta)
        kappa = kappa_new

        # convergence check
        if history["grad_norm"][-1] < 1e-10:
            print(f"converged at iter {t}: ||∇J|| < 1e-10")
            break

    return kappa, history


# --------------------------------------------------------------------------
# Reference: infinite-horizon stationary optimum from P-RHPG at large N
# --------------------------------------------------------------------------


def reference_optimum(tp, n, Q, R, Sigma_0, N_ref):
    sf = PRHPGProblem(tp, n=n)
    sf.set_stage_cost(Q=Q, R=R)
    sf.set_initial_covariance(Sigma_0)
    sf.set_horizon(N=N_ref)
    K_0_flat, J_star = sf.solve()
    # K_0 from P-RHPG is in the (A + BK stable) convention; we want u = -Kx
    # convention (A - BK stable), so negate.
    kappa_star = -K_0_flat                                # (R̃, m, n)
    return kappa_star, J_star


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
    print(f"P-QRNPO sanity test (channel={CHANNEL})")
    print("=" * 72)

    # build TP form
    poly_full = flipased_polytope(CHANNEL)
    poly_red = flipased_polytope_reduced(CHANNEL)
    poly_red_d = discretize_polytope(poly_red, dt=DEFAULT_DT)
    n = poly_red_d.n

    tp = flipased_tpform(poly_red_d, hull=HULL,
                          gridsize=DEFAULT_GRIDSIZE, verbose=True)

    Q = _default_Q(n, CHANNEL)
    R = DEFAULT_R_CTRL.copy()
    Sigma_0 = np.eye(n)

    # extract grid quantities
    A_v, B_v, R_total, m = _extract_vertices(tp.S, n)
    alpha = _evaluate_weights_on_grid(tp.U)
    A_grid, B_grid = _evaluate_AB_on_grid(A_v, B_v, alpha)
    weights = _build_quadrature_weights(tp.U, tp.domain)

    print(f"\nn={n}, m={m}, R̃={R_total}, N_q={alpha.shape[0]}")
    print(f"Hessian size: {R_total*m*n} × {R_total*m*n} "
          f"({R_total*m*n} FD calls per Newton iter)")

    # reference optimum (large-horizon P-RHPG)
    print(f"\n[reference] P-RHPG at N={HORIZON_FOR_REF} ...")
    kappa_star, J_star = reference_optimum(tp, n, Q, R, Sigma_0,
                                              HORIZON_FOR_REF)
    # Compute J at kappa_star using *our* infinite-horizon cost
    J_star_ih, _, _ = cost_and_gradient(kappa_star, A_grid, B_grid, alpha,
                                           weights, Q, R, Sigma_0)
    print(f"  P-RHPG J (finite horizon, accumulated): {J_star:.6e}")
    print(f"  Our J (infinite horizon, at κ*):        {J_star_ih:.6e}")
    print(f"  ||κ*||_F = {np.linalg.norm(kappa_star):.4e}")

    # initialize: open-loop FliPASED is UNSTABLE above V_F (flutter), so κ=0
    # doesn't stabilize. Need a stabilizing starting point. Use a scaled version
    # of κ* for guaranteed stability + a real Newton convergence test.
    rng = np.random.default_rng(SEED)
    if INIT == "zero":
        kappa_0 = np.zeros((R_total, m, n))
    elif INIT == "perturb_optimum":
        # start at (1 - δ) κ*, a perturbation toward zero
        kappa_0 = (1.0 - INIT_PERTURB) * kappa_star
    elif INIT == "small_random":
        kappa_0 = INIT_PERTURB * rng.standard_normal((R_total, m, n))
    else:
        raise ValueError(INIT)

    # check that init is stabilizing (open-loop FliPASED IS stable, so κ=0 works)
    ok, rho_init = is_stabilizing(kappa_0, A_grid, B_grid, alpha)
    print(f"\n[init] κ=0, max ρ(A_cl) = {rho_init:.4f}, stabilizing = {ok}")
    if not ok:
        raise ValueError("κ=0 is not stabilizing; pick a different init")

    # run P-QRNPO
    print(f"\n[P-QRNPO] running {MAX_ITER} iterations ...")
    kappa_final, history = p_qrnpo(kappa_0, A_grid, B_grid, alpha, weights,
                                      Q, R, Sigma_0, max_iter=MAX_ITER)

    # final stats
    J_final = history["J"][-1]
    rel_J = J_final / J_star_ih - 1.0
    rel_dist = float(np.linalg.norm(kappa_final - kappa_star)
                      / np.linalg.norm(kappa_star))
    print(f"\nfinal: J = {J_final:.6e}")
    print(f"       J/J* - 1 = {rel_J:.4e}")
    print(f"       ||κ - κ*|| / ||κ*|| = {rel_dist:.4e}")

    # plot
    iters = np.arange(len(history["J"]))
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    ax = axes[0]
    ax.semilogy(iters, np.abs(np.array(history["J"]) - J_star_ih), "-o",
                color=COLORS["PO"], lw=1.2, ms=4, label=r"$|J - J^*|$")
    ax.set_ylabel("suboptimality")
    ax.set_title(rf"FliPASED [{CHANNEL}] P-QRNPO (Euclidean connection)")
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
    ax.plot(iters[:-1] if len(history["step_size"]) < len(iters)
             else iters,
             history["step_size"], "-o",
             color=COLORS["opt"], lw=1.2, ms=4, label=r"step size $\eta$")
    ax.axhline(1.0, color=COLORS["ref"], lw=0.8, ls=":", label=r"$\eta = 1$")
    ax.set_ylabel("step size")
    ax.set_xlabel("Newton iteration")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in")

    fig.tight_layout()
    out_path = fig_dir / f"p_qrnpo_{CHANNEL}.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out_path.name}")

    np.savez(res_dir / f"p_qrnpo_{CHANNEL}.npz",
              J_hist=np.array(history["J"]),
              grad_norm=np.array(history["grad_norm"]),
              step_size=np.array(history["step_size"]),
              rho_max=np.array(history["rho_max"]),
              kappa_star=kappa_star, kappa_final=kappa_final,
              J_star=J_star_ih)
    print(f"saved results/p_qrnpo_{CHANNEL}.npz")


if __name__ == "__main__":
    main()
