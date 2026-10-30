"""
Joint LQG Polytopic Quasi-Riemannian Newton PO.

Optimizes the polytopic state-feedback stack κ AND the polytopic observer
stack λ TOGETHER on the LQG cost (controller uses observer estimate).
Tests the polytopic LQG separation principle empirically:

    Conjecture (polytopic LQG separation):
        argmin_(κ, λ) J^LQG(κ, λ) = (κ*_primal,  λ*_dual)

    where κ*_primal solves the primal polytopic LQR (full-state feedback)
    and λ*_dual solves the polytopic Kalman / dual LQR.

If this holds, joint Newton from any stabilizing init converges to the
SAME (κ, λ) we got from running primal and dual P-QRNPO independently.

Augmented dynamics on z = (x, x̂):
    z_{t+1} = A_aug(V) z_t + B_w(V) ξ + B_v(V) ω
    A_aug(V) = [ A(V)         -B(V) K(V)                          ]
               [ L(V) C(V)    A(V) - B(V) K(V) - L(V) C(V)        ]
    B_w(V)   = [ I            ]
               [ 0            ]
    B_v(V)   = [ 0            ]
               [ L(V)         ]

LQG cost (with control = -K x̂, so u^T R u = x̂^T K^T R K x̂):
    J^LQG = ∫ tr[ Q_aug Σ_z(V) ] dμ(V)
    Q_aug = [ Q              0           ]
            [ 0              K^T R K     ]

State covariance Σ_z(V) solves Σ_z = A_aug Σ_z A_aug^T + Σ_noise(V)
where Σ_noise = B_w W B_w^T + B_v V B_v^T = blockdiag(W, L V L^T).

Run from project root:
    python -m examples.lqg_flipased_demo.experiment_p_qrnpo_lqg
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
    DAMPING, HESSIAN_FD_EPS, MAX_ITER,
    cost_and_gradient as primal_cost_and_grad,
    deploy_gain, is_stabilizing,
)
from examples.lqg_flipased_demo.experiment_p_qrnpo_dual import (
    V_MEAS, W_PROC,
)
from examples.lqg_flipased_demo.pipeline import (
    DEFAULT_DT, DEFAULT_GRIDSIZE, DEFAULT_R_CTRL, _default_Q,
)
from tptool.prhpg import ObserverPRHPGProblem, PRHPGProblem
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
MAX_ITER_LQG = 30
HESSIAN_FD_EPS_LQG = 1e-6
DAMPING_LQG = 1e-6

COLORS = {
    "PO":  "#0072B2",
    "opt": "#D55E00",
    "ref": "#999999",
    "sep": "#009E73",
}


# --------------------------------------------------------------------------
# Joint LQG cost and gradient (numerical via finite differences for now)
# --------------------------------------------------------------------------


def lqg_closed_loop_at_grid(kappa, lam, A_grid, B_grid, C_grid, alpha):
    """Build A_aug(V_q), the 2n x 2n augmented closed-loop matrix.

    kappa : (R̃, m, n)        primal vertex stack
    lam   : (R̃, n, p)        dual vertex stack
    A_grid: (Nq, n, n), B_grid: (Nq, n, m), C_grid: (Nq, p, n)
    alpha : (Nq, R̃)
    Returns: (Nq, 2n, 2n)
    """
    Nq = A_grid.shape[0]
    n = A_grid.shape[1]
    K_grid = deploy_gain(kappa, alpha)                    # (Nq, m, n)
    L_grid = np.einsum("qi,inp->qnp", alpha, lam)         # (Nq, n, p)

    A_aug = np.zeros((Nq, 2 * n, 2 * n))
    for q in range(Nq):
        A_q = A_grid[q]; B_q = B_grid[q]; C_q = C_grid[q]
        K_q = K_grid[q]; L_q = L_grid[q]
        A_aug[q, :n, :n] = A_q
        A_aug[q, :n, n:] = -B_q @ K_q
        A_aug[q, n:, :n] = L_q @ C_q
        A_aug[q, n:, n:] = A_q - B_q @ K_q - L_q @ C_q
    return A_aug, K_grid, L_grid


def lqg_noise_cov_at_grid(L_grid, W, V_meas, n):
    """Σ_noise(V_q) = blockdiag(W, L V L^T)."""
    Nq = L_grid.shape[0]
    LV = np.einsum("qnp,pl->qnl", L_grid, V_meas)         # (Nq, n, p)
    LVLt = np.einsum("qnl,qml->qnm", LV, L_grid)          # (Nq, n, n)
    Sigma_noise = np.zeros((Nq, 2 * n, 2 * n))
    Sigma_noise[:, :n, :n] = W[None, :, :]
    Sigma_noise[:, n:, n:] = LVLt
    return Sigma_noise


def lqg_state_weight(K_grid, Q, R, n):
    """Q_aug(V_q) = blockdiag(Q, K^T R K)."""
    Nq = K_grid.shape[0]
    KRK = np.einsum("qji,jl,qlk->qik", K_grid, R, K_grid)  # (Nq, n, n)
    Q_aug = np.zeros((Nq, 2 * n, 2 * n))
    Q_aug[:, :n, :n] = Q[None, :, :]
    Q_aug[:, n:, n:] = KRK
    return Q_aug


def lqg_cost(kappa, lam, A_grid, B_grid, C_grid, alpha, weights, Q, R, W, V_meas):
    """Polytopic LQG cost J^LQG = ∫ tr(Q_aug Σ_z(V)) dμ(V)."""
    n = A_grid.shape[1]
    A_aug, K_grid, L_grid = lqg_closed_loop_at_grid(kappa, lam, A_grid, B_grid,
                                                       C_grid, alpha)
    Sigma_noise = lqg_noise_cov_at_grid(L_grid, W, V_meas, n)
    Q_aug = lqg_state_weight(K_grid, Q, R, n)

    Nq = A_grid.shape[0]
    J_per = np.zeros(Nq)
    for q in range(Nq):
        # check stability before solving Lyapunov
        rho = float(np.max(np.abs(np.linalg.eigvals(A_aug[q]))))
        if rho >= 1.0:
            raise ValueError(f"joint LQG closed-loop unstable at grid point {q}: "
                              f"ρ = {rho:.6f}")
        # Σ_z = A_aug Σ_z A_aug^T + Σ_noise
        Sigma_z = solve_discrete_lyapunov(A_aug[q], Sigma_noise[q])
        J_per[q] = float(np.trace(Q_aug[q] @ Sigma_z))
    return float(np.sum(weights * J_per)), A_aug, K_grid, L_grid


def is_lqg_stabilizing(kappa, lam, A_grid, B_grid, C_grid, alpha,
                         tol=1.0 - 1e-8):
    A_aug, _, _ = lqg_closed_loop_at_grid(kappa, lam, A_grid, B_grid, C_grid, alpha)
    rhos = [float(np.max(np.abs(np.linalg.eigvals(A_aug[q]))))
             for q in range(A_aug.shape[0])]
    return max(rhos) < tol, max(rhos)


def lqg_gradient_analytic(kappa, lam, A_grid, B_grid, C_grid, alpha, weights,
                            Q, R, W, V_meas):
    """Analytic LQG gradient via Lyapunov sensitivity.

    Returns: J, grad_kappa (R̃, m, n), grad_lambda (R̃, n, p)
    """
    Nq = A_grid.shape[0]
    n = A_grid.shape[1]
    R_tilde = kappa.shape[0]
    m = kappa.shape[1]
    p = lam.shape[2]

    K_grid = np.einsum("qi,imn->qmn", alpha, kappa)
    L_grid = np.einsum("qi,inp->qnp", alpha, lam)

    grad_kappa = np.zeros_like(kappa)
    grad_lambda = np.zeros_like(lam)
    J_total = 0.0

    for q in range(Nq):
        A_q = A_grid[q]; B_q = B_grid[q]; C_q = C_grid[q]
        K_q = K_grid[q]; L_q = L_grid[q]
        w_q = weights[q]
        alpha_q = alpha[q]

        A_aug = np.zeros((2 * n, 2 * n))
        A_aug[:n, :n] = A_q
        A_aug[:n, n:] = -B_q @ K_q
        A_aug[n:, :n] = L_q @ C_q
        A_aug[n:, n:] = A_q - B_q @ K_q - L_q @ C_q

        rho = float(np.max(np.abs(np.linalg.eigvals(A_aug))))
        if rho >= 1.0:
            raise ValueError(f"unstable A_aug at q={q}, ρ={rho:.6f}")

        KRK = K_q.T @ R @ K_q
        Q_aug = np.zeros((2 * n, 2 * n))
        Q_aug[:n, :n] = Q
        Q_aug[n:, n:] = KRK

        LVL = L_q @ V_meas @ L_q.T
        N_aug = np.zeros((2 * n, 2 * n))
        N_aug[:n, :n] = W
        N_aug[n:, n:] = LVL

        Sigma_z = solve_discrete_lyapunov(A_aug, N_aug)
        P_aug = solve_discrete_lyapunov(A_aug.T, Q_aug)

        J_total += w_q * float(np.trace(Q_aug @ Sigma_z))

        Shh = Sigma_z[n:, n:]
        P22 = P_aug[n:, n:]
        M = P_aug @ A_aug @ Sigma_z
        M12 = M[:n, n:]
        M21 = M[n:, :n]
        M22 = M[n:, n:]

        # gradient w.r.t. K(V_q)
        gK_from_Q = 2.0 * R @ K_q @ Shh
        gK_from_A = 2.0 * (-B_q.T) @ (M12 + M22)
        gK_total = gK_from_Q + gK_from_A

        # gradient w.r.t. L(V_q)
        gL_from_A = 2.0 * (M21 - M22) @ C_q.T
        gL_from_N = 2.0 * P22 @ L_q @ V_meas
        gL_total = gL_from_A + gL_from_N

        # polytopic routing: ∂J/∂κ_r += w_q · α_r(V_q) · gK_total
        grad_kappa += w_q * alpha_q[:, None, None] * gK_total[None, :, :]
        grad_lambda += w_q * alpha_q[:, None, None] * gL_total[None, :, :]

    return J_total, grad_kappa, grad_lambda


def lqg_gradient_fd(kappa, lam, A_grid, B_grid, C_grid, alpha, weights, Q, R,
                     W, V_meas, eps=1e-6):
    """Wrapper that calls the analytic gradient (renamed for compatibility)."""
    _, gk, gl = lqg_gradient_analytic(kappa, lam, A_grid, B_grid, C_grid,
                                          alpha, weights, Q, R, W, V_meas)
    return gk, gl


def lqg_hessian_fd(kappa, lam, A_grid, B_grid, C_grid, alpha, weights, Q, R,
                    W, V_meas, eps=1e-5):
    """Forward-FD Hessian of J^LQG: H[i,j] = (∇J(p+ε e_j) - ∇J(p))_i / ε.

    p = (κ, λ) flattened.
    """
    R_tilde, m, n = kappa.shape
    p = lam.shape[2]
    d_k = R_tilde * m * n
    d_l = R_tilde * n * p
    d = d_k + d_l

    g_k0, g_l0 = lqg_gradient_fd(kappa, lam, A_grid, B_grid, C_grid, alpha,
                                    weights, Q, R, W, V_meas, eps=eps)
    g0 = np.concatenate([g_k0.reshape(-1), g_l0.reshape(-1)])

    H = np.zeros((d, d))
    for j in range(d):
        if j < d_k:
            kappa_p = kappa.copy().reshape(-1)
            kappa_p[j] += eps
            kappa_p = kappa_p.reshape(R_tilde, m, n)
            lam_p = lam
        else:
            kappa_p = kappa
            lam_p = lam.copy().reshape(-1)
            lam_p[j - d_k] += eps
            lam_p = lam_p.reshape(R_tilde, n, p)
        try:
            g_kp, g_lp = lqg_gradient_fd(kappa_p, lam_p, A_grid, B_grid, C_grid,
                                            alpha, weights, Q, R, W, V_meas,
                                            eps=eps)
            gp = np.concatenate([g_kp.reshape(-1), g_lp.reshape(-1)])
            H[:, j] = (gp - g0) / eps
        except ValueError:
            # backward diff fallback
            if j < d_k:
                kappa_m = kappa.copy().reshape(-1)
                kappa_m[j] -= eps
                kappa_m = kappa_m.reshape(R_tilde, m, n)
                lam_m = lam
            else:
                kappa_m = kappa
                lam_m = lam.copy().reshape(-1)
                lam_m[j - d_k] -= eps
                lam_m = lam_m.reshape(R_tilde, n, p)
            g_km, g_lm = lqg_gradient_fd(kappa_m, lam_m, A_grid, B_grid, C_grid,
                                            alpha, weights, Q, R, W, V_meas,
                                            eps=eps)
            gm = np.concatenate([g_km.reshape(-1), g_lm.reshape(-1)])
            H[:, j] = (g0 - gm) / eps

    H = 0.5 * (H + H.T)
    return H


def joint_p_qrnpo(kappa_init, lam_init, A_grid, B_grid, C_grid, alpha, weights,
                    Q, R, W, V_meas, max_iter=MAX_ITER_LQG,
                    fd_eps=HESSIAN_FD_EPS_LQG, damping=DAMPING_LQG, verbose=True):
    kappa = kappa_init.copy()
    lam = lam_init.copy()
    R_tilde, m, n = kappa.shape
    p = lam.shape[2]

    history = {"J": [], "grad_norm": [], "step_size": [], "rho_max": []}

    for t in range(max_iter):
        try:
            J, _, _, _ = lqg_cost(kappa, lam, A_grid, B_grid, C_grid, alpha,
                                     weights, Q, R, W, V_meas)
            ok, rho = is_lqg_stabilizing(kappa, lam, A_grid, B_grid, C_grid,
                                            alpha)
        except ValueError as e:
            print(f"iter {t}: stability violated, {e}")
            break

        history["J"].append(J)
        history["rho_max"].append(rho)

        # gradient
        g_k, g_l = lqg_gradient_fd(kappa, lam, A_grid, B_grid, C_grid, alpha,
                                      weights, Q, R, W, V_meas, eps=fd_eps)
        g_flat = np.concatenate([g_k.reshape(-1), g_l.reshape(-1)])
        history["grad_norm"].append(float(np.linalg.norm(g_flat)))

        if verbose:
            print(f"iter {t:>3}: J = {J:.6e}, ||∇J|| = "
                  f"{history['grad_norm'][-1]:.4e}, max ρ = {rho:.4f}")

        if history["grad_norm"][-1] < 1e-9:
            print(f"converged at iter {t}: ||∇J|| < 1e-9")
            break

        # Hessian
        H = lqg_hessian_fd(kappa, lam, A_grid, B_grid, C_grid, alpha, weights,
                              Q, R, W, V_meas, eps=fd_eps)
        H_damped = H + damping * np.eye(H.shape[0])

        try:
            G_flat = -np.linalg.solve(H_damped, g_flat)
        except np.linalg.LinAlgError:
            print(f"iter {t}: Hessian singular; aborting")
            break

        d_k = R_tilde * m * n
        G_k = G_flat[:d_k].reshape(R_tilde, m, n)
        G_l = G_flat[d_k:].reshape(R_tilde, n, p)

        # try unit step, backtrack on instability
        eta = 1.0
        kappa_new = kappa + eta * G_k
        lam_new = lam + eta * G_l
        ok_new, rho_new = is_lqg_stabilizing(kappa_new, lam_new, A_grid, B_grid,
                                                C_grid, alpha)
        if not ok_new:
            for _ in range(20):
                eta *= 0.5
                kappa_new = kappa + eta * G_k
                lam_new = lam + eta * G_l
                ok_new, rho_new = is_lqg_stabilizing(kappa_new, lam_new,
                                                        A_grid, B_grid, C_grid,
                                                        alpha)
                if ok_new:
                    break
            if not ok_new:
                print(f"iter {t}: backtracking failed, max ρ = {rho_new:.4f}")
                break

        history["step_size"].append(eta)
        kappa = kappa_new
        lam = lam_new

    return kappa, lam, history


# --------------------------------------------------------------------------
# Reference: separated primal κ* and dual λ* from existing P-RHPG
# --------------------------------------------------------------------------


def reference_separated(tp, n, p, Q, R, W, V_meas, Sigma_0, N_ref, C_mid):
    """Compute separated κ* and λ* using existing P-RHPG (primal and dual)."""
    # primal
    sf = PRHPGProblem(tp, n=n)
    sf.set_stage_cost(Q=Q, R=R)
    sf.set_initial_covariance(Sigma_0)
    sf.set_horizon(N=N_ref)
    K_0_flat, _ = sf.solve()
    kappa_star = -K_0_flat                                          # (R̃, m, n)

    # dual
    obs = ObserverPRHPGProblem(tp=tp, C=C_mid, n=n)
    obs.set_noise_covariances(W=W, V=V_meas)
    obs.set_horizon(N=N_ref)
    L_flat, _ = obs.solve()
    lambda_star = L_flat                                            # (R̃, n, p)

    return kappa_star, lambda_star


def main():
    out_dir = Path("examples/lqg_flipased_demo")
    fig_dir = out_dir / "figures"
    res_dir = out_dir / "results"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"Joint LQG P-QRNPO + separation principle test (channel={CHANNEL})")
    print("=" * 72)

    # build TP form
    poly_full = flipased_polytope(CHANNEL)
    poly_red = flipased_polytope_reduced(CHANNEL)
    poly_red_d = discretize_polytope(poly_red, dt=DEFAULT_DT)
    n = poly_red_d.n
    p = poly_red_d.p

    tp = flipased_tpform(poly_red_d, hull=HULL, gridsize=DEFAULT_GRIDSIZE,
                          verbose=True)

    Q = _default_Q(n, CHANNEL)
    R = DEFAULT_R_CTRL.copy()
    W = W_PROC * np.eye(n)
    V_meas = V_MEAS * np.eye(p)
    Sigma_0 = np.eye(n)

    # extract grid quantities
    A_v, B_v, R_total, m = _extract_vertices(tp.S, n)
    alpha = _evaluate_weights_on_grid(tp.U)
    A_grid, B_grid = _evaluate_AB_on_grid(A_v, B_v, alpha)
    weights = _build_quadrature_weights(tp.U, tp.domain)

    C_mid = poly_red_d.C[poly_red_d.R // 2]
    C_grid = np.tile(C_mid[None, :, :], (alpha.shape[0], 1, 1))

    print(f"\nn={n}, m={m}, p={p}, R̃={R_total}, N_q={alpha.shape[0]}")
    d_k = R_total * m * n
    d_l = R_total * n * p
    d = d_k + d_l
    print(f"joint dim: d = {d_k} + {d_l} = {d}")
    print(f"Hessian size: {d}×{d} (≈ {d*(d+1)//2} unique entries; "
          f"{d} FD calls per Newton iter, but each call needs {d} more for grad)")

    # ---- separated reference ----
    print(f"\n[reference] separated κ*, λ* via P-RHPG at N={HORIZON_FOR_REF} ...")
    kappa_star, lambda_star = reference_separated(tp, n, p, Q, R, W, V_meas,
                                                    Sigma_0, HORIZON_FOR_REF,
                                                    C_mid)
    # evaluate joint LQG cost at separated optimum
    J_star_sep, _, _, _ = lqg_cost(kappa_star, lambda_star, A_grid, B_grid,
                                       C_grid, alpha, weights, Q, R, W, V_meas)
    print(f"  J^LQG at (κ*_primal, λ*_dual) = {J_star_sep:.6e}")
    print(f"  ||κ*||_F = {np.linalg.norm(kappa_star):.4e}")
    print(f"  ||λ*||_F = {np.linalg.norm(lambda_star):.4e}")

    # ---- sanity check: analytic vs FD gradient at the separated optimum ----
    # At the separated optimum, IF separation holds, the joint gradient
    # should be zero. We'll check this AND verify analytic == FD on a
    # nearby point first.
    print(f"\n[gradient sanity check]")
    test_kappa = (1.0 - INIT_PERTURB) * kappa_star
    test_lambda = (1.0 - INIT_PERTURB) * lambda_star
    ga_k, ga_l = lqg_gradient_fd(test_kappa, test_lambda, A_grid, B_grid,
                                    C_grid, alpha, weights, Q, R, W, V_meas)
    # compare to direct FD on cost (slow but trusted)
    eps_check = 1e-5
    R_tilde = test_kappa.shape[0]
    m_dim = test_kappa.shape[1]
    n_dim = test_kappa.shape[2]
    # check just the first 5 entries of grad_kappa for speed
    print(f"  spot-check first 5 entries of ∂J/∂κ (analytic vs FD):")
    flat_k = test_kappa.copy().reshape(-1)
    J0, _, _, _ = lqg_cost(test_kappa, test_lambda, A_grid, B_grid, C_grid,
                              alpha, weights, Q, R, W, V_meas)
    for j in range(min(5, flat_k.size)):
        flat_p = flat_k.copy()
        flat_p[j] += eps_check
        kappa_p = flat_p.reshape(test_kappa.shape)
        Jp, _, _, _ = lqg_cost(kappa_p, test_lambda, A_grid, B_grid, C_grid,
                                  alpha, weights, Q, R, W, V_meas)
        fd_val = (Jp - J0) / eps_check
        an_val = ga_k.reshape(-1)[j]
        rel = abs(fd_val - an_val) / max(abs(fd_val), 1e-10)
        print(f"    κ[{j}]: FD = {fd_val:+.4e}, analytic = {an_val:+.4e}, "
              f"rel err = {rel:.2e}")

    # also check first 5 of grad_lambda
    print(f"  spot-check first 5 entries of ∂J/∂λ (analytic vs FD):")
    flat_l = test_lambda.copy().reshape(-1)
    for j in range(min(5, flat_l.size)):
        flat_p = flat_l.copy()
        flat_p[j] += eps_check
        lam_p = flat_p.reshape(test_lambda.shape)
        Jp, _, _, _ = lqg_cost(test_kappa, lam_p, A_grid, B_grid, C_grid,
                                  alpha, weights, Q, R, W, V_meas)
        fd_val = (Jp - J0) / eps_check
        an_val = ga_l.reshape(-1)[j]
        rel = abs(fd_val - an_val) / max(abs(fd_val), 1e-10)
        print(f"    λ[{j}]: FD = {fd_val:+.4e}, analytic = {an_val:+.4e}, "
              f"rel err = {rel:.2e}")

    # ---- initialize: perturbed-from-separated, both κ and λ ----
    kappa_0 = (1.0 - INIT_PERTURB) * kappa_star
    lambda_0 = (1.0 - INIT_PERTURB) * lambda_star

    ok_init, rho_init = is_lqg_stabilizing(kappa_0, lambda_0, A_grid, B_grid,
                                              C_grid, alpha)
    print(f"\n[init] (1-{INIT_PERTURB})·(κ*, λ*), max ρ(A_aug) = {rho_init:.4f}, "
          f"stabilizing = {ok_init}")
    if not ok_init:
        raise ValueError("init not stabilizing; reduce INIT_PERTURB")

    # ---- run joint P-QRNPO ----
    print(f"\n[joint P-QRNPO] running {MAX_ITER_LQG} iterations ...")
    print(f"  (each iter does 1 grad call (d FD evals) + d grad calls for "
          f"Hessian = {d * (d + 1)} cost evaluations per Newton iter)")
    kappa_final, lambda_final, history = joint_p_qrnpo(
        kappa_0, lambda_0, A_grid, B_grid, C_grid, alpha, weights,
        Q, R, W, V_meas, max_iter=MAX_ITER_LQG)

    # ---- final stats ----
    J_final = history["J"][-1]
    rel_J = J_final / J_star_sep - 1.0
    rel_dist_kappa = float(np.linalg.norm(kappa_final - kappa_star)
                              / np.linalg.norm(kappa_star))
    rel_dist_lambda = float(np.linalg.norm(lambda_final - lambda_star)
                               / np.linalg.norm(lambda_star))

    print(f"\n{'='*72}")
    print(f"Separation principle check:")
    print(f"{'='*72}")
    print(f"  J^LQG_final / J^LQG_separated - 1 = {rel_J:.4e}")
    print(f"  ||κ_final - κ*_primal|| / ||κ*||  = {rel_dist_kappa:.4e}")
    print(f"  ||λ_final - λ*_dual||  / ||λ*||  = {rel_dist_lambda:.4e}")
    if rel_dist_kappa < 0.05 and rel_dist_lambda < 0.05:
        print(f"  -> SEPARATION HOLDS empirically: joint LQG optimum matches "
              f"separated primal+dual.")
    else:
        print(f"  -> SEPARATION DOES NOT HOLD: joint optimum differs from "
              f"separated.")

    # plot
    iters = np.arange(len(history["J"]))
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    ax = axes[0]
    ax.semilogy(iters, np.abs(np.array(history["J"]) - J_star_sep), "-o",
                color=COLORS["PO"], lw=1.2, ms=4,
                label=r"$|J^{\mathrm{LQG}} - J^{\mathrm{LQG}}_{\mathrm{sep}}|$")
    ax.set_ylabel("LQG suboptimality")
    ax.set_title(rf"FliPASED [{CHANNEL}] joint LQG P-QRNPO "
                 rf"(separation principle test)")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in")

    ax = axes[1]
    ax.semilogy(iters, history["grad_norm"], "-o",
                color=COLORS["PO"], lw=1.2, ms=4)
    ax.set_ylabel(r"$\|\nabla J^{\mathrm{LQG}}\|$")
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
    out_path = fig_dir / f"p_qrnpo_lqg_{CHANNEL}.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out_path.name}")

    np.savez(res_dir / f"p_qrnpo_lqg_{CHANNEL}.npz",
              J_hist=np.array(history["J"]),
              grad_norm=np.array(history["grad_norm"]),
              step_size=np.array(history["step_size"]),
              rho_max=np.array(history["rho_max"]),
              kappa_star=kappa_star, lambda_star=lambda_star,
              kappa_final=kappa_final, lambda_final=lambda_final,
              J_star_separated=J_star_sep)
    print(f"saved results/p_qrnpo_lqg_{CHANNEL}.npz")


if __name__ == "__main__":
    main()
