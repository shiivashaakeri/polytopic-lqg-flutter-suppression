"""
Backward sweep for P-RHPG: Algorithm 1 in the CDC 2026 paper.

For h = N-1, N-2, ..., 0:
    1. Solve {K_{i,h}^*} = argmin Phi_h({K_{i,h}})   (per-stage subproblem)
    2. Update P_h^*(p_q) at every grid point via eq. (10)

Returns the K and P sequences in REVERSE chronological order:
    K_stages[0] = K_0    (deployable, returned to user)
    K_stages[1] = K_1
    ...
    K_stages[N-1] = K_{N-1}

Plus the cost sequence J_h = int Tr(P_h(p) Sigma_0) d mu(p) over h = 0, ..., N.
"""

from tptool.prhpg.synthesis import (
    _integrate_cost,
    _solve_stage_direct,
    _solve_stage_gradient,
    _update_P_at_grid,
)


def backward_sweep(
    N,
    P_terminal_grid,
    alpha,
    A_grid,
    B_grid,
    weights,
    Q_mat,
    R_mat,
    Sigma_0,
    n,
    m,
    R_count,
    method="direct",
    gd_max_iter=500,
    gd_tol=1e-8,
):
    """
    Run the P-RHPG backward sweep.

    Parameters
    ----------
    N : int
        Horizon length (>= 1).
    P_terminal_grid : ndarray, shape (N_q, n, n)
        Terminal cost-to-go P_N(p_q) = Q_N at every grid point.
    alpha : ndarray, shape (N_q, R)
        Joint weighting alpha_i(p_q).
    A_grid : ndarray, shape (N_q, n, n)
        A(p_q) at every grid point.
    B_grid : ndarray, shape (N_q, n, m)
    weights : ndarray, shape (N_q,)
        Quadrature weights summing to 1.
    Q_mat : ndarray, shape (n, n)
    R_mat : ndarray, shape (m, m)
    Sigma_0 : ndarray, shape (n, n)
    n, m : int
    R_count : int
        Total number of vertices.
    method : {'direct', 'gradient'}
    gd_max_iter, gd_tol : passed to gradient solver

    Returns
    -------
    K_stages : list of ndarray, length N
        K_stages[h] is the optimal vertex gain at stage h, shape (R, m, n).
        Indexed forward: K_stages[0] is stage-0 (deployable).
    P_stages : list of ndarray, length N+1
        P_stages[h] is the cost-to-go at stage h evaluated on the grid,
        shape (N_q, n, n). P_stages[N] = P_terminal_grid.
    J_stages : list of float, length N+1
        J_stages[h] = integrated cost-to-go at stage h.
        J_stages[0] is the optimal value J_N^* of the full horizon.
    """
    if N < 1:
        raise ValueError(f"Horizon N must be >= 1, got {N}")

    # Initialize at terminal stage
    P_grid = P_terminal_grid.copy()

    # Allocate stage-indexed lists. We fill them backward (h = N-1 first)
    # but store them with the natural forward index.
    K_stages = [None] * N
    P_stages = [None] * (N + 1)
    J_stages = [None] * (N + 1)

    P_stages[N] = P_grid.copy()
    J_stages[N] = _integrate_cost(P_grid, Sigma_0, weights)

    # Backward iteration: h = N-1, ..., 0
    for h in range(N - 1, -1, -1):
        # Per-stage subproblem
        if method == "direct":
            K_h = _solve_stage_direct(
                alpha,
                A_grid,
                B_grid,
                P_grid,
                weights,
                Q_mat,
                R_mat,
                Sigma_0,
                n,
                m,
                R_count,
            )
        elif method == "gradient":
            K_h = _solve_stage_gradient(
                alpha,
                A_grid,
                B_grid,
                P_grid,
                weights,
                Q_mat,
                R_mat,
                Sigma_0,
                n,
                m,
                R_count,
                K_init=None,
                max_iter=gd_max_iter,
                tol=gd_tol,
            )
        else:
            raise ValueError(f"Unknown method: {method!r}")

        # Apply optimal gains: update cost-to-go at all grid points
        P_grid = _update_P_at_grid(
            alpha,
            A_grid,
            B_grid,
            P_grid,
            K_h,
            Q_mat,
            R_mat,
        )

        K_stages[h] = K_h
        P_stages[h] = P_grid.copy()
        J_stages[h] = _integrate_cost(P_grid, Sigma_0, weights)

    return K_stages, P_stages, J_stages
