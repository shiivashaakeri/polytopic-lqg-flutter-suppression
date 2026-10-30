"""
"Close" hull decomposition: deterministic tight enclosing hull via QP.

Ported from MATLAB TP Tool: hull/close_decomp.m and hull/opt_decomp.m.

Algorithm (close_decomp):
  Given U of shape (m, n_U), pick n = n_U + 1 "ideal target" rows evenly
  spaced along the data, where each weight column should peak.
  Build target matrix Uhat of shape (m, n): Uhat[idx[j], j] = +1 (peak),
  other entries -1 (don't care). Then call opt_decomp.

  The QP produces the smallest-area triangle (in U-space) tangent to the
  data on all sides. No post-processing is needed: the hull is already
  tight in singular-vector space. When the vertices are projected back
  to data space (via V @ S), the picture may look loose due to asymmetric
  singular-value scaling, but the underlying hull is geometrically tight.

Algorithm (opt_decomp):
  Given U (m x n_U) and Uhat (m x n) with n = n_U + 1, solve a QP:
    minimize   sum_{(i,j): Uhat[i,j] != -1}  ([U_aug @ M][i,j] - Uhat[i,j])^2
    subject to U_aug @ M @ 1 = 1     (sum-normalized weights)
               U_aug @ M >= 0        (non-negative weights)
  where U_aug = [U | 1_m]. Decision variable M is n x n.

  Output:
    W = U_aug @ M           (m x n)
    V = inv(M)[:, :n_U]     so that W @ V = U
"""

import numpy as np
from scipy.optimize import LinearConstraint, minimize


def close_decomp(U):
    """
    "Close" hull decomposition: tight enclosing hull with n_U + 1 vertices.

    Parameters
    ----------
    U : ndarray, shape (m, n_U)

    Returns
    -------
    W : ndarray, shape (m, n_U + 1)
        Weight matrix (SN, NN), with each column peaked near a chosen sample.
    V : ndarray, shape (n_U + 1, n_U)
        Vertex matrix. W @ V = U.
    """
    U = np.asarray(U, dtype=float)
    m, n_U = U.shape
    n = n_U + 1

    if n > m:
        raise ValueError(f"close_decomp requires n_U + 1 <= m; got m={m}, n_U={n_U}")

    # Pick n target peak indices, evenly spaced along the data.
    idx = np.floor(np.linspace(0, m - 1, n)).astype(int)

    # Build Uhat: -1 everywhere ("don't care"), +1 at chosen peaks.
    Uhat = -np.ones((m, n))
    for j, i in enumerate(idx):
        Uhat[i, j] = 1.0

    return opt_decomp(U, Uhat)


def opt_decomp(U, Uhat):
    """
    Optimal decomposition matching prescribed target weights.

    Solve for M (n x n) such that W = [U | 1] @ M is as close as possible
    to Uhat (in least-squares over entries where Uhat != -1), subject to
    SN and NN constraints on W.

    Parameters
    ----------
    U : ndarray, shape (m, n_U)
    Uhat : ndarray, shape (m, n) with n = n_U + 1
        Target weights. Entries set to -1 are ignored ("don't care").

    Returns
    -------
    W : ndarray, shape (m, n)
    V : ndarray, shape (n, n_U) such that W @ V = U
    """
    U = np.asarray(U, dtype=float)
    Uhat = np.asarray(Uhat, dtype=float)
    m, n_U = U.shape
    n = n_U + 1

    if Uhat.shape != (m, n):
        raise ValueError(f"Uhat shape mismatch: expected ({m}, {n}), got {Uhat.shape}")

    # Augment U with a column of ones so [U | 1] @ M can produce ones in any column.
    U_aug = np.hstack([U, np.ones((m, 1))])

    # Decision variable: vec(M) in column-major (Fortran) order, length n*n.
    # vec(M)[j*n + k] = M[k, j], matching the MATLAB convention.

    # ---- Objective: ||Uvec @ vec(M) - uhat_active||^2 ----
    # For each (i, j) where Uhat[i, j] != -1:
    #   (U_aug @ M)[i, j] = sum_k U_aug[i, k] M[k, j]
    # Coefficient on vec(M)[j*n + k] is U_aug[i, k]; coefficients for other
    # columns of M are zero.

    active = Uhat > -1.0
    rows_active, cols_active = np.where(active)
    n_active = len(rows_active)
    uhat_active = Uhat[active]

    Uvec = np.zeros((n_active, n * n))
    for k_active in range(n_active):
        i = rows_active[k_active]
        j = cols_active[k_active]
        Uvec[k_active, j * n : (j + 1) * n] = U_aug[i, :]

    # ||Uvec x - uhat||^2 = x^T (Uvec.T Uvec) x - 2 uhat.T Uvec x + const
    H = Uvec.T @ Uvec
    f = -Uvec.T @ uhat_active

    def fun(x):
        return 0.5 * x @ H @ x + f @ x

    def grad(x):
        return H @ x + f

    # ---- NN constraint: U_aug @ M >= 0 elementwise ----
    # m * n constraint rows.
    A_nn = np.zeros((m * n, n * n))
    for j in range(n):
        for i in range(m):
            row = i + j * m
            A_nn[row, j * n : (j + 1) * n] = U_aug[i, :]

    # ---- SN constraint: U_aug @ M @ 1 = 1 ----
    # For each sample i: sum_j sum_k U_aug[i, k] M[k, j] = 1.
    # Coefficient on vec(M)[j*n + k] is U_aug[i, k] (independent of j).
    # So the constraint matrix has m rows, with U_aug tiled n times across columns.
    A_eq_full = np.tile(U_aug, (1, n))  # (m, n*n)
    b_eq_full = np.ones(m)

    # The m equality constraints have rank only equal to rank(U_aug) <= n.
    # Reduce to a full-row-rank set via SVD to avoid solver issues.
    U_svd, s_svd, _ = np.linalg.svd(A_eq_full, full_matrices=False)
    rank_eq = int(np.sum(s_svd > 1e-10 * s_svd[0]))
    A_eq = U_svd[:, :rank_eq].T @ A_eq_full
    b_eq = U_svd[:, :rank_eq].T @ b_eq_full

    constraints = [
        LinearConstraint(A_nn, 0.0, np.inf),
        LinearConstraint(A_eq, b_eq, b_eq),
    ]

    x0 = np.eye(n).flatten(order="F")

    res = minimize(
        fun,
        x0,
        jac=grad,
        method="trust-constr",
        constraints=constraints,
        options={"maxiter": 500, "xtol": 1e-10, "gtol": 1e-8, "verbose": 0},
    )

    # If solver hits maxiter, verify constraints are still met before erroring.
    if not res.success:
        x_sol = res.x
        nn_min = (A_nn @ x_sol).min()
        eq_err = np.linalg.norm(A_eq @ x_sol - b_eq)
        if nn_min < -1e-6 or eq_err > 1e-6:
            raise RuntimeError(
                f"opt_decomp QP did not converge cleanly: status={res.status}, "
                f"message={res.message}, NN min={nn_min:.3e}, EQ err={eq_err:.3e}"
            )

    M = res.x.reshape(n, n, order="F")
    W = U_aug @ M
    V = np.linalg.inv(M)[:, :n_U]
    return W, V
