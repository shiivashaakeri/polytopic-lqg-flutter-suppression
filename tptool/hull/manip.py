"""
Hull manipulation: transform an existing hull toward a prescribed target.

Ported from MATLAB TP Tool: hull/hull_manip.m.

Given an existing weight matrix U (SN, NN) and a target matrix Uhat
specifying desired weight values at certain (sample, vertex) entries,
compute the transformation M (square, column-stochastic) such that the
new weights W = U @ M are as close as possible to Uhat in least-squares,
subject to:
  - W >= 0  (NN preserved)
  - column sums of M equal 1  (SN preserved when U is SN)

The vertex transformation in the dual sense is V_new = inv(M) @ V_old.

This is a sibling of opt_decomp:
  - opt_decomp: builds a NEW hull from raw U (n_U + 1 vertices)
  - hull_manip: WARPS an existing hull (same number of vertices)
"""

import numpy as np
from scipy.optimize import LinearConstraint, minimize


def hull_manip(U, Uhat):
    """
    Compute the SN-NN-preserving transformation toward a target hull.

    Parameters
    ----------
    U : ndarray, shape (m, N)
        Existing weight matrix. Should be SN (rows sum to 1) and NN
        (entries non-negative).
    Uhat : ndarray, shape (m, N)
        Target weights. Entries set to -1 are ignored ("don't care").
        Other entries should typically lie in [0, 1].

    Returns
    -------
    M : ndarray, shape (N, N)
        Transformation matrix. New hull's weight matrix is W = U @ M;
        new vertex matrix is V_new = inv(M) @ V_old (where V_old is the
        original hull's vertex matrix).

    Notes
    -----
    Equality constraint: column sums of M equal 1. This ensures that the
    JOINT hull representation (W, V) preserves the underlying data, where
    W = U @ M and V_new = inv(M) @ V_old. Specifically:
        W @ V_new = U @ M @ inv(M) @ V_old = U @ V_old
    so the original system reconstruction is preserved.

    Note: the new W is generally NOT row-sum-to-1 by itself. The hull's
    SN property lives in the (W, V) pair as a whole, not in W alone.
    Use this function only when you have an existing hull (W_old, V_old)
    you want to warp; do not call it on a raw orthonormal U.

    Inequality constraint: U @ M >= 0 elementwise, preserving NN.
    """
    U = np.asarray(U, dtype=float)
    Uhat = np.asarray(Uhat, dtype=float)
    m, N = U.shape

    if Uhat.shape != (m, N):
        raise ValueError(f"Uhat shape mismatch: expected ({m}, {N}), got {Uhat.shape}")

    # Decision variable: vec(M) in column-major (Fortran) order, length N*N.
    # vec(M)[j*N + k] = M[k, j].

    # ---- Objective: ||Uvec @ vec(M) - uhat_active||^2 ----
    # For each (i, j) where Uhat[i, j] != -1:
    #   (U @ M)[i, j] = sum_k U[i, k] * M[k, j]
    # Coefficient on vec(M)[j*N + k] is U[i, k].

    active = Uhat > -1.0
    rows_active, cols_active = np.where(active)
    n_active = len(rows_active)
    uhat_active = Uhat[active]

    Uvec = np.zeros((n_active, N * N))
    for k_active in range(n_active):
        i = rows_active[k_active]
        j = cols_active[k_active]
        Uvec[k_active, j * N : (j + 1) * N] = U[i, :]

    H = Uvec.T @ Uvec
    f = -Uvec.T @ uhat_active

    def fun(x):
        return 0.5 * x @ H @ x + f @ x

    def grad(x):
        return H @ x + f

    # ---- NN constraint: U @ M >= 0 elementwise ----
    # m * N constraint rows.
    A_nn = np.zeros((m * N, N * N))
    for j in range(N):
        for i in range(m):
            row = i + j * m
            A_nn[row, j * N : (j + 1) * N] = U[i, :]

    # ---- SN constraint: column sums of M equal 1 ----
    # That is, sum_k M[k, j] = 1 for each j = 0..N-1.
    # Coefficient on vec(M)[j*N + k] is 1; coefficients for other columns are 0.
    # MATLAB equivalent: Aeq = horzcat({eye(N), eye(N), ...}) gives shape (N, N*N)
    # where row j picks out vec(M)[j*N : (j+1)*N].
    A_eq = np.zeros((N, N * N))
    for j in range(N):
        A_eq[j, j * N : (j + 1) * N] = 1.0
    b_eq = np.ones(N)

    constraints = [
        LinearConstraint(A_nn, 0.0, np.inf),
        LinearConstraint(A_eq, b_eq, b_eq),
    ]

    # Initial guess: identity transform.
    x0 = np.eye(N).flatten(order="F")

    res = minimize(
        fun,
        x0,
        jac=grad,
        method="trust-constr",
        constraints=constraints,
        options={"maxiter": 500, "xtol": 1e-10, "gtol": 1e-8, "verbose": 0},
    )

    if not res.success:
        x_sol = res.x
        nn_min = (A_nn @ x_sol).min()
        eq_err = np.linalg.norm(A_eq @ x_sol - b_eq)
        if nn_min < -1e-6 or eq_err > 1e-6:
            raise RuntimeError(
                f"hull_manip QP did not converge cleanly: status={res.status}, NN min={nn_min:.3e}, EQ err={eq_err:.3e}"
            )

    return res.x.reshape(N, N, order="F")
