"""
SNNN (Sum-Normalized, Non-Negative) hull decomposition.

Ported from MATLAB TP Tool: hull/snnn_decomp.m and hull/snnnbasis.m.

Algorithm:
  Given an orthonormal-column matrix U, find W and V such that
    W @ V = U,  W @ 1 = 1  (rows sum to 1),  W >= 0  (elementwise).

  Steps:
    1. Form X = diag(column sums of U), with a fix for near-zero columns.
    2. Compute U_sn = U @ X. Now rows of U_sn sum to 1 but may be negative.
    3. Find scalars c, d with c + n*d = 1 such that M = c*I + d*ones(n,n)
       maps U_sn to a non-negative matrix W = U_sn @ M while preserving
       the sum-to-one property.
    4. Pick c at the boundary of feasibility:
         c1 = 1 / (1 - n * u_min)   if u_min < 0  (forces W >= 0 from below)
         c2 = 1 / (1 - n * u_max)   if u_max > 2/n (forces W >= 0 from above)
       The trick: M sends the constant 1 to itself (M @ 1 = 1), and adding
       a ones-matrix component shifts every entry uniformly, allowing
       just-barely-non-negative W.
"""

import numpy as np

_SUM_TOL = 1e-5


def snnn_decomp(U):
    """
    SNNN decomposition: return weight matrix W and vertex matrix V
    such that W @ V = U, with W having SN and NN properties.

    Parameters
    ----------
    U : ndarray, shape (m, n)
        Matrix with orthonormal columns (U.T @ U = I).

    Returns
    -------
    W : ndarray, shape (m, n) or (m, n+1)
        Weight matrix. Rows sum to 1, entries non-negative. May have one
        more column than U if U did not span the constant function.
    V : ndarray, shape (n, n) or (n+1, n)
        Vertex matrix satisfying W @ V = U.
    """
    U = np.asarray(U, dtype=float)

    U_sn = _to_sn(U)  # rows sum to 1, possibly with one extra column
    W = _make_nonneg(U_sn)

    # Solve W @ V = U for V (least-squares; for full-rank W this is exact).
    V, *_ = np.linalg.lstsq(W, U, rcond=None)
    return W, V


def snnn_basis(U):
    """
    SNNN basis: return only the weight matrix W, without computing V.

    Parameters
    ----------
    U : ndarray, shape (m, n)

    Returns
    -------
    W : ndarray
        Weight matrix with SN and NN properties.
    """
    U = np.asarray(U, dtype=float)
    U_sn = _to_sn(U)
    return _make_nonneg(U_sn)


def _to_sn(U):
    """
    Step 1+2: transform U to have row sums equal to 1 (SN property),
    appending a column if necessary.

    Logic mirrors the MATLAB code:
      - Compute column sums Usum.
      - Form X = diag(Usum). For near-zero columns, set X[i,i] = X[i,i] + 1
        and X[i, i+/-1] = -1 to keep X invertible while still having
        U @ X have unit row sums in those columns' contribution.
      - If U @ U.T is close to identity (i.e., rows of U have unit norm,
        which happens when U comes from an SVD of a unit-row-sum matrix),
        then U @ X already has the right structure; otherwise we append
        a column 1 - sum(U, axis=1) to give the SN property by construction.
    """
    m, n = U.shape
    U_sum = U.sum(axis=0)  # column sums, length n

    X = np.diag(U_sum.copy())
    if n > 1:
        # Fix near-zero diagonal entries by borrowing from an adjacent column.
        # For n == 1 there is no adjacent column; we leave X as-is and accept
        # that a near-zero column sum will be handled by the singular-warning
        # branch below.
        near_zero = np.abs(U_sum) < _SUM_TOL
        for i in np.where(near_zero)[0]:
            X[i, i] += 1.0
            if i == 0:
                X[i, i + 1] = -1.0
            else:
                X[i, i - 1] = -1.0

    # Check whether 1_m is in the column space of U (i.e., row sums of U @ U.T are 1).
    row_sums_UUT = (U @ U.T).sum(axis=1)
    in_colspace = np.all(np.abs(row_sums_UUT - 1.0) < _SUM_TOL)

    if in_colspace:
        if np.all(np.abs(U_sum) < _SUM_TOL):
            print("[snnn] Warning: SN transformation close to singular, expect errors")
        U_sn = U @ X
    else:
        # Append a column to enforce row-sum-to-1 by construction.
        extra = 1.0 - U.sum(axis=1, keepdims=True)
        U_sn = np.hstack([U, extra])

    return U_sn


def _make_nonneg(U_sn):
    """
    Step 3+4: pick (c, d) such that W = U_sn @ M is non-negative,
    where M = c*I + d*ones(n,n) and c + n*d = 1.

    M satisfies M @ 1 = (c + n*d) * 1 = 1, so W @ 1 = U_sn @ M @ 1 = U_sn @ 1 = 1
    (preserves SN). The remaining freedom in c is used to push entries
    of W non-negative. M acts as: W[i,j] = c * U_sn[i,j] + d * sum(U_sn[i,:]).
    Since sum(U_sn[i,:]) = 1, we have W[i,j] = c * U_sn[i,j] + d * 1
    = c * U_sn[i,j] + (1-c)/n.

    For W[i,j] >= 0 we need c * U_sn[i,j] >= -(1-c)/n, i.e.,
        c * (U_sn[i,j] - 1/n) >= -1/n.

    Tightest constraints come from the most extreme entries of U_sn:
      u_min = min(U_sn, 0)        (most negative or zero)
      u_max = max(U_sn, 2/n)      (most positive or 2/n)

    The two boundary values of c are:
      c1 = 1 / (1 - n * u_min)    (positive; ensures u_min entries >= 0)
      c2 = 1 / (1 - n * u_max)    (negative; ensures u_max entries >= 0 after sign flip)

    The MATLAB code picks: if c1 < -c2 use c2, else use c1.
    """
    m, n = U_sn.shape

    u_min = min(U_sn.min(), 0.0)
    u_max = max(U_sn.max(), 2.0 / n)

    c1 = 1.0 / (1.0 - u_min * n)
    c2 = 1.0 / (1.0 - u_max * n)

    c = c2 if c1 < -c2 else c1
    d = (1.0 - c) / n

    M = d * np.ones((n, n)) + c * np.eye(n)
    return U_sn @ M
