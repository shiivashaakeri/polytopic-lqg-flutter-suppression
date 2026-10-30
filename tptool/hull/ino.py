"""
INO and IRNO hull decompositions.

Ported from MATLAB TP Tool: hull/old/ino.m, hull/old/rnodiff.m, hull/old/rnoino.m.

INO ("Interior NO"):
  Given an SN matrix U (rows sum to 1, m x r), construct an enclosing
  simplex such that the resulting weight matrix has the property that
  each column reaches 0 at some sample (the "Interior" property: the
  data sits strictly inside the simplex with at least one weight-zero
  point per face).

  Closed-form: the envelope simplex's vertices are placed by reflecting
  the row-min vector across each coordinate axis.

IRNO ("RNO + INO"):
  An iterative procedure that combines the RNO property (every column's
  maximum is the same value, between 0 and 1) with INO. Reduces dimension
  one at a time via projection onto lower simplices, using bisection at
  each step to find the parameter that satisfies the RNO max-uniformity
  condition.

Both algorithms assume the input matrix already has the SN property
(rows sum to 1). They are typically called as a post-processing step
after snnn_basis().
"""

import numpy as np


def ino(U):
    """
    INO decomposition of an SN matrix.

    Parameters
    ----------
    U : ndarray, shape (m, r)
        SN matrix (rows sum to 1), m > r.

    Returns
    -------
    W : ndarray, shape (m, r)
        Weight matrix. SN, NN, and INO (each column has minimum 0).
    V : ndarray, shape (r, r)
        Vertex matrix. SN. W @ V = U.
    """
    U = np.asarray(U, dtype=float)
    m, r = U.shape

    # MATLAB code: zeros out the last column of U temporarily for "egy" computation.
    # "egy" is the equation/half-space describing the envelope simplex's faces.
    U_work = U.copy()
    U_work[:, -1] = 0.0

    # egy[j] for j < r-1: minimum of column j (over rows) — face equation /1
    # egy[r-1]: max of row sums of U_work
    egy = np.empty(r)
    egy[: r - 1] = U_work.min(axis=0)[: r - 1]
    egy[r - 1] = U_work.sum(axis=1).max()

    # Construct the envelope simplex vertices "cs":
    # Start with each row = egy[:r-1], then offset diagonally:
    #   cs[a, a] = 2*egy[r-1] + egy[a] - sum(egy)   for a = 0 .. r-2
    cs = np.tile(egy[: r - 1], (r, 1))  # shape (r, r-1)
    for a in range(r - 1):
        cs[a, a] = 2 * egy[r - 1] + egy[a] - egy.sum()

    # Append the r-th column to keep cs SN: cs[:, r-1] = 1 - sum(cs[:, :r-1])
    last_col = 1.0 - cs.sum(axis=1, keepdims=True)
    cs_full = np.hstack([cs, last_col])  # shape (r, r)

    # Interpolation matrix: W = U @ inv(cs_full)
    W = U @ np.linalg.inv(cs_full)
    V = cs_full
    return W, V


def rnodiff(U, c):
    """
    Helper for rnoino bisection: returns a scalar whose sign indicates
    whether the parameter c is too small, too large, or just right for
    the RNO property at the current iteration.

    Parameters
    ----------
    U : ndarray, shape (n, r)
    c : float

    Returns
    -------
    d : float
        d > 0  --> c is too large
        d < 0  --> c is too small
        d == 0 --> exactly right

    Algorithm (mirrors MATLAB rnodiff.m):
      U0 = U[:, :-1]
      s = row sums of U0
      U1 = U0 + (c-1)/r * s * ones (shifts toward / away from centroid)
      U2 = U1 - row_min(U1)         (translate so column min is 0)
      U3 = U2 / row_max(U2)         (normalize so column max is 1)
      U4 = U3 / max(row sums of U3) (re-scale)
      d  = 1 - min(row sums of U4) - max(U4[:, 0])
    """
    U = np.asarray(U, dtype=float)
    U0 = U[:, :-1]
    n, r = U0.shape

    s = U0.sum(axis=1)  # length n
    U1 = U0 + ((c - 1.0) / r) * s[:, None] * np.ones((1, r))  # shape (n, r)
    U2 = U1 - U1.min(axis=0)[None, :]  # column-wise shift

    col_max = U2.max(axis=0)
    # Guard against zero column max (would divide by zero).
    col_max_safe = np.where(col_max > 0, col_max, 1.0)
    U3 = U2 / col_max_safe[None, :]

    row_sums_U3 = U3.sum(axis=1)
    max_row_sum = row_sums_U3.max()
    if max_row_sum <= 0:
        max_row_sum = 1.0
    U4 = U3 / max_row_sum

    d = 1.0 - U4.sum(axis=1).min() - U4[:, 0].max()
    return d


def rnoino(U0):
    """
    IRNO decomposition: combined RNO + INO via iterative dimension reduction.

    Parameters
    ----------
    U0 : ndarray, shape (n, r)
        SN matrix (rows sum to 1), n > r.

    Returns
    -------
    W : ndarray, shape (n, r)
        Weight matrix with SN, NN, INO, and RNO properties.
    V : ndarray, shape (r, r)
        Vertex matrix. SN. W @ V = U0.

    Algorithm:
      1. Start with U = U0.
      2. For i = r-1 down to 2:
           Save column i+1 as 'vet[:, i]' (will be re-added later).
           Project U onto the (i)-simplex by absorbing column i+1 into
           the remaining columns: U <- U[:, :i] + vet[:, i] * 1/i.
      3. For i = 2 to r-1 (re-expand):
           Undo projection: U <- U - vet[:, i] * 1/i, then add column to
           restore SN: U <- [U | 1 - sum(U)].
           Bisect on parameter c in some range to find c* such that
           rnodiff(U, c*) = 0.  Apply that c*'s transformation to U.
      4. Compute V = inv(U[:r, :]) @ U0[:r, :].
    """
    U0 = np.asarray(U0, dtype=float)
    n, r = U0.shape
    U = U0.copy()

    # Storage for the projected-out columns at each level.
    # vet[:, i] = the column dropped when reducing from dimension i+1 to i.
    vet = np.zeros((n, r))

    # ----- Phase 1: project down -----
    for i in range(r - 1, 1, -1):
        vet[:, i] = U[:, i]  # save column i (last column at this stage)
        U = U[:, :i]  # drop it
        U = U + (vet[:, i : i + 1] / i) * np.ones((1, i))  # absorb into remaining columns

    # ----- Phase 2: re-expand with bisection on c -----
    for i in range(2, r):
        # Undo absorption from phase 1.
        U = U - (vet[:, i : i + 1] / i) * np.ones((1, i))
        # Re-append the SN-completion column.
        last_col = 1.0 - U.sum(axis=1, keepdims=True)
        U = np.hstack([U, last_col])  # now shape (n, i+1)

        # Bisect on c to find rnodiff(U, c) == 0.
        a_lo, a_hi = 0.0, 1.0
        # Expand a_hi until rnodiff(U, a_hi) >= 0.
        # Safety: cap doublings to avoid infinite loop on pathological inputs.
        for _ in range(60):
            if rnodiff(U, a_hi) >= 0:
                break
            a_hi *= 2.0
        else:
            raise RuntimeError(f"rnoino: failed to bracket c at level i={i} (rnodiff stays negative)")

        d_lo = rnodiff(U, a_lo)
        d_hi = rnodiff(U, a_hi)
        # Bisection until residual is small.
        for _ in range(200):
            if abs(d_lo) + abs(d_hi) < 1e-6:
                break
            mid = 0.5 * (a_lo + a_hi)
            d_mid = rnodiff(U, mid)
            if d_mid > 0:
                a_hi = mid
                d_hi = d_mid
            else:
                a_lo = mid
                d_lo = d_mid
        c_star = 0.5 * (a_lo + a_hi)

        # Apply the c_star transformation: same operations as inside rnodiff,
        # but here we keep the result and proceed to the next level.
        U00 = U[:, :-1]
        s = U00.sum(axis=1)
        U1 = U00 + ((c_star - 1.0) / i) * s[:, None] * np.ones((1, i))
        U2 = U1 - U1.min(axis=0)[None, :]
        col_max = U2.max(axis=0)
        col_max_safe = np.where(col_max > 0, col_max, 1.0)
        U3 = U2 / col_max_safe[None, :]
        max_row_sum = U3.sum(axis=1).max()
        U4 = U3 / max_row_sum if max_row_sum > 0 else U3
        # Re-attach the SN-completion column.
        U = np.hstack([U4, 1.0 - U4.sum(axis=1, keepdims=True)])

    # ----- Compute V -----
    # V = inv(U[:r, :]) @ U0[:r, :]
    V = np.linalg.inv(U[:r, :]) @ U0[:r, :]
    return U, V
