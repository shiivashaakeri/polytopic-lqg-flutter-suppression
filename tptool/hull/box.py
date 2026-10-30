"""
Box hull decomposition: bounding box around the data.

Ported from MATLAB TP Tool: hull/box_decomp.m.

Algorithm:
  Given U of shape (m, n), compute the per-column min and max.
  The hull is the axis-aligned bounding box: 2^n vertices at all
  combinations of (min_j or max_j) for each column j.

  Weights for each row of U are computed via the multilinear
  interpolation formula: each weight is the product of (relative
  position in box) factors along each dimension.

Note: the box hull is NOT a minimal-vertex hull. For n columns it
produces 2^n vertices, which can be much larger than the rank+1
vertices of SNNN. It is useful when one wants vertices aligned with
parameter axes (e.g., for gain-scheduled control where each vertex
corresponds to an extreme operating condition on each parameter).
"""

import numpy as np


def box_decomp(U):
    """
    Box (bounding-box) hull decomposition.

    Parameters
    ----------
    U : ndarray, shape (m, n)
        Data matrix. Columns are typically orthonormal but the algorithm
        works for any U.

    Returns
    -------
    W : ndarray, shape (m, 2**n)
        Weight matrix. Rows sum to 1, entries in [0, 1].
    V : ndarray, shape (2**n, n)
        Vertex matrix: each row is a corner of the bounding box of U.
    """
    U = np.asarray(U, dtype=float)
    m, n = U.shape

    u_min = U.min(axis=0)  # length n
    u_max = U.max(axis=0)  # length n
    L = u_max - u_min  # length n, box widths

    # Guard: zero-width dimensions (constant column) would cause divide-by-zero.
    # Replace with 1 — the corresponding factor will be 1 (or 0) regardless.
    L_safe = np.where(L > 0, L, 1.0)

    n_vertices = 2**n

    # Build vertex matrix V: row i has bit pattern selecting min/max per column.
    # bit 0 of i -> column 0: 0 means u_min[0], 1 means u_max[0], etc.
    V = np.zeros((n_vertices, n))
    for i in range(n_vertices):
        for j in range(n):
            bit = (i >> j) & 1
            V[i, j] = u_max[j] if bit else u_min[j]

    # Build weight matrix W via multilinear interpolation.
    # For each data point u (a row of U) and each vertex i,
    #   W[k, i] = prod over j of  (u_max[j] - U[k,j]) / L[j]   if bit j of i is 0,
    #                             (U[k,j] - u_min[j]) / L[j]   if bit j of i is 1.
    # This is the standard multilinear basis: gives barycentric weights on the box.

    # q_low[k, j] = (u_max[j] - U[k,j]) / L[j]      coefficient for "low" (min) corner in dim j
    # q_high[k, j] = (U[k,j] - u_min[j]) / L[j]     coefficient for "high" (max) corner in dim j
    # For degenerate axes (L[j] == 0), use 0.5 for both (uniform splitting)
    # so the weight along that axis is averaged across both "ends" of the
    # zero-width box. This keeps row sums = 1.
    q_low = np.where(L[None, :] > 0,
                     (u_max[None, :] - U) / L_safe[None, :],
                     0.5)
    q_high = np.where(L[None, :] > 0,
                     (U - u_min[None, :]) / L_safe[None, :],
                     0.5)

    W = np.ones((m, n_vertices))
    for i in range(n_vertices):
        for j in range(n):
            bit = (i >> j) & 1
            W[:, i] *= q_high[:, j] if bit else q_low[:, j]

    return W, V
