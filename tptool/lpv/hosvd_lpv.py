"""
LPV-aware HOSVD for cell-array LPV models.

Ported from MATLAB TP Tool: lpv/hosvd_lpv.m.

Differences from the generic tptool.array.hosvd:
  - Operates on a CELL ARRAY of sampled tensors (one per matrix entry of
    the LPV model), each with potentially different shape (depending on
    which parameters that entry depends on).
  - For each parameter, builds a "stack" combining contributions from
    parameter-dependent and parameter-independent cells.
  - Returns per-parameter basis functions, singular values, and tolerances.

Skips the MATLAB version's interactive prompt for choosing rank: the user
must pass either svtol (truncation tolerance) or keep (per-parameter rank).
"""

import numpy as np

from tptool.array import ndim_unfold, svdtrunc
from tptool.lpv.coretensor import coretensor, dep2idx


def hosvd_lpv(data, dep, gridsize, svtol=1e-8, keep=None):  # noqa: C901, PLR0915, PLR0912
    """
    LPV-aware HOSVD: returns canonical basis and core tensor.

    Parameters
    ----------
    data : list of lists of ndarrays
        Cell-array of sampled tensors from sampling_lpv.
    dep : ndarray
        Dependency mask. Shape (Sy, Sx, P) for matrix LPV or (Sy, P) for vector.
    gridsize : sequence of int, length P
        Grid size per parameter.
    svtol : float, default 1e-8
        Singular value tolerance for truncation.
    keep : sequence of int, length P, optional
        Per-parameter rank to keep. If None, ranks are determined by svtol.
        If keep[k] = 0 or None, parameter k uses svtol-based truncation.

    Returns
    -------
    S : ndarray, shape (R_1, ..., R_P, Sy, Sx)
        Core tensor of the canonical (HOSVD-based) representation.
    U : list of ndarrays
        Per-parameter basis functions. U[k] has shape (gridsize[k], R_k).
    sv : list of ndarrays
        Per-parameter singular values (kept ones).
    tol : list of floats
        Per-parameter discarded-singular-value upper bounds.
    """
    dep = np.asarray(dep, dtype=int)
    P = dep.shape[-1]
    gridsize = list(gridsize)

    if dep.ndim == 2:
        Sy = dep.shape[0]
        Sx = 1
        is_matrix = False
    else:
        Sy, Sx, _ = dep.shape
        is_matrix = True

    if keep is None:
        keep = [0] * P
    keep = list(keep)

    depidx = dep2idx(dep)

    gridprod = int(np.prod(gridsize))

    U = [None] * P
    sv = [None] * P
    tol = [None] * P

    # ---- For each parameter k, build the "stack" and SVD it ----

    for k in range(P):
        Mprod = gridprod // gridsize[k]  # product of all OTHER grid sizes

        stack_cols = []  # to be hstacked into a (gridsize[k], cols) matrix
        c_const = 0.0  # accumulator for constant contributions

        for i in range(Sy):
            for j in range(Sx):
                if is_matrix:
                    cell = data[i][j]
                    dep_ij = dep[i, j, :]
                    label = depidx[i, j, k]
                else:
                    cell = data[i]
                    dep_ij = dep[i, :]
                    label = depidx[i, k]

                cell = np.asarray(cell, dtype=float)
                n_dep = int(np.sum(dep_ij > 0))

                if n_dep > 0:
                    # Cell depends on at least one parameter
                    if label > 0:
                        # Cell depends on parameter k: unfold along its k-axis
                        axis = label - 1
                        lay = ndim_unfold(cell, axis)  # (gridsize[k], rest)

                        if n_dep > 2:
                            # Reduce trailing dimensions to column space first
                            # (equivalent to MATLAB: [u,s] = svd(lay,'econ'); lay = u*s;)
                            u_lay, s_lay, _ = np.linalg.svd(lay, full_matrices=False)
                            lay = u_lay * s_lay[None, :]

                        # Scale by sqrt of product of grid sizes for the
                        # parameters this cell does NOT depend on.
                        # (these "extra dims" are shared across all (i,j) cells)
                        nondep_for_cell = np.where(dep_ij == 0)[0]
                        scale = np.sqrt(float(np.prod([gridsize[p] for p in nondep_for_cell])))
                        stack_cols.append(lay * scale)
                    else:
                        # Cell depends on at least one param, but NOT k.
                        # Add its "energy" to the constant term.
                        # (MATLAB: c = c + sum(data{j,k}(:).^2) * prod(gridsize(dep==0)) / gridsize(i))
                        nondep_for_cell = np.where(dep_ij == 0)[0]
                        nondep_excluding_k = [p for p in nondep_for_cell if p != k]
                        prod_nondep_excl_k = (
                            np.prod([gridsize[p] for p in nondep_excluding_k]) if nondep_excluding_k else 1.0
                        )
                        c_const += float(np.sum(cell**2)) * float(prod_nondep_excl_k)
                else:
                    # Constant cell (depends on no parameter)
                    # (MATLAB: c = c + data{j,k}^2 * Mprod)
                    val = float(np.asarray(cell).ravel()[0])
                    c_const += val * val * Mprod

        # Append constant column: a column of ones * sqrt(c_const)
        if stack_cols:
            stack = np.hstack([*stack_cols, np.ones((gridsize[k], 1)) * np.sqrt(c_const)])
        else:
            stack = np.ones((gridsize[k], 1)) * np.sqrt(c_const)

        # SVD with truncation
        U_k, sv_k, _ = svdtrunc(stack, tol=svtol)

        # Apply user-specified rank cap
        if keep[k] and keep[k] > 0 and keep[k] < len(sv_k):
            U[k] = U_k[:, : keep[k]]
            sv[k] = sv_k[: keep[k]]
            tol[k] = sv_k[keep[k]]
        else:
            U[k] = U_k
            sv[k] = sv_k
            # tolerance: smallest kept singular value... or 0 if we kept everything
            tol[k] = sv_k[-1] if len(sv_k) > 0 else 0.0

    # ---- Compute core tensor with the canonical basis ----
    S = coretensor(U, data, dep)

    return S, U, sv, tol
