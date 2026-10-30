"""
Core tensor computation for the polytopic representation.

Ported from MATLAB TP Tool: lpv/coretensor.m and lpv/dep2idx.m.

Given the basis functions {U_k} from HOSVD and the original sampled cell
array {D[i, j]}, computes the core tensor S of the polytopic representation
such that the LPV model can be reconstructed as
    A(p) ≈ sum over vertex tuple v of  (prod_k W_k(p)[v_k])  *  S[v_1, ..., v_P, i, j]
where W_k = U_k after hull manipulation.
"""

import numpy as np

from tptool.array import ndim_expand, ndim_fold, ndim_unfold


def dep2idx(dep):
    """
    Convert a 0/1 dependency mask to per-parameter index labels.

    Parameters
    ----------
    dep : ndarray, shape (Sy, Sx, P) or (Sy, P)
        Dependency mask. dep[i, j, k] = 1 means cell (i, j) depends on
        parameter k.

    Returns
    -------
    depidx : ndarray, same shape as dep
        For each (i, j), the entries of depidx[i, j, :] are 0 for non-
        dependent params, and 1, 2, 3, ... (in order) for dependent ones,
        labeling which dimension of the cell's tensor corresponds to which
        parameter.

    Examples
    --------
    >>> dep = np.array([[[1, 0, 1]]])  # cell (0,0) depends on params 0 and 2
    >>> dep2idx(dep)
    array([[[1, 0, 2]]])  # param 0 -> dim 1 (1-indexed), param 2 -> dim 2
    """
    dep = np.asarray(dep, dtype=int)
    depidx = np.zeros_like(dep)

    # Handle both (Sy, Sx, P) and (Sy, P) shapes
    if dep.ndim == 2:
        # Vector LPV: shape (Sy, P)
        Sy, P = dep.shape
        for i in range(Sy):
            d_indices = np.where(dep[i, :] > 0)[0]
            for k_label, k in enumerate(d_indices):
                depidx[i, k] = k_label + 1  # MATLAB uses 1-indexed labels
    else:
        Sy, Sx, P = dep.shape
        for i in range(Sy):
            for j in range(Sx):
                d_indices = np.where(dep[i, j, :] > 0)[0]
                for k_label, k in enumerate(d_indices):
                    depidx[i, j, k] = k_label + 1
    return depidx


def coretensor(U_list, data, dep):  # noqa: C901, PLR0912
    """
    Compute the polytopic core tensor.

    Parameters
    ----------
    U_list : list of ndarrays
        Per-parameter basis functions (typically from genhull). U_list[k]
        has shape (gridsize_k, R_k) where R_k is the number of vertices in
        parameter k.
    data : list of lists of ndarrays
        Cell-array of sampled tensors from sampling_lpv. data[i][j] has
        dimensions only over the parameters cell (i, j) depends on.
    dep : ndarray, shape (Sy, Sx, P)
        Dependency mask.

    Returns
    -------
    S : ndarray, shape (R_1, ..., R_P, Sy, Sx)
        Core tensor in the polytopic basis.

    Notes
    -----
    The construction proceeds in two phases:

    1. For each param k and each cell (i, j) depending on k, transform the
       cell's tensor from data-space to weight-coefficient space by
       multiplying along its k-axis with U_k's pseudoinverse (left-inverse).

    2. Assemble the global core tensor S by extending each cell to all P
       parameter dimensions: for non-dependent params, "extend" via outer
       product with U_k's pseudoinverse-row-sum (the coefficient that
       represents the constant function in the U_k basis).
    """
    dep = np.asarray(dep, dtype=int)
    P = dep.shape[-1]

    # Determine matrix shape (Sy, Sx)
    if dep.ndim == 2:
        Sy, _ = dep.shape
        Sx = 1
        is_matrix = False
    else:
        Sy, Sx, _ = dep.shape
        is_matrix = True

    # Use the indexed dependency labels
    depidx = dep2idx(dep)

    # ----- Phase 1: transform each dependent cell to coefficient space -----
    # For each parameter k, compute U_k pseudoinverse and multiply each
    # dependent cell along its k-th-dependency axis.

    UP = [np.linalg.pinv(U_list[k]) for k in range(P)]  # pinv(U_k): (R_k, gridsize_k)
    coeff = [UP[k].sum(axis=1) for k in range(P)]  # length R_k each

    # Make a deep copy of data because we'll modify in place
    data_t = [[None] * Sx for _ in range(Sy)]
    for i in range(Sy):
        for j in range(Sx):
            cell = data[i][j] if is_matrix else data[i]
            data_t[i][j] = np.asarray(cell, dtype=float).copy()

    for k in range(P):
        for i in range(Sy):
            for j in range(Sx):
                # MATLAB: depidx[i, j, k] (1-indexed). Our depidx has the same
                # 1-indexed convention from dep2idx, but we work in 0-indexed
                # axis numbers.
                label = depidx[i, j, k] if is_matrix else depidx[i, k]
                if label == 0:
                    continue  # cell doesn't depend on param k
                axis = label - 1  # convert 1-indexed label to 0-indexed axis
                cell = data_t[i][j]
                lay = ndim_unfold(cell, axis)  # (gridsize_k, rest)
                tmp = UP[k] @ lay  # (R_k, rest)
                new_shape = list(cell.shape)
                new_shape[axis] = UP[k].shape[0]  # = R_k
                data_t[i][j] = ndim_fold(tmp, axis, tuple(new_shape))

    # ----- Phase 2: build the global core tensor S -----
    # Final shape: (R_1, ..., R_P, Sy, Sx)
    R = [U_list[k].shape[1] for k in range(P)]
    S_shape = (*tuple(R), Sy, Sx)
    S = np.zeros(S_shape)

    # Sconst: the "all-constant" core, used for purely constant cells.
    # Sconst[v_1, ..., v_P] = prod_k coeff[k][v_k]
    Sconst = np.array(1.0)
    for k in range(P):
        Sconst = ndim_expand(Sconst, coeff[k])
    # ndim_expand on a scalar returns a 1D vector first, then a matrix, etc.
    # After P calls, Sconst has shape (R_1, ..., R_P).

    for i in range(Sy):
        for j in range(Sx):
            if is_matrix:
                dep_ij = dep[i, j, :]
                cell = data_t[i][j]
            else:
                dep_ij = dep[i, :]
                cell = data_t[i][j]

            if np.any(dep_ij > 0):
                # Cell depends on at least one parameter.
                # Current shape: (R_dep1, R_dep2, ...) — only dependent params.
                # We need to extend to all P parameter dims.

                # Build permutation: dependent dims first (in their current order),
                # then non-dependent dims appended via ndim_expand.

                # Find which params are dependent and which aren't
                dep_params = np.where(dep_ij > 0)[0]  # 0-indexed param indices
                nondep_params = np.where(dep_ij == 0)[0]

                # Extend along each non-dependent param via outer product
                # with the corresponding coeff vector. This appends new
                # trailing axes.
                Sp = cell.copy()
                for k in nondep_params:
                    Sp = ndim_expand(Sp, coeff[k])

                # Now Sp has axes in order: [dep_params (in their natural order),
                # nondep_params (in append order)]. We need to permute to
                # [param 0, param 1, ..., param P-1].

                # Current axis order:
                current_order = list(dep_params) + list(nondep_params)
                # Where does each param k currently live?
                axis_of_param = {k: pos for pos, k in enumerate(current_order)}
                # Build inverse permutation: target axis k <- current axis of param k
                perm = [axis_of_param[k] for k in range(P)]
                Sp = np.transpose(Sp, perm)

                # Now Sp has shape (R_1, ..., R_P)
                S[..., i, j] = Sp
            else:
                # Constant cell: scalar value times Sconst
                # data_t[i][j] is a 0-d or 1-element array
                scalar = float(np.asarray(cell).ravel()[0])
                S[..., i, j] = scalar * Sconst

    return S
