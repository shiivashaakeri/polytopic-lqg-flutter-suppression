"""
LPV model sampling on parameter grids.

Ported from MATLAB TP Tool: lpv/sampling_func.m, sampling_vec.m,
sampling_dep.m, sampling_lpv.m.

Four functions of increasing generality:
  sampling_func — scalar f(p) over an N-dim grid
  sampling_vec  — vector- or matrix-valued f(p) over an N-dim grid
  sampling_dep  — f depends on a SUBSET of parameters; sample only over those
  sampling_lpv  — top-level: cell-array LPV model with per-entry dependencies

The "cell array" structure (a list-of-lists of functions, each with its own
parameter dependencies) is the toolbox's way of representing matrix-valued
LPV systems efficiently — only sample each entry over the parameters it
actually depends on.
"""

import numpy as np


def sampling_func(f, domain, gridsize):
    """
    Sample a scalar function f: R^N -> R on a uniform grid.

    Parameters
    ----------
    f : callable
        Function taking a length-N vector and returning a scalar.
    domain : array_like, shape (N, 2)
        Per-parameter intervals: [[a_1, b_1], ..., [a_N, b_N]].
    gridsize : int or sequence of ints
        Number of grid points per parameter. If a single int, applied to
        all dimensions.

    Returns
    -------
    D : ndarray, shape (gridsize_1, ..., gridsize_N)
        f evaluated on the grid.

    Examples
    --------
    >>> sampling_func(lambda x: x[0] + x[1], [[-1, 1], [0, 3]], [7, 5])
    """
    domain = np.asarray(domain, dtype=float)
    N = domain.shape[0]

    if N == 0:
        return np.asarray(f(np.array([])))

    gridsize = _normalize_gridsize(gridsize, N)

    # Per-axis sample points
    axes = [np.linspace(domain[i, 0], domain[i, 1], gridsize[i]) for i in range(N)]

    # Evaluate on the grid
    D = np.zeros(tuple(gridsize))
    for idx in np.ndindex(*gridsize):
        p = np.array([axes[k][idx[k]] for k in range(N)])
        D[idx] = f(p)
    return D


def sampling_vec(f, domain, gridsize):
    """
    Sample a vector- or matrix-valued function f: R^N -> R^{out_shape} on a grid.

    Parameters
    ----------
    f : callable
        Function taking a length-N vector and returning an array of fixed shape.
    domain : array_like, shape (N, 2)
    gridsize : int or sequence of ints

    Returns
    -------
    D : ndarray, shape (gridsize_1, ..., gridsize_N, *out_shape)
        f evaluated on the grid. The output shape comes last.

    Examples
    --------
    >>> sampling_vec(lambda x: np.array([1, x[0], x[1]]), [[-2, 2], [0, 3]], [3, 5])
    >>> # returns array of shape (3, 5, 3)
    """
    domain = np.asarray(domain, dtype=float)
    N = domain.shape[0]

    if N == 0:
        return np.asarray(f(np.array([])))

    gridsize = _normalize_gridsize(gridsize, N)

    # Probe output shape by evaluating at the first corner
    sample = np.asarray(f(domain[:, 0]))
    out_shape = sample.shape

    axes = [np.linspace(domain[i, 0], domain[i, 1], gridsize[i]) for i in range(N)]

    D = np.zeros(tuple(gridsize) + out_shape)
    for idx in np.ndindex(*gridsize):
        p = np.array([axes[k][idx[k]] for k in range(N)])
        D[idx] = f(p)
    return D


def sampling_dep(f, depidx, domain, gridsize):
    """
    Sample a function that depends on a subset of parameters.

    Used internally when the LPV model's individual entries depend on only
    some of the parameters. Avoids unnecessary work by sampling only over
    the active parameter axes.

    Parameters
    ----------
    f : callable
        Takes a length-P vector (full parameter set), depends only on the
        positions in depidx.
    depidx : array_like of int
        Indices of the dependent parameters (0-indexed). Length P_dep <= P.
    domain : array_like, shape (P_dep, 2)
        Domains of the dependent parameters only.
    gridsize : sequence of int, length P_dep
        Grid sizes for the dependent parameters only.

    Returns
    -------
    D : ndarray, shape (gridsize_1, ..., gridsize_{P_dep})
        f sampled over only the dependent parameter axes.

    Notes
    -----
    The constant entries of the input vector to f (positions not in depidx)
    are set to 0. Since f doesn't depend on them, this doesn't affect output.
    """
    depidx = np.asarray(depidx, dtype=int).ravel()
    domain = np.asarray(domain, dtype=float)
    P_dep = len(depidx)

    if P_dep == 0:
        # f doesn't depend on any parameter — just evaluate once.
        # We need to know how many parameters f expects total; use max(depidx)+1
        # but if depidx is empty we have no info, so pass empty.
        return np.asarray(f(np.array([])))

    gridsize = _normalize_gridsize(gridsize, P_dep)

    # Total parameter count: depidx values are 0-indexed; full vector goes up
    # to max(depidx) + 1, but we don't know if there are higher-index params
    # the caller might also pass. We leave that to sampling_lpv to handle.
    P_full = int(depidx.max()) + 1

    axes = [np.linspace(domain[i, 0], domain[i, 1], gridsize[i]) for i in range(P_dep)]

    # We don't know the output shape of f, probe it
    p_probe = np.zeros(P_full)
    for k, idx in enumerate(depidx):
        p_probe[idx] = domain[k, 0]
    sample = np.asarray(f(p_probe))

    D = np.zeros(tuple(gridsize)) if sample.ndim == 0 else np.zeros(tuple(gridsize) + sample.shape)

    for grid_idx in np.ndindex(*gridsize):
        p = np.zeros(P_full)
        for k, full_idx in enumerate(depidx):
            p[full_idx] = axes[k][grid_idx[k]]
        D[grid_idx] = f(p)
    return D


def sampling_lpv(lpv, dep, domain, gridsize):
    """
    Sample a matrix-valued LPV model with per-entry parameter dependencies.

    Parameters
    ----------
    lpv : list of lists of callables, shape (Sy, Sx)
        Cell array equivalent. lpv[i][j] is the function for entry (i, j).
        Each function takes a length-P vector and returns a scalar.
    dep : ndarray, shape (Sy, Sx, P), 0/1 entries
        Dependency mask. dep[i, j, k] = 1 iff lpv[i][j] depends on parameter k.
    domain : ndarray, shape (P, 2)
        Per-parameter domain.
    gridsize : sequence of int, length P
        Grid size per parameter.

    Returns
    -------
    data : list of lists of ndarrays, shape (Sy, Sx)
        data[i][j] is the sampled tensor for entry (i, j), with dimensions
        only over the parameters that entry actually depends on.

    Notes
    -----
    The cell-array structure is preserved: each entry has its own array shape
    determined by which parameters it depends on. A constant entry returns a
    0-d array; an entry depending on parameters {1, 3} returns a 2-d array
    of shape (gridsize[1], gridsize[3]).
    """
    P = np.asarray(domain).shape[0]
    gridsize = _normalize_gridsize(gridsize, P)
    dep = np.asarray(dep, dtype=int)

    # Determine matrix shape
    if isinstance(lpv, (list, tuple)) and len(lpv) > 0 and isinstance(lpv[0], (list, tuple)):
        Sy, Sx = len(lpv), len(lpv[0])
        is_matrix = True
    else:
        # Vector form: lpv is a flat list, dep is (n, P)
        Sy = len(lpv)
        Sx = 1
        is_matrix = False

    data = [[None] * Sx for _ in range(Sy)]

    for i in range(Sy):
        for j in range(Sx):
            if is_matrix:
                f_ij = lpv[i][j]
                dep_ij = dep[i, j, :]
            else:
                f_ij = lpv[i]
                dep_ij = dep[i, :]

            depidx = np.where(dep_ij > 0)[0]

            if len(depidx) == 0:
                # constant entry — evaluate at any single point
                p_zero = np.zeros(P)
                data[i][j] = np.asarray(f_ij(p_zero))
            else:
                domain_dep = np.asarray(domain)[depidx]
                gridsize_dep = [gridsize[k] for k in depidx]
                data[i][j] = sampling_dep(f_ij, depidx, domain_dep, gridsize_dep)

    if not is_matrix:
        # Flatten to a 1D list-of-arrays
        return [data[i][0] for i in range(Sy)]
    return data


def _normalize_gridsize(gridsize, N):
    """Convert gridsize to a length-N list of ints."""
    gs = np.atleast_1d(np.asarray(gridsize, dtype=int)).ravel()
    if gs.size == 1:
        return [int(gs[0])] * N
    if gs.size != N:
        raise ValueError(f"gridsize must be a scalar or length-{N} vector; got length {gs.size}")
    return [int(x) for x in gs]
