"""
Tensor operations: HOSVD, mode-n folding/unfolding, tensor-matrix products.

Ported from the MATLAB TP Tool array/ folder:
  hosvd.m, ndim_fold.m, ndim_unfold.m, svdtrunc.m, tprod.m, tprod1.m, wshift.m
"""

import numpy as np


def ndim_unfold(T, mode):
    """
    Mode-n unfolding of a tensor.

    Reshapes the input tensor T into a 2D matrix whose rows are indexed
    by the chosen mode and whose columns enumerate all other modes.

    Parameters
    ----------
    T : ndarray
        Input tensor of arbitrary order.
    mode : int
        Mode (axis) along which to unfold. 0-indexed.

    Returns
    -------
    M : ndarray, shape (T.shape[mode], prod of other dims)
        Mode-n unfolding of T.

    Notes
    -----
    Convention: column index runs through the other modes in order
    (mode+1, mode+2, ..., N-1, 0, 1, ..., mode-1) with the first
    listed mode varying fastest. This matches the standard Kolda-Bader
    convention.
    """
    T = np.asarray(T)
    N = T.ndim
    if not (0 <= mode < N):
        raise ValueError(f"mode {mode} out of range for tensor of order {N}")

    # Move the chosen mode to the front, then flatten the rest.
    other_modes = [k for k in range(N) if k != mode]
    perm = [mode, *other_modes]
    T_perm = np.transpose(T, perm)
    return T_perm.reshape(T.shape[mode], -1)


def ndim_fold(M, mode, shape):
    """
    Inverse of ndim_unfold: rebuild a tensor of given shape from its
    mode-n unfolding.

    Parameters
    ----------
    M : ndarray, 2D
        Mode-n unfolding.
    mode : int
        Mode along which the unfolding was performed.
    shape : tuple of int
        Target tensor shape.

    Returns
    -------
    T : ndarray of given shape
    """
    M = np.asarray(M)
    N = len(shape)
    if not (0 <= mode < N):
        raise ValueError(f"mode {mode} out of range for shape of order {N}")

    other_modes = [k for k in range(N) if k != mode]
    perm = [mode, *other_modes]
    perm_shape = tuple(shape[p] for p in perm)

    T_perm = M.reshape(perm_shape)

    # Inverse permutation: where does each original axis go?
    inv_perm = [0] * N
    for new_pos, original_pos in enumerate(perm):
        inv_perm[original_pos] = new_pos
    return np.transpose(T_perm, inv_perm)


def svdtrunc(M, tol=None, max_rank=None):
    """
    Truncated SVD.

    Parameters
    ----------
    M : ndarray, 2D
    tol : float, optional
        Singular values below tol * max(sigma) are dropped.
        Default: 1e-12 (essentially numerical rank).
    max_rank : int, optional
        Cap on retained rank, applied after the tolerance filter.

    Returns
    -------
    U : ndarray, shape (m, r)
    sigma : ndarray, shape (r,)
    Vt : ndarray, shape (r, n)
        Truncated factors so that M ≈ U @ diag(sigma) @ Vt.
    """
    if tol is None:
        tol = 1e-12

    U, sigma, Vt = np.linalg.svd(M, full_matrices=False)
    if sigma.size == 0:
        return U, sigma, Vt

    smax = sigma[0]
    keep = sigma > tol * smax
    r = int(np.sum(keep))
    if max_rank is not None:
        r = min(r, max_rank)
    return U[:, :r], sigma[:r], Vt[:r, :]


def hosvd(T, keep_modes=None, tol=None, max_ranks=None):
    """
    Higher-Order Singular Value Decomposition.

    Decomposes a tensor T as
        T = S x_1 U_1 x_2 U_2 x_3 ... x_N U_N
    where S is the core tensor, U_n are orthonormal mode-n factor matrices,
    and x_n denotes the mode-n product.

    Parameters
    ----------
    T : ndarray
        Input tensor of arbitrary order N.
    keep_modes : list of bool, optional
        For each mode, whether to compute the SVD basis (True) or keep
        the mode untouched (False). Default: True for all modes.
        This mirrors the MATLAB hosvd(D, mask) interface where mask=[1 0]
        means "do mode 0, leave mode 1 alone."
    tol : float, optional
        Singular value tolerance for truncation (default: 1e-12).
    max_ranks : list of int, optional
        Per-mode cap on the retained rank.

    Returns
    -------
    S : ndarray
        Core tensor.
    U_list : list of ndarrays
        Factor matrices U_n. For modes where keep_modes[n] is False,
        U_list[n] is None (mode left untouched in S).
    sigma_list : list of ndarrays
        Mode-n singular values. None for skipped modes.

    Notes
    -----
    For each active mode n, the factor U_n is the left-singular-vector
    matrix of the mode-n unfolding of T.
    """
    T = np.asarray(T, dtype=float)
    N = T.ndim

    if keep_modes is None:
        keep_modes = [True] * N
    if len(keep_modes) != N:
        raise ValueError(f"keep_modes must have length {N}, got {len(keep_modes)}")

    if max_ranks is None:
        max_ranks = [None] * N

    U_list = [None] * N
    sigma_list = [None] * N
    S = T.copy()

    for n in range(N):
        if not keep_modes[n]:
            continue
        Mn = ndim_unfold(S, n)
        Un, sigma_n, _ = svdtrunc(Mn, tol=tol, max_rank=max_ranks[n])
        # Update core: contract mode n with U_n^T
        # S_new = S x_n U_n^T  (i.e., apply Un.T to mode n)
        S = mode_n_product(S, Un.T, n)
        U_list[n] = Un
        sigma_list[n] = sigma_n

    return S, U_list, sigma_list


def mode_n_product(T, M, mode):
    """
    Mode-n product of tensor T with matrix M.

    The result has the same shape as T except axis `mode` is replaced
    by M.shape[0] (assuming M is applied as M @ unfold).

    Parameters
    ----------
    T : ndarray
        Input tensor.
    M : ndarray, 2D
        Factor matrix. Number of columns must equal T.shape[mode].
    mode : int
        Mode along which to contract.

    Returns
    -------
    Result : ndarray
    """
    T = np.asarray(T)
    M = np.asarray(M)
    if M.shape[1] != T.shape[mode]:
        raise ValueError(
            f"mode-{mode} product shape mismatch: M has {M.shape[1]} cols, T has {T.shape[mode]} along that axis"
        )

    Tu = ndim_unfold(T, mode)  # (T.shape[mode], rest)
    Pu = M @ Tu  # (M.shape[0], rest)
    new_shape = list(T.shape)
    new_shape[mode] = M.shape[0]
    return ndim_fold(Pu, mode, tuple(new_shape))


def tprod(S, U_list):
    """
    Tensor product: contract a core tensor S with a list of factor matrices.

    Computes S x_1 U_1 x_2 U_2 ... x_N U_N where x_n is the mode-n product.

    Parameters
    ----------
    S : ndarray
        Core tensor of order N.
    U_list : list of (ndarray or None)
        Factor matrices. None entries mean "skip this mode" (identity).

    Returns
    -------
    Result : ndarray
        Reconstructed tensor.
    """
    if len(U_list) != S.ndim:
        raise ValueError(f"tprod: got {len(U_list)} factors but S has order {S.ndim}")

    out = S
    for n, U in enumerate(U_list):
        if U is None:
            continue
        out = mode_n_product(out, U, n)
    return out


def tprod1(S, U, mode):
    """
    Single-mode tensor product: S x_n U.

    Convenience wrapper around mode_n_product matching the MATLAB tprod1.m
    function name.
    """
    return mode_n_product(S, U, mode)


def wshift(M, k):
    """
    Cyclic shift of the rows of a 2D array by k positions.

    MATLAB's wshift(1, M, k) shifts dimension 1 (rows) by k. We expose a
    minimal version: shift rows of M by k (positive = shift down/right
    in the cyclic sense, matching MATLAB convention).

    Parameters
    ----------
    M : ndarray, 2D
    k : int

    Returns
    -------
    M_shifted : ndarray, same shape
    """
    return np.roll(M, k, axis=0)


def ndim_expand(A, v):
    """
    Expand an array in a new (last) dimension by tensor product with a vector.

    Given input A of arbitrary shape and a vector v of length n, returns an
    array B with one extra last dimension of size n, such that
        B[..., i] = v[i] * A

    Parameters
    ----------
    A : ndarray
        Input array of any shape, or a Python scalar.
    v : array_like
        Vector of length n.

    Returns
    -------
    B : ndarray
        If A is a scalar (or 0-d array), B has shape (n,) with B[i] = v[i] * A.
        Otherwise B has shape A.shape + (n,) with B[..., i] = v[i] * A.

    Notes
    -----
    Mirrors MATLAB's array/ndim_expand.m. Used internally by coretensor() to
    extend cell-array tensors along parameter dimensions they don't depend on.
    """
    A = np.asarray(A)
    v = np.asarray(v).ravel()

    if A.ndim == 0:
        # Scalar input: result is just v * A.
        return v * A.item()

    # General case: broadcast A * v along a new trailing axis.
    return A[..., None] * v
