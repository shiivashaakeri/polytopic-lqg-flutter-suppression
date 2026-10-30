"""
Query utilities for LPV and TP models.

Ported from MATLAB TP Tool: util/querylpv.m, util/queryw1.m, util/querytp.m.

Three functions:
  querylpv — evaluate an LPV cell-array model at a single parameter point
  queryw1  — interpolate the discretized weighting functions at a single point
  querytp  — evaluate the TP model at a single parameter point (combines the
             core tensor with interpolated weights via mode-n products)
"""

import numpy as np

from tptool.array import tprod


def querylpv(lpv, p):
    """
    Evaluate an LPV cell-array model at a single parameter point.

    Parameters
    ----------
    lpv : list of lists of callables, or list of callables
        The cell-array LPV model. lpv[i][j] (or lpv[i] for a vector model)
        is a function R^P -> scalar.
    p : array_like
        Parameter point, length P.

    Returns
    -------
    S : ndarray, shape (Sy, Sx)
        The system matrix evaluated at p.
    """
    p = np.asarray(p, dtype=float).ravel()

    # Detect matrix vs vector form
    is_matrix = isinstance(lpv, (list, tuple)) and len(lpv) > 0 and isinstance(lpv[0], (list, tuple))

    if is_matrix:
        Sy = len(lpv)
        Sx = len(lpv[0])
        S = np.zeros((Sy, Sx))
        for i in range(Sy):
            for j in range(Sx):
                S[i, j] = lpv[i][j](p)
    else:
        Sy = len(lpv)
        S = np.zeros((Sy, 1))
        for i in range(Sy):
            S[i, 0] = lpv[i](p)

    return S


def queryw1(U, domain, p):
    """
    Interpolate the discretized weighting functions at a single parameter point.

    For each parameter k, the weighting function U[k] is a discrete array of
    shape (gridsize_k, R_k). Given a query parameter point p, this function
    returns the interpolated weight vector for each parameter dimension by
    linear interpolation along the parameter axis.

    Parameters
    ----------
    U : list of ndarrays
        Per-parameter weighting functions. U[k] has shape (gridsize_k, R_k).
    domain : array_like, shape (P, 2)
        Per-parameter domain bounds.
    p : array_like
        Query parameter point, length P.

    Returns
    -------
    W : list of ndarrays
        Interpolated weights. W[k] has shape (R_k,) — a row of weights for
        parameter k at the query point.

    Notes
    -----
    Out-of-domain query points are clamped to the boundary (no extrapolation).
    Linear interpolation is used; higher-order interpolation could be added
    via a method parameter (TODO from the MATLAB version).
    """
    domain = np.asarray(domain, dtype=float)
    p = np.asarray(p, dtype=float).ravel()

    P = domain.shape[0]
    if len(U) != P:
        raise ValueError(f"queryw1: domain has {P} parameters but U has {len(U)} elements")

    W = [None] * P
    for k in range(P):
        U_k = np.asarray(U[k])
        gridsize_k = U_k.shape[0]
        x = np.linspace(domain[k, 0], domain[k, 1], gridsize_k)

        pk = p[k]
        if pk <= x[0]:
            W[k] = U_k[0, :].copy()
        elif pk >= x[-1]:
            W[k] = U_k[-1, :].copy()
        else:
            # Linear interpolation along axis 0 of U_k.
            # np.interp handles 1D, but U_k has multiple columns (vertices),
            # so we interpolate column-wise.
            R_k = U_k.shape[1]
            W_k = np.zeros(R_k)
            for r in range(R_k):
                W_k[r] = np.interp(pk, x, U_k[:, r])
            W[k] = W_k

    return W


def querytp(S, U, domain, p):
    """
    Evaluate a TP model at a single parameter point.

    Parameters
    ----------
    S : ndarray, shape (R_1, ..., R_P, Sy, Sx)
        Core tensor of the TP model.
    U : list of ndarrays
        Per-parameter weighting functions, length P.
    domain : array_like, shape (P, 2)
    p : array_like, length P
        Query parameter point.

    Returns
    -------
    Sp : ndarray, shape (Sy, Sx)
        The system matrix at parameter point p.

    Notes
    -----
    Combines the core tensor with the interpolated weight vectors via
    mode-n products. The first P axes of S are contracted against W[0..P-1];
    the trailing two axes (Sy, Sx) survive as the system matrix shape.
    """
    W = queryw1(U, domain, p)

    # Build factor list for tprod: each W[k] is a row vector that contracts
    # mode k of S. tprod requires factor matrices; pass each W[k] as a 1xR_k
    # matrix so the mode-k product reduces that axis to size 1.
    P = len(W)
    factors = [W[k][None, :] for k in range(P)]
    # The last two axes (Sy, Sx) are not contracted, so we pass None for them.
    factors_full = factors + [None] * (S.ndim - P)

    Sp = tprod(S, factors_full)

    # Squeeze out the singleton dims left from the contractions.
    # First P axes are size 1 after each W[k] contraction.
    Sp = np.squeeze(Sp, axis=tuple(range(P)))

    return Sp
