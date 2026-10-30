"""
TP controller deployment utility.

Ported from MATLAB TP Tool: util/tpcontroller.m.
"""

import numpy as np

from tptool.array import tprod
from tptool.util.query import queryw1


def tpcontroller(p, x, K, U, domain):
    """
    Compute the control signal u = -K(p) x for a TP-form feedback controller.

    Parameters
    ----------
    p : array_like
        Parameter point at which to evaluate the controller. Length P.
    x : array_like
        State vector. Length n.
    K : ndarray, shape (R_1, ..., R_P, m, n)
        Core tensor of the feedback TP model. The trailing two axes give
        the (m x n) gain matrix; leading axes are vertex indices.
    U : list of ndarrays
        Per-parameter weighting functions of the controller TP model.
    domain : array_like, shape (P, 2)
        Per-parameter domain bounds.

    Returns
    -------
    u : ndarray, shape (m,)
        Control signal at the current state and parameter.

    Notes
    -----
    Feedback convention: u = -K(p) x.

    The blended gain K(p) is obtained by contracting the core tensor K with
    the interpolated weight vectors w_k(p) along each parameter axis k:
        K(p) = sum over (i_1, ..., i_P) of  prod_k w_k[i_k] * K[i_1,...,i_P, :, :]
    """
    p = np.asarray(p, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()

    # Interpolate weights at p
    W = queryw1(U, domain, p)

    # Contract K along each parameter mode via tprod.
    # Each W[k] has shape (R_k,); we pass it as a (1, R_k) matrix so the
    # mode-k product reduces axis k of K to size 1.
    P = len(W)
    factors = [W[k][None, :] for k in range(P)]
    factors_full = factors + [None] * (K.ndim - P)

    K_p_with_singletons = tprod(K, factors_full)

    # Squeeze out the P leading singleton axes from the contractions.
    K_p = np.squeeze(K_p_with_singletons, axis=tuple(range(P)))
    # K_p now has shape (m, n)

    # Apply: u = -K(p) x
    u = -K_p @ x
    return u
