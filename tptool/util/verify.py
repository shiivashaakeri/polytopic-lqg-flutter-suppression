"""
TP model verification utilities.

Ported from MATLAB TP Tool: util/tperror.m.

(util/topsys.m is intentionally not ported — it serializes to MATLAB's
psys format from the Robust Control Toolbox, which has no Python analog.)
"""

import numpy as np

from tptool.util.query import querylpv, querytp


def tperror(lpv, S, U, domain, n=1000, rng=None):
    """
    Estimate the TP model's reconstruction error at random parameter points.

    Samples n random points uniformly in the parameter domain, evaluates
    both the original LPV model and the TP model at each point, and
    returns the maximum and mean L2 reconstruction errors.

    Parameters
    ----------
    lpv : list of lists of callables
        Original LPV model.
    S : ndarray
        TP core tensor.
    U : list of ndarrays
        TP weighting functions.
    domain : array_like, shape (P, 2)
        Parameter domain.
    n : int, default 1000
        Number of random test points.
    rng : numpy.random.Generator, optional
        For reproducibility. If None, uses default_rng().

    Returns
    -------
    max_err : float
        Maximum L2 error across all test points.
    mean_err : float
        Mean L2 error across all test points.

    Notes
    -----
    The "error" at a point p is the Frobenius norm of
        S_lpv(p) - S_tp(p)
    where S_lpv(p) is the original LPV system matrix and S_tp(p) is the
    TP model's interpolated reconstruction. Useful for quantifying how
    much information the polytopic representation lost.
    """
    if rng is None:
        rng = np.random.default_rng()

    domain = np.asarray(domain, dtype=float)
    P = domain.shape[0]

    # Random points uniformly in domain
    span = domain[:, 1] - domain[:, 0]
    x = rng.random((n, P)) * span + domain[:, 0]

    err = np.zeros(n)
    for i in range(n):
        S_lpv = querylpv(lpv, x[i])
        S_tp = querytp(S, U, domain, x[i])
        # Match shapes: querylpv returns (Sy, Sx); querytp returns (Sy, Sx).
        err[i] = np.linalg.norm(S_lpv - S_tp, ord=2)

    return float(err.max()), float(err.mean())
