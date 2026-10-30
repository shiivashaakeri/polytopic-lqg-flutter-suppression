"""
Top-level TP model transformation pipeline.

Ported from MATLAB TP Tool: lpv/tptrans.m.

Combines sampling, HOSVD, hull generation, and core-tensor recomputation
into a single call.
"""

from typing import NamedTuple

import numpy as np

from tptool.hull import genhull
from tptool.lpv.coretensor import coretensor
from tptool.lpv.hosvd_lpv import hosvd_lpv
from tptool.lpv.sampling import sampling_lpv


class TPForm(NamedTuple):
    """Output of a TP transformation: core tensor, weighting functions, domain."""

    S: object
    U: list
    domain: object


def tptrans(lpv, dep, domain, gridsize, hull="close", svtol=1e-6, keep=None,
            hull_seed=0):
    """
    TP model transformation: sample LPV, run HOSVD, generate polytopic hull.

    Parameters
    ----------
    lpv : list of lists of callables
        Cell-array LPV model. lpv[i][j] is a function R^P -> scalar.
    dep : ndarray, shape (Sy, Sx, P) or (Sy, P)
        Dependency mask. dep[i, j, k] = 1 iff entry (i, j) depends on param k.
    domain : array_like, shape (P, 2)
        Per-parameter domain.
    gridsize : sequence of int, length P
        Sampling grid size per parameter.
    hull : str or list of str, default 'close'
        Hull type for the polytopic representation. See tptool.hull.decomp.
    svtol : float, default 1e-6
        Singular value tolerance for HOSVD truncation.
    keep : sequence of int, length P, optional
        Per-parameter rank cap.

    Returns
    -------
    TPForm
        NamedTuple with fields (S, U, domain). Behaves as a regular tuple,
        so existing code that does ``S, U = tptrans(...)`` continues to work
        (the third element is discarded). New code can access ``tp.domain``.

    Examples
    --------
    >>> # Pendulum-like 2-parameter LPV
    >>> lpv = [[lambda p: 1.0 + p[0], lambda p: p[1]],
    ...        [lambda p: -p[0]**2,    lambda p: 1.0 + 0.3*p[1]]]
    >>> dep = np.array([[[1, 0], [0, 1]],
    ...                 [[1, 0], [0, 1]]])
    >>> tp = tptrans(lpv, dep, [[-1, 1], [-1, 1]], [21, 21], hull='close')
    >>> tp.S.shape, [u.shape for u in tp.U], tp.domain.shape
    >>> # Old-style unpacking still works:
    >>> S, U = tptrans(lpv, dep, [[-1, 1], [-1, 1]], [21, 21], hull='close')

    Notes
    -----
    Pipeline:
        1. Sample lpv on the parameter grid (sampling_lpv)
        2. Compute canonical HOSVD basis (hosvd_lpv with svtol truncation)
        3. Generate polytopic hull from canonical basis (genhull)
        4. Recompute core tensor in the polytopic basis (coretensor)
    """
    # 1. Sample
    data = sampling_lpv(lpv, dep, domain, gridsize)

    # 2. Canonical HOSVD basis
    Scan, Ucan, sv, tol = hosvd_lpv(data, dep, gridsize, svtol=svtol, keep=keep)

    # 3. Polytopic hull from canonical basis.
    # hull_seed makes stochastic hulls (cno) REPRODUCIBLE by default;
    # pass hull_seed=None to restore the old nondeterministic behavior.
    rng = np.random.default_rng(hull_seed) if hull_seed is not None else None
    U_list, _ = genhull(Ucan, hull, rng=rng)

    # 4. Core tensor in the polytopic basis
    S = coretensor(U_list, data, dep)

    return TPForm(S=S, U=U_list, domain=np.asarray(domain))
