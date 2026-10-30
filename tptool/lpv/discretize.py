"""
Zero-order-hold (ZOH) discretization of polytopic LPV systems.

Given a continuous-time TP form (vertices interpreted as continuous-time
matrices), produce a discrete-time TP form by ZOH discretization of each
vertex independently. The weighting functions and grid structure are
unchanged; only the per-vertex matrices A_i, B_i are converted.

ZOH for one vertex (A, B) over a sampling interval dt:
    A_d = exp(A * dt)
    B_d = integral_0^dt exp(A * tau) dtau . B

These can be computed jointly via a single matrix exponential:

    expm( [[A, B], [0, 0]] * dt ) = [[A_d, B_d], [0, I]]

This avoids numerically integrating B_d separately.

The result is exact for systems where the input is held constant over
each interval [k*dt, (k+1)*dt) -- the standard ZOH assumption.

Notes
-----
The TP transform's exactness property degrades slightly under ZOH:
the discretized TP form exactly represents the underlying nonlinear
plant only when (1) the original LPV exactly captures the nonlinear
dynamics and (2) the sampling time dt is small enough that the
parameter does not vary appreciably within one step. For most practical
choices of dt this is negligible.
"""

import numpy as np
from scipy.linalg import expm

from tptool.lpv.tptrans import TPForm


def discretize_tp(tp, dt, n=None):
    """
    ZOH-discretize the vertex matrices of a TP form.

    Parameters
    ----------
    tp : TPForm
        Continuous-time TP form. The core tensor is interpreted as
        ``[A | B]`` per vertex, with state dim ``n`` and input dim
        ``S.shape[-1] - n``.
    dt : float
        Sampling time (seconds). Must be positive.
    n : int, optional
        State dimension. If not given, inferred as the minimum of the
        last two axes of the core tensor (assumes the last axis is
        ``n + m`` and the second-to-last is ``n``).

    Returns
    -------
    tp_d : TPForm
        Discrete-time TP form with the same weighting functions and
        domain. Only the core tensor is changed.

    Examples
    --------
    >>> # Build a continuous-time TP form
    >>> tp_c = tptrans(lpv_c, dep, domain, gridsize, hull='close')
    >>> # Discretize for use with discrete-time synthesis
    >>> tp_d = discretize_tp(tp_c, dt=0.01)
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    S = tp.S

    if n is None:
        # Infer n from the core tensor shape: assume last two dims are (n, n+m)
        n = S.shape[-2]

    if S.shape[-2] != n:
        raise ValueError(f"Core tensor's second-to-last axis is {S.shape[-2]}, expected n = {n}.")

    n_plus_m = S.shape[-1]
    if n_plus_m <= n:
        raise ValueError(f"Core tensor last axis is {n_plus_m}, must exceed n = {n}.")

    # Flatten vertex axes for batched discretization
    R_total = int(np.prod(S.shape[:-2]))
    S_flat = S.reshape(R_total, n, n_plus_m)

    # Discretize each vertex independently
    S_flat_d = np.zeros_like(S_flat)
    for r in range(R_total):
        A = S_flat[r, :, :n]
        B = S_flat[r, :, n:]
        A_d, B_d = _zoh_one_vertex(A, B, dt)
        S_flat_d[r, :, :n] = A_d
        S_flat_d[r, :, n:] = B_d

    # Reshape back to original vertex axes
    S_d = S_flat_d.reshape(S.shape)

    return TPForm(S=S_d, U=tp.U, domain=tp.domain)


def _zoh_one_vertex(A, B, dt):
    """
    ZOH-discretize a single vertex (A, B) over interval dt.

    Uses the augmented-matrix expm trick:
        expm([[A, B], [0, 0]] * dt) = [[A_d, B_d], [0, I]]

    Parameters
    ----------
    A : ndarray, shape (n, n)
    B : ndarray, shape (n, m)
    dt : float

    Returns
    -------
    A_d : ndarray, shape (n, n)
    B_d : ndarray, shape (n, m)
    """
    n = A.shape[0]
    m = B.shape[1]

    # Build augmented matrix
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A
    M[:n, n:] = B
    # Bottom rows are zero

    # Compute exp(M * dt)
    expM = expm(M * dt)

    A_d = expM[:n, :n]
    B_d = expM[:n, n:]

    return A_d, B_d
