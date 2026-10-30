"""
Pretty-print a TP model's vertex LTI systems.

Ported from MATLAB TP Tool: util/printtp.m.
"""

import numpy as np


def printtp(S):
    """
    Print all LTI vertex systems of a TP model.

    The core tensor S has shape (R_1, ..., R_P, Sy, Sx). Each combination
    of (i_1, ..., i_P) over the first P axes selects one vertex's LTI
    system matrix of shape (Sy, Sx). This function flattens and prints them.

    Parameters
    ----------
    S : ndarray
        Core tensor of a TP model.

    Returns
    -------
    LTI : ndarray, shape (Sy, Sx, R_1, ..., R_P)
        The vertex LTI systems with the system-matrix axes moved to the front.
        (Mirrors the MATLAB shiftdim trick: cycle the last two dims to the front.)

    Notes
    -----
    The MATLAB version uses `shiftdim(S, length(n)-2)` which is equivalent to
    moving the last two axes to the front. We do the same and also print
    each vertex's system matrix.
    """
    S = np.asarray(S)
    if S.ndim < 2:
        print(S)
        return S

    # Move last two axes (Sy, Sx) to the front.
    # shiftdim by k cycles axes: in numpy, this is np.moveaxis with axis lists.
    P = S.ndim - 2
    LTI = np.moveaxis(S, [S.ndim - 2, S.ndim - 1], [0, 1])
    # Now shape is (Sy, Sx, R_1, ..., R_P).

    # Print each vertex's LTI system.
    if P == 0:
        # No parameter dimensions; S is already a single LTI matrix.
        print(LTI)
    else:
        vertex_shape = LTI.shape[2:]
        for idx in np.ndindex(*vertex_shape):
            mat = LTI[(slice(None), slice(None), *idx)]
            print(f"Vertex {idx}:")
            print(mat)
            print()

    return LTI
