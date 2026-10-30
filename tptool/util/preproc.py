"""
Pre/post-processing utilities for LPV models and TP cores.

Ported from MATLAB TP Tool:
  util/scale_core.m, util/scale_lpv.m, util/shift_core.m, util/shift_lpv.m.

These let you rescale/recenter the entries of an LPV cell-array model
before transformation, or apply matching transformations to a TP core
tensor afterward.
"""

import numpy as np


def scale_core(D, v):
    """
    Scale each (Sy, Sx) entry of a tensor by a corresponding scalar.

    Parameters
    ----------
    D : ndarray
        Multidimensional array. The last len(v.shape) axes (or just the
        last axis if v is 1D) are treated as the (Sy, Sx) "system matrix"
        axes; D's leading axes are vertex/grid axes.
    v : array_like
        Scale tensor of shape matching D's trailing axes (typically
        shape (Sy, Sx)).

    Returns
    -------
    S : ndarray, same shape as D
        Each element of D's trailing slab multiplied by the corresponding
        element of v.

    Notes
    -----
    Mirrors MATLAB scale_core. Reshapes D into (r, n) where n = numel(v),
    multiplies each row by v, then reshapes back.
    """
    D = np.asarray(D, dtype=float)
    v = np.asarray(v, dtype=float)
    n = v.size
    r = D.size // n
    flat = D.reshape(r, n)
    flat = flat * v.ravel()[None, :]
    return flat.reshape(D.shape)


def scale_lpv(D, v):
    """
    Scale each entry of a sampled LPV cell-array.

    Parameters
    ----------
    D : list of lists of ndarrays, shape (Sy, Sx)
        Sampled LPV model.
    v : ndarray, shape (Sy, Sx)
        Scale factor for each entry.

    Returns
    -------
    S : list of lists of ndarrays
        Each entry D[i][j] multiplied by v[i, j].

    Notes
    -----
    Mirrors MATLAB scale_lpv. Returns a fresh nested list rather than
    modifying in place.
    """
    v = np.asarray(v, dtype=float)
    Sy = len(D)
    if Sy == 0:
        return []
    Sx = len(D[0])

    if v.shape != (Sy, Sx):
        raise ValueError(f"scale_lpv: v shape {v.shape} does not match D shape ({Sy}, {Sx})")

    return [[np.asarray(D[i][j], dtype=float) * v[i, j] for j in range(Sx)] for i in range(Sy)]


def shift_core(D, v):
    """
    Shift each (Sy, Sx) entry of a tensor by a corresponding scalar.

    Parameters
    ----------
    D : ndarray
        Multidimensional array.
    v : array_like
        Shift tensor whose shape matches D's trailing axes.

    Returns
    -------
    S : ndarray, same shape as D
        Each element of D's trailing slab plus the corresponding element of v.

    Notes
    -----
    Mirrors MATLAB shift_core. Same reshape trick as scale_core but with
    addition instead of multiplication.
    """
    D = np.asarray(D, dtype=float)
    v = np.asarray(v, dtype=float)
    n = v.size
    r = D.size // n
    flat = D.reshape(r, n)
    flat = flat + v.ravel()[None, :]
    return flat.reshape(D.shape)


def shift_lpv(D, v):
    """
    Shift each entry of a sampled LPV cell-array by a constant.

    Parameters
    ----------
    D : list of lists of ndarrays, shape (Sy, Sx)
    v : ndarray, shape (Sy, Sx)
        Shift to add to each entry.

    Returns
    -------
    S : list of lists of ndarrays
        Each entry D[i][j] plus v[i, j].
    """
    v = np.asarray(v, dtype=float)
    Sy = len(D)
    if Sy == 0:
        return []
    Sx = len(D[0])

    if v.shape != (Sy, Sx):
        raise ValueError(f"shift_lpv: v shape {v.shape} does not match D shape ({Sy}, {Sx})")

    return [[np.asarray(D[i][j], dtype=float) + v[i, j] for j in range(Sx)] for i in range(Sy)]
