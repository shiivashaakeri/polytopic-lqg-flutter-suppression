"""
Hull decomposition dispatcher.

Ported from MATLAB TP Tool: hull/decomp.m and hull/genhull.m.

decomp() handles a single matrix; genhull() handles a list (the MATLAB
"cell array" of basis matrices). Both dispatch to the specific hull
algorithms based on a wtype string.

Important preprocessing: for convex hull types, decomp() first centers
the data and projects it onto its affine subspace via SVD before
calling the chosen hull algorithm, then maps the resulting vertices
back. This handles cases where U has linearly dependent columns or
columns that don't span the constant function.
"""

import numpy as np

from tptool.hull.box import box_decomp
from tptool.hull.close import close_decomp
from tptool.hull.cno import cno
from tptool.hull.ino import ino, rnoino
from tptool.hull.snnn import snnn_decomp

_HULL_TYPES_NEEDING_AFFINE_SHIFT = {
    "snnn",
    "hull",
    "close",
    "box",
    "ino",
    "irno",
    "cno",
}

_NON_CONVEX_TYPES = {"eye", "ortho"}


def decomp(U, wtype="close", rng=None):
    """
    Single-matrix hull decomposition.

    Parameters
    ----------
    U : ndarray, shape (m, n)
        Matrix with orthonormal columns (typically from HOSVD).
    rng : numpy.random.Generator, optional
        Passed to stochastic hull algorithms (currently only 'cno').
        None keeps their own defaults.
    wtype : str
        Hull type. Options:
            'eye'   — trivial: W = I, V = U
            'ortho' — pass-through: W = U, V = I (not a hull)
            'snnn' or 'hull' — SNNN convex decomposition
            'box'   — bounding-box hull (2^n vertices)
            'close' — QP-based tight hull (default)
            'ino'   — INO (closed-form)
            'irno'  — IRNO (RNO + INO via bisection)
            'cno'   — stochastic CNO (not yet implemented in this port)

    Returns
    -------
    W : ndarray
        Weight matrix.
    V : ndarray
        Vertex matrix. W @ V = U.

    Raises
    ------
    NotImplementedError
        If wtype is recognized but its algorithm is not yet ported.
    ValueError
        If wtype is unknown.
    """
    U = np.asarray(U, dtype=float)

    if wtype in _NON_CONVEX_TYPES:
        return _decomp_no_shift(U, wtype, rng=rng)

    # --- Affine-subspace preprocessing ---
    # Center U, then SVD to find the actual affine subspace it spans.
    # If rank < n_cols, we operate in the smaller subspace and map back.
    n1, n2 = U.shape
    Ushift = U.mean(axis=0)
    U_centered = U - Ushift

    Uu, Su_vec, VuT = np.linalg.svd(U_centered, full_matrices=False)
    sv = Su_vec
    ns = int(np.sum(sv > 1e-5 * (sv[0] if sv.size else 1.0)))

    if ns == 0:
        raise ValueError("decomp: input has no variation (all rows identical)")

    if ns < n2:
        # Reduce to the affine subspace.
        Uu = Uu[:, :ns]
        Su_diag = np.diag(sv[:ns])
        Vu = VuT[:ns, :].T  # shape (n2, ns), columns are right singular vectors
        U_for_hull = Uu  # work in reduced space
    else:
        # Full rank already; no shift back needed at the end.
        U_for_hull = U

    # --- Run the actual hull algorithm ---
    W, V_reduced = _decomp_no_shift(U_for_hull, wtype, rng=rng)

    # --- Map V back to original space if we reduced ---
    # In reduced space, W @ V_reduced = U_for_hull = Uu.
    # We want to recover V_original such that W @ V_original = U.
    # Note: U = Uu @ Su_diag @ Vu.T + 1 @ Ushift (using outer product),
    #       so V_original = V_reduced @ Su_diag @ Vu.T + 1 @ Ushift.
    V = V_reduced @ Su_diag @ Vu.T + Ushift[None, :] if ns < n2 else V_reduced

    return W, V


def _decomp_no_shift(U, wtype, rng=None):  # noqa: PLR0911
    """
    Inner dispatcher — runs the chosen hull algorithm on U directly,
    without the affine-subspace preprocessing.
    """
    m, n = U.shape

    if wtype == "eye":
        return np.eye(m), U.copy()

    if wtype == "ortho":
        return U.copy(), np.eye(n)

    if wtype in ("snnn", "hull"):
        return snnn_decomp(U)

    if wtype == "box":
        return box_decomp(U)

    if wtype == "close":
        return close_decomp(U)

    if wtype == "ino":
        # INO requires SN input -> first run snnn_decomp, then post-process.
        W_sn, V_sn = snnn_decomp(U)
        W_ino, V_ino_local = ino(W_sn)
        # Compose: W_ino @ V_ino_local = W_sn, and W_sn @ V_sn = U,
        # so W_ino @ (V_ino_local @ V_sn) = U. Final V = V_ino_local @ V_sn.
        V = V_ino_local @ V_sn
        return W_ino, V

    if wtype == "irno":
        W_sn, V_sn = snnn_decomp(U)
        W_irno, V_irno_local = rnoino(W_sn)
        V = V_irno_local @ V_sn
        return W_irno, V

    if wtype == "cno":
        # cno needs SN input -> first run snnn, then post-process.
        W_sn, V_sn = snnn_decomp(U)
        W_cno, V_cno_local = cno(W_sn, rng=rng)
        V = V_cno_local @ V_sn
        return W_cno, V

    raise ValueError(f"unknown wtype: {wtype!r}")


def genhull(U, wtype="close", rng=None):
    """
    Multi-matrix hull decomposition (analog of MATLAB cell-array genhull).

    Parameters
    ----------
    U : ndarray or list of ndarrays
        Either a single matrix (passed through to decomp), or a list of
        matrices to be decomposed independently.
    wtype : str or list of str
        Hull type(s). If a single string, applied to all matrices.
        If a list, must match the length of U.

    Returns
    -------
    W, V : same structure as input
        If U is a single matrix, returns (W, V) tuple.
        If U is a list, returns (list_of_W, list_of_V).

    Notes
    -----
    Empty entries in U (None) are propagated as None in the output.
    """
    if not isinstance(U, list):
        return decomp(U, wtype, rng=rng)

    if isinstance(wtype, str):
        wtype = [wtype] * len(U)

    if len(wtype) != len(U):
        raise ValueError(f"genhull: wtype list length ({len(wtype)}) does not match U list length ({len(U)})")

    W_list = []
    V_list = []
    for Ui, wt in zip(U, wtype):
        if Ui is None or (hasattr(Ui, "size") and Ui.size == 0):
            W_list.append(None)
            V_list.append(None)
        else:
            W, V = decomp(Ui, wt, rng=rng)
            W_list.append(W)
            V_list.append(V)

    return W_list, V_list
