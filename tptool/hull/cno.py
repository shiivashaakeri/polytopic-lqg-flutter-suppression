"""
Stochastic CNO (Close-to-NO) hull decomposition.

Ported from MATLAB TP Tool: hull/old/cno.m, hull/old/closeness.m,
hull/old/polarorto.m.

Algorithm:
  CNO finds a tight enclosing simplex around an SN matrix U such that the
  resulting hull has the "Close-to-NO" property: each column of the weight
  matrix W = U @ inv(V) reaches close to 1 at some sample.

  For r == 2 (3 vertices from 2 columns of U): the smallest enclosing
  segment trivially has its endpoints at the two extreme rows of U;
  closed-form.

  For r >= 3: the algorithm parameterizes the simplex orientation via
  spherical (polar) coordinates and runs stochastic search:
    1. Random initialization of orientation angles (n_random tries).
    2. For each, run local search (Nelder-Mead) on the closeness objective.
    3. Locally refine the best so far (n_local tries).
    4. Perturb the best (n_perturb tries) and re-refine each.

  The closeness objective combines:
    - Negative log of column maxima (smaller when each column reaches 1)
    - Penalty term on first/last sample max (encourages tightness at the
      parameter range endpoints)

  Vertex placement (polarorto): given simplex orientation angles, inflate
  the simplex from each vertex outward until the opposite face is tangent
  to the data. This is closed-form geometry.
"""

import numpy as np
from scipy.optimize import minimize


def cno(U, h=1.0, hh=5.0, n_random=50, n_perturb=15, n_local=10, rng=None):  # noqa: C901, PLR0915, PLR0912
    """
    Stochastic CNO (Close-to-NO) hull decomposition.

    Parameters
    ----------
    U : ndarray, shape (m, r)
        SN matrix (rows sum to 1), m >= r >= 2.
    h : float, default 1.0
        Norm exponent for the closeness objective. Smaller h pushes a few
        column maxima very close to 1; larger h spreads them more uniformly.
        Recommended range: 0.5 to 5.
    hh : float, default 5.0
        Weight for the endpoint penalty (encourages U[0,:] and U[-1,:]
        to also have a column maximum close to 1). Recommended: 0 to ~r.
    n_random : int, default 50
        Number of random initializations.
    n_perturb : int, default 15
        Number of perturbations of the best solution after local refinement.
    n_local : int, default 10
        Number of local-search iterations per refinement attempt.
    rng : numpy.random.Generator, optional
        For reproducibility. If None, uses default_rng().

    Returns
    -------
    W : ndarray, shape (m, r)
        Weight matrix. SN, NN, and approximately NO (each column max near 1).
    V : ndarray, shape (r, r)
        Vertex matrix. SN. W @ V = U.

    Notes
    -----
    Stochastic algorithm: different random seeds give different (but similar)
    hulls. For deterministic results, fix the seed via the rng parameter.

    For larger r (e.g. r >= 4), increasing n_random improves robustness at
    the cost of runtime. The MATLAB defaults of (50, 15, 10) typically
    converge in seconds for r in {3, 4, 5}; runtime grows quickly with r.
    """
    if rng is None:
        rng = np.random.default_rng()

    U = np.asarray(U, dtype=float)
    m, r = U.shape

    if r > m:
        raise ValueError(f"cno requires m >= r; got m={m}, r={r}")

    # ----- Trivial r == 2 case -----
    # CNO of a 2-column SN matrix is the segment between its row-minima.
    if r == 2:
        # MATLAB: [fi2, i] = min(U1); fi2 = [U1(i(1),:); U1(i(2),:)];
        # The two column minima might be at different rows; pick those rows.
        i_min = np.argmin(U, axis=0)  # length-2 indices
        V = U[i_min, :]  # shape (2, 2)
        # Solve W = U @ inv(V) so that W @ V = U
        W = U @ np.linalg.inv(V)
        return W, V

    # ----- General case r >= 3 -----
    # Step 1: center and whiten U so the optimization is conditioned well.
    # This mirrors the MATLAB preprocessing in cno.m.

    U_first_r_minus_1 = U[:, : r - 1]
    center = U_first_r_minus_1.mean(axis=0, keepdims=True)
    U_centered = U_first_r_minus_1 - center
    inertia = U_centered.T @ U_centered  # (r-1, r-1) PSD matrix

    eigvals, eigvecs = np.linalg.eigh(inertia)
    # Whitening: U_white = U_centered @ V_eig @ diag(eigvals)^{-1/2}
    # so that U_white.T @ U_white = I.
    eps = 1e-12
    inv_sqrt_eig = 1.0 / np.sqrt(np.maximum(eigvals, eps))
    U_white = U_centered @ eigvecs * inv_sqrt_eig[None, :]

    # Append the SN-completion column.
    U_work = np.hstack([U_white, np.ones((m, 1))])
    U_work[:, -1] = 1.0 - U_work[:, :-1].sum(axis=1)

    # ----- Step 2: stochastic search for best orientation -----
    # Polar parameterization: r vertices, each described by (r - 2) spherical
    # angles. Total free parameters: r * (r - 2).

    n_dof = r - 2

    def to_polar_array(flat):
        """Reshape flat parameter vector into (r, r-2) angle matrix."""
        return flat.reshape(r, n_dof)

    def to_flat(polar):
        return polar.ravel()

    def random_polar():
        """Random angles in [-pi/2, pi/2] except the last which is in [-pi, pi]."""
        polar = (rng.random((r, n_dof)) - 0.5) * np.pi
        polar[:, -1] *= 2.0  # last angle in [-pi, pi]
        return polar

    def objective_flat(flat):
        return _closeness(to_polar_array(flat), U_work, h, hh)

    # 2a: random init phase
    best_value = np.inf
    best_polar = None

    for _ in range(n_random):
        x0 = to_flat(random_polar())
        res = minimize(
            objective_flat, x0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 2000}
        )
        if res.fun < best_value:
            best_value = res.fun
            best_polar = res.x.copy()

    # 2b: local refinement of the best so far
    counter = 0
    while counter < n_local:
        res = minimize(
            objective_flat, best_polar, method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-5, "maxiter": 2000}
        )
        if res.fun < best_value - 1e-3:
            best_value = res.fun
            best_polar = res.x.copy()
            counter = 0
        else:
            counter += 1

    # 2c: perturbation phase
    for _ in range(n_perturb):
        perturb = (rng.random((r, n_dof)) - 0.5) * np.pi
        perturb[:, -1] *= 2.0
        x_perturbed = best_polar + 0.1 * rng.random() * perturb.ravel()
        res = minimize(
            objective_flat, x_perturbed, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 2000}
        )
        # Local refinement of the perturbed result
        local_counter = 0
        local_best = res.fun
        local_best_x = res.x.copy()
        while local_counter < n_local:
            res2 = minimize(
                objective_flat,
                local_best_x,
                method="Nelder-Mead",
                options={"xatol": 1e-5, "fatol": 1e-5, "maxiter": 2000},
            )
            if res2.fun < local_best - 1e-3:
                local_best = res2.fun
                local_best_x = res2.x.copy()
                local_counter = 0
            else:
                local_counter += 1
        if local_best < best_value:
            best_value = local_best
            best_polar = local_best_x.copy()

    # ----- Step 3: extract final W, V from best polar parameterization -----
    polar_min = to_polar_array(best_polar)

    # Normalize the polar angles to a canonical range (matches MATLAB code).
    if r > 3:
        for i in range(r):
            for j in range(r - 3):
                if (polar_min[i, j] + np.pi / 2) % (2 * np.pi) > np.pi:
                    polar_min[i, j + 1] = polar_min[i, j + 1] + np.pi
                    polar_min[i, j] = np.pi - polar_min[i, j]
        polar_min[:, r - 3] = polar_min[:, r - 3] % (2 * np.pi)
    polar_min[:, : r - 2] = (polar_min[:, : r - 2] + np.pi / 2) % (2 * np.pi) - np.pi / 2

    # Sort rows for determinism (mirrors sortrows in MATLAB).
    sort_keys = [polar_min[:, k] for k in range(n_dof)]
    sort_idx = np.lexsort(sort_keys[::-1])
    polar_min = polar_min[sort_idx, :]

    # Compute final W, V via polarorto.
    W, V_white = _polarorto(U_work, polar_min)

    # ----- Step 4: recover V in the ORIGINAL (un-whitened) space -----
    # We have W from the optimizer (working on U_work, the whitened+SN-complete
    # version of U). To get V in the original coordinate system, solve
    #     W @ V = U
    # directly via least-squares. This sidesteps any whitening bookkeeping.
    V_full, *_ = np.linalg.lstsq(W, U, rcond=None)

    return W, V_full


# ---------- Internal helpers (closeness, polarorto) ----------


def _closeness(polar, U, h, hh):
    """
    Closeness objective: distance of the polar-parameterized simplex from NO.

    Smaller is better. The objective combines:
      - Norm of log(column maxima of W) — penalizes columns that don't
        reach 1
      - Endpoint penalty: norm of log of column maxima restricted to the
        first and last rows of U

    Parameters
    ----------
    polar : ndarray, shape (r, r-2)
    U : ndarray, shape (m, r), SN
    h : float
        Norm exponent (h-norm).
    hh : float
        Endpoint penalty weight.

    Returns
    -------
    norma : float
    """
    m, r = U.shape
    if polar.shape != (r, r - 2):
        raise ValueError(f"_closeness: polar shape mismatch, expected ({r}, {r - 2}), got {polar.shape}")
    if m < r:
        raise ValueError("_closeness: m must be >= r")

    # Convert polar angles to Cartesian unit vectors on the (r-1)-sphere.
    # MATLAB layout: v[:, i] = sin(a[:, i]) * prod(cos(a[:, :i]), axis=1) for i in 0..r-3
    #                v[:, r-2] = prod(cos(a[:, :]), axis=1)
    #                v[:, r-1] = 1 - sum(v[:, :r-1], axis=1)  (SN-complete)
    a = polar
    v = np.zeros((r, r))
    for i in range(r - 2):
        if i == 0:
            v[:, i] = np.sin(a[:, i])
        else:
            v[:, i] = np.sin(a[:, i]) * np.prod(np.cos(a[:, :i]), axis=1)
    v[:, r - 2] = np.prod(np.cos(a), axis=1)
    v[:, r - 1] = 1.0 - v[:, : r - 1].sum(axis=1)

    # Translate so U[0, :] is inside the simplex.
    centroid = v.mean(axis=0)
    v = v + (U[0:1, :] - centroid[None, :])

    # Inflate from each vertex until the opposite face touches the data.
    for i in range(r):
        # Solve U2 = U @ inv(v) — but v might not be invertible; use lstsq.
        try:
            U2 = U @ np.linalg.inv(v)
        except np.linalg.LinAlgError:
            return 1e10  # degenerate: heavily penalize
        m_min = U2[:, i].min()
        # Pull vertex i and scale opposite face outward.
        v = v - m_min * (v - np.broadcast_to(v[i : i + 1, :], v.shape))
        v[:, -1] = 0.0
        v[:, -1] = 1.0 - v[:, : r - 1].sum(axis=1)

    # Final W
    try:
        U2 = U @ np.linalg.inv(v)
    except np.linalg.LinAlgError:
        return 1e10

    col_max = U2.max(axis=0)
    if np.any(col_max <= 0):
        return 1e10  # degenerate

    norma = np.linalg.norm(np.log(col_max), ord=h)

    # Endpoint penalty: include max over first and last samples
    endpoint_max = np.sort(np.maximum(U2[0, :], U2[-1, :]))
    if endpoint_max[-1] <= 0 or endpoint_max[-2] <= 0:
        return 1e10
    norma += hh * np.linalg.norm(np.log(endpoint_max[-2:]), ord=h)

    return float(norma)


def _polarorto(U, polar):
    """
    Compute (W, V) given simplex orientation in polar coordinates.

    Same geometric construction as _closeness but returning the simplex
    rather than the closeness measure.

    Parameters
    ----------
    U : ndarray, shape (m, r), SN
    polar : ndarray, shape (r, r-2)

    Returns
    -------
    W : ndarray, shape (m, r)
    V : ndarray, shape (r, r)
    """
    m, r = U.shape
    a = polar
    v = np.zeros((r, r))
    for i in range(r - 2):
        if i == 0:
            v[:, i] = np.sin(a[:, i])
        else:
            v[:, i] = np.sin(a[:, i]) * np.prod(np.cos(a[:, :i]), axis=1)
    v[:, r - 2] = np.prod(np.cos(a), axis=1)
    v[:, r - 1] = 1.0 - v[:, : r - 1].sum(axis=1)

    centroid = v.mean(axis=0)
    v = v + (U[0:1, :] - centroid[None, :])

    for i in range(r):
        U2 = U @ np.linalg.inv(v)
        m_min = U2[:, i].min()
        v = v - m_min * (v - np.broadcast_to(v[i : i + 1, :], v.shape))
        v[:, -1] = 0.0
        v[:, -1] = 1.0 - v[:, : r - 1].sum(axis=1)

    W = U @ np.linalg.inv(v)
    return W, v
