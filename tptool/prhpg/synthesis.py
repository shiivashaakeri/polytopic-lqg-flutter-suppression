"""
Polytopic Receding-Horizon Policy Gradient (P-RHPG) synthesis.

Reference: the CDC 2026 paper (eqs. 9-15 and Algorithm 1).

Mirrors the LMI module's structure:
  - PRHPGProblem class with set_*() methods to configure
  - Functional API: prhpg_struct, prhpg_solve

Per-stage problem (eq. 9): minimize integrated cost-to-go
    Phi_h({K_{i,h}}) = integral over Omega of Tr(P_h(p) Sigma_0) d mu(p)

is a strongly convex quadratic in the vertex gains (Theorem 1).
The minimum is reached either by (a) direct solution of the linear system
(Hessian @ vec(K_h*) = -gradient at zero) or (b) gradient descent.

Author: Shiva Shakeri.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from tptool.lpv.tptrans import TPForm

# Tolerance for convergence-warning checks
DEFAULT_CONV_TOL = 0.01


# ----------------------------------------------------------------------
# Vertex extraction (mirrors LMI module's helper)
# ----------------------------------------------------------------------


def _extract_vertices(S, n):
    """
    Extract vertex matrices (A_i, B_i) from the TP core tensor S.

    The core S has shape (R_1, ..., R_P, n, n+m) where the first P axes
    are vertex axes (one per scheduling parameter). After flattening the
    vertex axes, S becomes a 3D array of shape (R, n, n+m) where R is
    total vertex count.

    Parameters
    ----------
    S : ndarray
        TP core tensor.
    n : int
        State dimension. Used to split (n+m)-column matrix into A and B.

    Returns
    -------
    A_vertices : ndarray, shape (R, n, n)
    B_vertices : ndarray, shape (R, n, m)
    R : int
        Total vertex count.
    m : int
        Input dimension.
    """
    if S.ndim < 3:
        raise ValueError(f"core tensor must be at least 3-D; got shape {S.shape}")

    if S.shape[-2] != n:
        raise ValueError(f"State dim n={n} does not match core tensor's second-to-last axis (size {S.shape[-2]}).")

    n_plus_m = S.shape[-1]
    if n_plus_m <= n:
        raise ValueError(f"Core tensor's last axis is size {n_plus_m}, must exceed n={n}.")
    m = n_plus_m - n

    # Flatten the vertex axes into one
    # S.shape = (R_1, ..., R_P, n, n+m) -> (R_total, n, n+m)
    R = int(np.prod(S.shape[:-2]))
    S_flat = S.reshape(R, n, n + m)

    A_vertices = S_flat[:, :, :n].copy()
    B_vertices = S_flat[:, :, n:].copy()

    return A_vertices, B_vertices, R, m


# ----------------------------------------------------------------------
# Weighting evaluation on the grid
# ----------------------------------------------------------------------


def _evaluate_weights_on_grid(U_list):
    """
    Evaluate the polytopic weighting at every grid point: alpha_i(p_q).

    Given per-parameter weighting matrices U_list[k] of shape
    (gridsize_k, R_k), the joint weighting alpha_i(p_q) at vertex i and
    grid point p_q is the product of per-parameter weights, where i is a
    multi-index over (R_1, ..., R_P) and q is a multi-index over the grid.

    Returns
    -------
    alpha : ndarray, shape (N_q, R)
        alpha[q, i] = alpha_i(p_q), where N_q = product of gridsizes
        and R = product of vertex counts.
    """
    P = len(U_list)
    if P == 0:
        raise ValueError("U_list must contain at least one parameter axis.")

    # Sequentially take outer products of per-axis weights to build the full grid
    # Result has shape (gridsize_1, ..., gridsize_P, R_1, ..., R_P)

    grid_sizes = [u.shape[0] for u in U_list]
    vertex_counts = [u.shape[1] for u in U_list]

    # alpha[q1, ..., qP, r1, ..., rP] = U_list[0][q1, r1] * ... * U_list[P-1][qP, rP]
    alpha = U_list[0]
    for k in range(1, P):
        # Take outer product over both grid and vertex axes
        # Current shape: (g1*...*gk, r1*...*rk) flattened? No, multidim.
        # Simpler: use einsum across all axes at once.
        pass

    # Build with einsum directly. Use letters for grid axes (g_k) and vertex axes (r_k).
    grid_letters = "abcdefgh"[:P]
    vertex_letters = "ijklmnop"[:P]

    if P > 8:
        raise ValueError("Too many scheduling parameters (max 8 supported).")

    # einsum: 'gk,gk,...->grid_letters+vertex_letters'
    inputs = [(u, gl + vl) for u, gl, vl in zip(U_list, grid_letters, vertex_letters)]
    operands = []
    for u, sub in inputs:
        operands.extend([u, sub])
    output_subs = grid_letters + vertex_letters
    operands.append(output_subs)

    # Use np.einsum's string interface
    einsum_str = ",".join(sub for _, sub in inputs) + "->" + output_subs
    arrays = [u for u, _ in inputs]
    big = np.einsum(einsum_str, *arrays)  # shape (g1, ..., gP, r1, ..., rP)

    # Flatten grid axes and vertex axes separately
    N_q = int(np.prod(grid_sizes))
    R = int(np.prod(vertex_counts))
    alpha = big.reshape(N_q, R)

    return alpha


def _evaluate_AB_on_grid(A_vertices, B_vertices, alpha):
    """
    Evaluate A(p_q) and B(p_q) at every grid point.

    A(p_q) = sum_i alpha_i(p_q) A_i, similarly for B.

    Returns
    -------
    A_grid : ndarray, shape (N_q, n, n)
    B_grid : ndarray, shape (N_q, n, m)
    """
    # alpha shape: (N_q, R). A_vertices shape: (R, n, n).
    # Result: einsum 'qi, ijk -> qjk'
    A_grid = np.einsum("qi,ijk->qjk", alpha, A_vertices)
    B_grid = np.einsum("qi,ijk->qjk", alpha, B_vertices)
    return A_grid, B_grid


# ----------------------------------------------------------------------
# Quadrature weights from the grid
# ----------------------------------------------------------------------


def _build_quadrature_weights(U_list, domain):
    """
    Build quadrature weights for the trapezoidal rule on the grid.

    For each parameter axis k with gridsize g_k on interval [a, b]:
      h_k = (b - a) / (g_k - 1)
      w_k[0] = w_k[-1] = h_k / 2 (trapezoidal endpoints)
      w_k[1..-2] = h_k

    Joint weight: w[q] = prod_k w_k[q_k] / volume(Omega)
    so that sum_q w[q] = 1 (normalized for integration with measure mu).

    Returns
    -------
    weights : ndarray, shape (N_q,)
        Weights summing to 1 (the "probability mass" at each grid point).
    """
    P = len(U_list)
    grid_sizes = [u.shape[0] for u in U_list]

    per_axis_weights = []
    total_volume = 1.0
    for k in range(P):
        a, b = domain[k, 0], domain[k, 1]
        g = grid_sizes[k]
        h = (b - a) / (g - 1)
        w = np.full(g, h)
        w[0] *= 0.5
        w[-1] *= 0.5
        per_axis_weights.append(w)
        total_volume *= b - a

    # Tensor product of per-axis weights, then flatten
    big_weights = per_axis_weights[0]
    for w in per_axis_weights[1:]:
        big_weights = np.outer(big_weights, w).ravel()

    # Normalize so they sum to 1 (probability measure)
    big_weights = big_weights / total_volume
    return big_weights


# ----------------------------------------------------------------------
# Per-stage solver: build Hessian/gradient, solve linear system, OR run GD
# ----------------------------------------------------------------------


def _build_M_h(B_grid, P_next_grid, R_mat):
    """M_h(p_q) = R + B(p_q)^T P_{h+1}(p_q) B(p_q)."""
    # B_grid shape (N_q, n, m); P_next_grid shape (N_q, n, n)
    # M_h shape (N_q, m, m)
    BP = np.einsum("qji,qjk->qik", B_grid, P_next_grid)  # B^T P
    BPB = np.einsum("qik,qkl->qil", BP, B_grid)  # B^T P B
    M = R_mat[None, :, :] + BPB
    return M


def _solve_stage_direct(alpha, A_grid, B_grid, P_next_grid, weights, Q_mat, R_mat, Sigma_0, n, m, R_count):  # noqa: ARG001
    """
    Solve the per-stage linear system to get optimal vertex gains.

    From eq. (14), setting gradient to zero:
        sum_j [int alpha_i alpha_j M_h dmu] K_{j,h} = -int alpha_i (B^T P A) dmu

    Sigma_0 cancels (since it appears on both sides as a right-multiplier on
    PD matrices). The system has shape (R*m, R*m) over a flat vector of vec(K)'s,
    OR equivalently can be solved column-by-column on the n columns of K.

    We solve the full (R*m, R*m) system on each of the n columns separately.
    """
    M_h_grid = _build_M_h(B_grid, P_next_grid, R_mat)  # (N_q, m, m)

    # M_int[i, j] = int alpha_i alpha_j M_h dmu, shape (R, R, m, m)
    M_int = np.einsum("q,qi,qj,qkl->ijkl", weights, alpha, alpha, M_h_grid)

    # Build block matrix of shape (R*m, R*m): blocks[i, j] = M_int[i, j]
    H_block = M_int.transpose(0, 2, 1, 3).reshape(R_count * m, R_count * m)

    # RHS: rhs[i] = int alpha_i B^T P A dmu, shape (R, m, n)
    BPA = np.einsum("qji,qjk,qkl->qil", B_grid, P_next_grid, A_grid)  # (N_q, m, n)
    rhs_blocks = np.einsum("q,qi,qjk->ijk", weights, alpha, BPA)  # (R, m, n)
    rhs = -rhs_blocks.reshape(R_count * m, n)  # (R*m, n)

    try:
        K_flat = np.linalg.solve(H_block, rhs)  # (R*m, n)
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Per-stage Hessian solve failed (singular matrix): {e}. "
            "Check that the Gram matrix is positive definite "
            "(weighting functions linearly independent)."
        )

    K_h = K_flat.reshape(R_count, m, n)
    return K_h


def _solve_stage_gradient(
    alpha,
    A_grid,
    B_grid,
    P_next_grid,
    weights,
    Q_mat,  # noqa: ARG001
    R_mat,
    Sigma_0,
    n,
    m,
    R_count,
    K_init=None,
    max_iter=500,
    tol=1e-8,
):
    """
    Solve the per-stage problem via gradient descent.

    Initialize at K_init (default zeros). At each iteration, compute the
    gradient (eq. 14) and step with eta = 1/sigma_max(Hessian).

    Linear convergence at rate (1 - 1/kappa(Hessian)).

    Returns
    -------
    K_h : ndarray, shape (R_count, m, n)
    """
    K_h = np.zeros((R_count, m, n)) if K_init is None else K_init.copy()

    M_h_grid = _build_M_h(B_grid, P_next_grid, R_mat)

    # Estimate step size from largest eigenvalue of the Hessian.
    # Approximate sigma_max using power iteration on a single block as a proxy.
    M_avg = np.einsum("q,qkl->kl", weights, M_h_grid)
    sigma_R = np.linalg.norm(Sigma_0, ord=2)
    sigma_M = np.linalg.norm(M_avg, ord=2)
    # Hessian factor of 2, kron with Sigma_0 multiplies by sigma_R, integral over alpha_i alpha_j gives at most R^2 mass
    eta = 1.0 / (2.0 * sigma_R * sigma_M * R_count + 1e-12)

    for k in range(max_iter):
        # Compute residual E_h(p_q) at all grid points
        # K(p_q, h) = sum_i alpha[q, i] * K_h[i]
        K_grid = np.einsum("qi,ijk->qjk", alpha, K_h)  # (N_q, m, n)
        # E_h(p) = M_h(p) K(p, h) + B^T P_{h+1} A(p)
        MK = np.einsum("qij,qjk->qik", M_h_grid, K_grid)  # (N_q, m, n)
        BPA = np.einsum("qji,qjk,qkl->qil", B_grid, P_next_grid, A_grid)
        E = MK + BPA  # (N_q, m, n)

        # gradient_i = 2 * int[ alpha_i E Sigma_0 ] d mu
        ES = np.einsum("qil,lj->qij", E, Sigma_0)
        grad = 2.0 * np.einsum("q,qi,qjk->ijk", weights, alpha, ES)

        K_new = K_h - eta * grad

        if np.linalg.norm(K_new - K_h) < tol * (np.linalg.norm(K_h) + 1e-12):
            K_h = K_new
            break
        K_h = K_new
    else:
        warnings.warn(
            f"Gradient descent did not converge in {max_iter} iterations "
            f"(final |delta K| = {np.linalg.norm(K_new - K_h):.3e}). "
            "Try increasing max_iter or using direct solver.",
            stacklevel=3,
        )

    return K_h


# ----------------------------------------------------------------------
# Update P_h(p_q) at all grid points (eq. 10)
# ----------------------------------------------------------------------


def _update_P_at_grid(alpha, A_grid, B_grid, P_next_grid, K_h, Q_mat, R_mat):
    """
    Apply eq. (10): P_h(p) = Q + K(p, h)^T R K(p, h) + Acl(p, h)^T P_{h+1}(p) Acl(p, h).

    Evaluated at every grid point.

    Returns
    -------
    P_h_grid : ndarray, shape (N_q, n, n)
    """
    K_grid = np.einsum("qi,ijk->qjk", alpha, K_h)  # (N_q, m, n)
    A_cl = A_grid + np.einsum("qij,qjk->qik", B_grid, K_grid)  # (N_q, n, n)

    # K(p)^T R K(p)
    KRK = np.einsum("qji,jk,qkl->qil", K_grid, R_mat, K_grid)  # (N_q, n, n)
    # Acl^T P_next Acl
    APA = np.einsum("qji,qjk,qkl->qil", A_cl, P_next_grid, A_cl)  # (N_q, n, n)

    return Q_mat[None, :, :] + KRK + APA


def _integrate_cost(P_grid, Sigma_0, weights):
    """Compute J = int Tr(P(p) Sigma_0) d mu(p) via quadrature."""
    # Tr(P Sigma_0) at each grid point: sum over diagonals of P @ Sigma_0
    PS = np.einsum("qij,jk->qik", P_grid, Sigma_0)
    traces = np.einsum("qii->q", PS)  # (N_q,) trace at each grid point
    return float(np.sum(weights * traces))


# ----------------------------------------------------------------------
# Public API: PRHPGProblem
# ----------------------------------------------------------------------


@dataclass
class PRHPGProblem:
    """
    Polytopic Receding-Horizon Policy Gradient (P-RHPG) synthesis.

    Set up by passing a TPForm (output of tptrans), then configure cost,
    horizon, and solver options via set_*() methods, and finally solve().

    Examples
    --------
    >>> tp = tptrans(lpv, dep, domain, gridsize, hull='close')
    >>> problem = PRHPGProblem(tp, n=4)
    >>> problem.set_stage_cost(Q=np.eye(4), R=np.eye(1))
    >>> problem.set_horizon(N=20)
    >>> K, J = problem.solve()
    """

    tp: TPForm
    n: int
    discrete: bool = True  # P-RHPG is discrete-time only (per CDC paper)

    # Cost matrices and horizon (set via methods)
    Q: Optional[np.ndarray] = None
    R: Optional[np.ndarray] = None
    Q_N: Optional[np.ndarray] = None
    Sigma_0: Optional[np.ndarray] = None
    N: int = 1

    # Solver options
    method: str = "direct"  # 'direct' or 'gradient'
    gd_max_iter: int = 500
    gd_tol: float = 1e-8
    conv_tol: float = DEFAULT_CONV_TOL

    # Outputs (populated by solve())
    K_stages: list = field(default_factory=list)  # backward sweep K_{N-1, ..., 0}
    P_stages_grid: list = field(default_factory=list)
    J_history: list = field(default_factory=list)

    # Cached precomputed grid quantities
    _alpha: Optional[np.ndarray] = None
    _A_grid: Optional[np.ndarray] = None
    _B_grid: Optional[np.ndarray] = None
    _weights: Optional[np.ndarray] = None
    _A_vertices: Optional[np.ndarray] = None
    _B_vertices: Optional[np.ndarray] = None
    _R: int = 0
    _m: int = 0

    def __post_init__(self):
        if not self.discrete:
            raise NotImplementedError(
                "P-RHPG is implemented for discrete-time only "
                "(per the CDC 2026 paper). Continuous-time variants "
                "would require a separate derivation."
            )

        # Extract vertex matrices and grid quantities
        self._A_vertices, self._B_vertices, self._R, self._m = _extract_vertices(self.tp.S, self.n)

        self._alpha = _evaluate_weights_on_grid(self.tp.U)
        self._A_grid, self._B_grid = _evaluate_AB_on_grid(self._A_vertices, self._B_vertices, self._alpha)
        self._weights = _build_quadrature_weights(self.tp.U, self.tp.domain)

        # Default Sigma_0 = I
        if self.Sigma_0 is None:
            self.Sigma_0 = np.eye(self.n)

    # ----- Configuration -----

    def set_stage_cost(self, Q, R):
        """Set per-stage state and input cost matrices."""
        Q = np.asarray(Q, dtype=float)
        R = np.asarray(R, dtype=float)
        if Q.shape != (self.n, self.n):
            raise ValueError(f"Q must be ({self.n},{self.n}), got {Q.shape}")
        if R.shape != (self._m, self._m):
            raise ValueError(f"R must be ({self._m},{self._m}), got {R.shape}")
        self.Q = Q
        self.R = R
        return self

    def set_terminal_cost(self, Q_N):
        """Set terminal cost matrix Q_N (default 0 if not called)."""
        Q_N = np.asarray(Q_N, dtype=float)
        if Q_N.shape != (self.n, self.n):
            raise ValueError(f"Q_N must be ({self.n},{self.n}), got {Q_N.shape}")
        self.Q_N = Q_N
        return self

    def set_horizon(self, N):
        """Set horizon length N."""
        if N < 1:
            raise ValueError(f"Horizon N must be >= 1, got {N}")
        self.N = int(N)
        return self

    def set_initial_covariance(self, Sigma_0):
        """Set initial state covariance Sigma_0 (default identity)."""
        Sigma_0 = np.asarray(Sigma_0, dtype=float)
        if Sigma_0.shape != (self.n, self.n):
            raise ValueError(f"Sigma_0 must be ({self.n},{self.n}), got {Sigma_0.shape}")
        self.Sigma_0 = Sigma_0
        return self

    def set_solver(self, method="direct", gd_max_iter=500, gd_tol=1e-8, conv_tol=DEFAULT_CONV_TOL):
        """Configure the per-stage solver: 'direct' or 'gradient'."""
        if method not in ("direct", "gradient"):
            raise ValueError(f"method must be 'direct' or 'gradient', got {method!r}")
        self.method = method
        self.gd_max_iter = gd_max_iter
        self.gd_tol = gd_tol
        self.conv_tol = conv_tol
        return self

    # ----- Solve -----

    def solve(self):
        """
        Run the backward sweep over h = N-1, ..., 0.

        Returns
        -------
        K : ndarray, shape (R, m, n)
            Stage-0 vertex gains (the deployable PDC controller).
        J : float
            Optimal integrated cost J_N^*.
        """
        # Validate setup
        if self.Q is None or self.R is None:
            raise ValueError("Must call set_stage_cost(Q, R) before solve().")
        if self.Q_N is None:
            self.Q_N = np.zeros((self.n, self.n))

        # Initialize P_N(p_q) = Q_N at every grid point
        N_q = self._alpha.shape[0]
        P_grid = np.tile(self.Q_N, (N_q, 1, 1))

        K_stages = []
        P_stages = [P_grid.copy()]
        J_stages = []

        # Backward sweep
        from tptool.prhpg.backward_sweep import backward_sweep  # noqa: PLC0415

        K_stages, P_stages, J_stages = backward_sweep(
            self.N,
            P_grid,
            self._alpha,
            self._A_grid,
            self._B_grid,
            self._weights,
            self.Q,
            self.R,
            self.Sigma_0,
            self.n,
            self._m,
            self._R,
            method=self.method,
            gd_max_iter=self.gd_max_iter,
            gd_tol=self.gd_tol,
        )

        self.K_stages = K_stages
        self.P_stages_grid = P_stages
        self.J_history = J_stages

        # Convergence diagnostic: compare J_0 to J_1 (cost from one less stage)
        if self.N >= 2:
            J0 = J_stages[0]
            J1 = J_stages[1]
            if J0 > 0 and abs(J0 - J1) / J0 > self.conv_tol:
                warnings.warn(
                    f"P-RHPG horizon may be insufficient: "
                    f"|J_N - J_{{N-1}}| / J_N = {abs(J0 - J1) / J0:.3f} > "
                    f"{self.conv_tol:.3f}. Increase N for tighter convergence.",
                    stacklevel=2,
                )

        K_0 = K_stages[0]
        return K_0, J_stages[0]


# ----------------------------------------------------------------------
# MATLAB-style functional API (mirrors LMI module)
# ----------------------------------------------------------------------


def prhpg_struct(tp, n, **kwargs):
    """Functional API: construct a PRHPGProblem (mirrors lmi_struct)."""
    return PRHPGProblem(tp, n=n, **kwargs)


def prhpg_solve(problem):
    """Functional API: solve a PRHPGProblem (mirrors lmi_solve)."""
    return problem.solve()
