"""
Observer P-RHPG synthesis via LQ-Kalman duality.

The dual of polytopic state-feedback LQ design is polytopic Kalman
filtering: the same backward-sweep / Riccati recursion that produces
the optimal LQ gain K, applied to the transposed plant with cost
matrices replaced by noise covariances, produces L^T where L is the
optimal observer gain. Concretely:

    Primal (this is what tptool.prhpg.synthesis does):
      min_K  E[ sum_t x_t^T Q x_t + u_t^T R u_t ]
      s.t.   x_{t+1} = A(p) x_t + B(p) u_t,  u_t = K(p) x_t

    NOTE on sign convention used by tptool.prhpg.synthesis: the solver
    returns K such that A + B K is stable (equivalent to "-K_LQ" in
    the standard u = -Kx convention). The pipeline negates this gain
    before passing to tpcontroller so deployment uses u = -K_dep x.

    Dual (observer; this module):
      For the system  x_{t+1} = A(p) x_t + w_t,  w ~ N(0, W)
                      y_t     = C(p) x_t + v_t,  v ~ N(0, V)
      the optimal observer gain L satisfies the same Riccati
      recursion as the primal with the substitutions
        A  -> A^T
        B  -> C^T
        Q  -> W      (process noise covariance, "state weight" of the dual)
        R  -> V      (measurement noise covariance, "control weight" of the dual)

      The primal solver, run on the dual plant, returns K_dual such
      that A^T + C^T K_dual is stable. We want L such that A - L C
      is stable (standard observer-error dynamics). Setting
        L = -K_dual^T
      gives  A - L C = A + K_dual^T C = (A^T + C^T K_dual)^T,
      which has the same eigenvalues as the dual closed-loop and is
      therefore stable.

    The negation in L = -K_dual^T mirrors the negation pipeline.py
    applies when deploying the primal gain (u = -K_prhpg x).

This file is a thin wrapper around tptool.prhpg.backward_sweep.backward_sweep.
No new algorithm; no new convergence theory; the same per-stage Hessian
conditioning analysis applies on the transposed plant.

The TPForm input is the same one used for state feedback. We extract C
from the LPV plant separately because the TPForm only carries (A, B).

Public API:
    ObserverPRHPGProblem(...)  -- mirrors PRHPGProblem
    observer_struct(...)        -- functional API
    observer_solve(...)         -- functional API
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import warnings

import numpy as np

from tptool.lpv.tptrans import TPForm
from tptool.prhpg.backward_sweep import backward_sweep
from tptool.prhpg.synthesis import (
    DEFAULT_CONV_TOL,
    _build_quadrature_weights,
    _evaluate_AB_on_grid,
    _evaluate_weights_on_grid,
    _extract_vertices,
)


# ----------------------------------------------------------------------
# Helper: extract C(p) on the same grid the primal uses.
#
# The TP form only stores (A | B). The output map C is plant-specific and
# typically constant in p (as in the FliPASED extraction) -- a flat C
# matrix is fine. If C is parameter-varying we need a C grid analogous to
# A_grid/B_grid, evaluated at the same quadrature points.
# ----------------------------------------------------------------------


def _broadcast_C_to_grid(C, N_q):
    """Lift a constant or per-grid-point C into shape (N_q, p, n).

    Accepts:
        C of shape (p, n)            -> tiled to (N_q, p, n)
        C of shape (N_q, p, n)       -> returned unchanged
    """
    C = np.asarray(C, dtype=float)
    if C.ndim == 2:
        return np.tile(C[None, :, :], (N_q, 1, 1))
    if C.ndim == 3 and C.shape[0] == N_q:
        return C
    raise ValueError(
        f"C must be (p, n) or (N_q, p, n) with N_q={N_q}; got shape {C.shape}"
    )


# ----------------------------------------------------------------------
# Public class
# ----------------------------------------------------------------------


@dataclass
class ObserverPRHPGProblem:
    """
    Polytopic LQ-Kalman observer synthesis via dual P-RHPG.

    Parameters
    ----------
    tp : TPForm
        The same polytopic plant used for state-feedback synthesis.
        Provides A(p) and B(p) as a tensor-product polytope. Only A is
        used here (we work on the transposed plant; B does not enter
        observer synthesis).
    C : ndarray
        Output matrix. Shape (p, n) for a parameter-independent C, or
        (N_q, p, n) evaluated on the same quadrature grid as A.
    n : int
        State dimension (must match tp).

    After construction, configure via:
        set_noise_covariances(W, V)
        set_horizon(N)
        set_initial_covariance(Sigma_0_dual)   # optional

    Returns from solve():
        L : ndarray, shape (R, n, p)
            Stage-0 observer gains. To deploy at parameter p:
                L(p) = sum_i alpha_i(p) * L[i]
            Use as: e_{k+1} = (A(p) - L(p) C) e_k
    """

    tp: TPForm
    C: np.ndarray
    n: int
    discrete: bool = True

    W: Optional[np.ndarray] = None       # process noise covariance (state weight of dual)
    V: Optional[np.ndarray] = None       # measurement noise covariance (control weight of dual)
    W_N: Optional[np.ndarray] = None     # terminal cost-to-go for dual (rarely needed; default 0)
    Sigma_0_dual: Optional[np.ndarray] = None
    N: int = 1

    method: str = "direct"
    gd_max_iter: int = 500
    gd_tol: float = 1e-8
    conv_tol: float = DEFAULT_CONV_TOL

    # outputs
    L_stages: list = field(default_factory=list)
    P_stages_grid: list = field(default_factory=list)
    J_history: list = field(default_factory=list)

    # cached duals of the primal grid quantities
    _alpha: Optional[np.ndarray] = None
    _A_dual_grid: Optional[np.ndarray] = None
    _C_dual_grid: Optional[np.ndarray] = None
    _weights: Optional[np.ndarray] = None
    _R_count: int = 0
    _p_dim: int = 0    # number of outputs == "input dim" of the dual

    def __post_init__(self):
        if not self.discrete:
            raise NotImplementedError(
                "Observer P-RHPG is implemented for discrete-time only "
                "(matches tptool.prhpg.synthesis convention)."
            )
        self._precompute_grid()

    # ------------------------------------------------------------------
    # configuration
    # ------------------------------------------------------------------

    def set_noise_covariances(self, W, V):
        """Set process and measurement noise covariances.

        W : (n, n) symmetric PSD  -- "state weight" of the dual problem
        V : (p, p) symmetric PD   -- "control weight" of the dual problem
        """
        W = np.asarray(W, dtype=float)
        V = np.asarray(V, dtype=float)
        if W.shape != (self.n, self.n):
            raise ValueError(f"W must be ({self.n}, {self.n}); got {W.shape}")
        if V.shape != (self._p_dim, self._p_dim):
            raise ValueError(f"V must be ({self._p_dim}, {self._p_dim}); got {V.shape}")
        self.W = W
        self.V = V

    def set_horizon(self, N):
        if N < 1:
            raise ValueError(f"horizon N must be >= 1, got {N}")
        self.N = int(N)

    def set_terminal_cost(self, W_N):
        """Optional terminal weight (defaults to zero)."""
        W_N = np.asarray(W_N, dtype=float)
        if W_N.shape != (self.n, self.n):
            raise ValueError(f"W_N must be ({self.n}, {self.n}); got {W_N.shape}")
        self.W_N = W_N

    def set_initial_covariance(self, Sigma_0_dual):
        """Initial covariance for the dual cost integral (defaults to I)."""
        S = np.asarray(Sigma_0_dual, dtype=float)
        if S.shape != (self.n, self.n):
            raise ValueError(f"Sigma_0_dual must be ({self.n}, {self.n}); got {S.shape}")
        self.Sigma_0_dual = S

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _precompute_grid(self):
        """Reuse the primal's grid machinery, but build A^T and C^T grids."""
        # Primal vertices and quadrature
        A_v, B_v, R_count, m = _extract_vertices(self.tp.S, self.n)
        alpha = _evaluate_weights_on_grid(self.tp.U)
        A_grid_primal, _B_grid_primal = _evaluate_AB_on_grid(A_v, B_v, alpha)
        weights = _build_quadrature_weights(self.tp.U, self.tp.domain)
        N_q = alpha.shape[0]

        # Dual: A_dual(p) = A(p)^T,  B_dual(p) = C(p)^T
        A_dual_grid = np.transpose(A_grid_primal, (0, 2, 1))   # (N_q, n, n)
        C_grid = _broadcast_C_to_grid(self.C, N_q)              # (N_q, p, n)
        C_dual_grid = np.transpose(C_grid, (0, 2, 1))           # (N_q, n, p)
        p_dim = C_grid.shape[1]

        self._alpha = alpha
        self._A_dual_grid = A_dual_grid
        self._C_dual_grid = C_dual_grid
        self._weights = weights
        self._R_count = R_count
        self._p_dim = p_dim

    # ------------------------------------------------------------------
    # solve
    # ------------------------------------------------------------------

    def solve(self):
        """Run the dual backward sweep and return polytopic observer gains.

        Returns
        -------
        L : ndarray, shape (R, n, p)
            Stage-0 vertex observer gains. To deploy at parameter p:
                L(p) = sum_i alpha_i(p) * L[i]
            Error dynamics: e_{k+1} = (A(p) - L(p) C) e_k.
        J_dual : float
            Optimal integrated dual cost (interpretable as steady-state
            estimation-error trace).
        """
        if self.W is None or self.V is None:
            raise ValueError("Must call set_noise_covariances(W, V) before solve().")
        if self.W_N is None:
            self.W_N = np.zeros((self.n, self.n))
        if self.Sigma_0_dual is None:
            self.Sigma_0_dual = np.eye(self.n)

        N_q = self._alpha.shape[0]
        P_terminal = np.tile(self.W_N, (N_q, 1, 1))

        # Run the existing backward sweep on the dual plant.
        # The solver thinks of (A_dual, B_dual) = (A^T, C^T) as a
        # state-feedback LQ problem. With the primal's sign convention
        # (returns K such that A + B K is stable), it returns K_dual
        # such that A^T + C^T K_dual is stable.
        K_stages, P_stages, J_stages = backward_sweep(
            N=self.N,
            P_terminal_grid=P_terminal,
            alpha=self._alpha,
            A_grid=self._A_dual_grid,
            B_grid=self._C_dual_grid,        # dual "B" = C^T
            weights=self._weights,
            Q_mat=self.W,                     # dual "Q" = W
            R_mat=self.V,                     # dual "R" = V
            Sigma_0=self.Sigma_0_dual,
            n=self.n,
            m=self._p_dim,                    # dual "m" = p (output dim)
            R_count=self._R_count,
            method=self.method,
            gd_max_iter=self.gd_max_iter,
            gd_tol=self.gd_tol,
        )

        # Translate dual K back to primal observer L.
        #
        # Dual closed-loop (stable):     A^T + C^T K_dual
        # Want observer-error stable:    A - L C
        # Setting L = -K_dual^T gives:
        #     A - L C = A + K_dual^T C = (A^T + C^T K_dual)^T
        # which has identical eigenvalues to the dual closed-loop.
        #
        # backward_sweep returns K_stages[h] of shape (R, m_dual, n)
        # where m_dual = p; we need L of shape (R, n, p).
        L_stages = [-np.transpose(K_h, (0, 2, 1)) for K_h in K_stages]

        self.L_stages = L_stages
        self.P_stages_grid = P_stages
        self.J_history = J_stages

        # convergence diagnostic mirroring the primal
        if self.N >= 2:
            J0 = J_stages[0]
            J1 = J_stages[1]
            if J0 > 0 and abs(J0 - J1) / J0 > self.conv_tol:
                warnings.warn(
                    f"Observer P-RHPG horizon may be insufficient: "
                    f"|J_N - J_{{N-1}}| / J_N = {abs(J0 - J1) / J0:.3f} > "
                    f"{self.conv_tol:.3f}. Increase N for tighter convergence.",
                    stacklevel=2,
                )

        return L_stages[0], J_stages[0]


# ----------------------------------------------------------------------
# Functional API (mirrors lmi/synthesis style)
# ----------------------------------------------------------------------


def observer_struct(tp, C, n, **kwargs):
    """Construct an ObserverPRHPGProblem (mirrors prhpg_struct)."""
    return ObserverPRHPGProblem(tp=tp, C=C, n=n, **kwargs)


def observer_solve(problem):
    """Solve an ObserverPRHPGProblem (mirrors prhpg_solve)."""
    return problem.solve()
