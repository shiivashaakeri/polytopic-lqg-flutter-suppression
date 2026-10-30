"""
LMI synthesis for TP-form polytopic LPV systems.

Ported from MATLAB TP Tool: lmi/lmistruct.m, lmi_asym.m, lmi_asym_decay.m,
lmi_input.m, lmi_solve.m.

The MATLAB version uses YALMIP. We use CVXPY as the modeling layer with
MOSEK as the default SDP solver (SCS as fallback). Algorithms follow the
textbook exactly:
  K. Tanaka, H. O. Wang, "Fuzzy Control Systems Design and Analysis"
  (Wiley, 2002), pages 60-69.

Both continuous-time and discrete-time formulations are supported.

Convention for the core tensor S:
  Shape (R_1, ..., R_P, n+p, n+m) where the trailing two axes hold the
  augmented system matrix [A B; C D] for each vertex. After flattening
  the leading axes we get an array of shape (R, n+p, n+m).

Two synthesis problems:

  1. LMIProblem  -- state-feedback gain K
        problem = LMIProblem(S, n, discrete=True)
        problem.add_stability()
        K, P = problem.solve()

  2. LMIObserverProblem  -- observer gain L (dual problem)
        obs = LMIObserverProblem(S, C, n, discrete=True)
        obs.add_stability()
        L, Pi = obs.solve()

MATLAB-style functional interface also provided (lmi_struct, lmi_asym, etc.).
"""

from __future__ import annotations

import numpy as np

try:
    import cvxpy as cp
    _HAS_CVXPY = True
except ImportError:
    cp = None
    _HAS_CVXPY = False

DEFAULT_SOLVER = "MOSEK"


# ===================================================================
# State-feedback LMI (primal)
# ===================================================================


class LMIProblem:
    """
    LMI synthesis problem for state-feedback gain K.

    Parameters
    ----------
    S : ndarray, shape (R_1, ..., R_P, n+p, n+m)
        TP core tensor. Trailing axes hold [A B; C D] at each vertex.
    n : int
        State dimension.
    discrete : bool, default False
        If True, uses discrete-time Schur LMIs.
        If False, uses Tanaka-Wang continuous-time formulation.
    """

    def __init__(self, S, n, discrete=False):
        if not _HAS_CVXPY:
            raise ImportError("tptool.lmi requires cvxpy.")
        self.discrete = discrete
        self._build_from_core(np.asarray(S), int(n))

    def _build_from_core(self, S, n):
        sizes = S.shape[:-2]
        R = int(np.prod(sizes))
        n_plus_p, n_plus_m = S.shape[-2], S.shape[-1]
        m = n_plus_m - n
        p = n_plus_p - n
        if m < 0 or p < 0:
            raise ValueError(
                f"LMIProblem: trailing axes ({n_plus_p}, {n_plus_m}) "
                f"too small for n={n}")
        S_flat = S.reshape(R, n_plus_p, n_plus_m)
        self.sizes = sizes
        self.R = R
        self.n = n
        self.m = m
        self.p = p
        self.A = S_flat[:, :n, :n].copy()
        self.B = S_flat[:, :n, n:n+m].copy()
        self.C = S_flat[:, n:n+p, :n].copy()
        self.D = S_flat[:, n:n+p, n:n+m].copy()

        self.X = cp.Variable((n, n), symmetric=True)
        self.M = [cp.Variable((m, n)) for _ in range(R)]
        self.constraints = [self.X >> 1e-3 * np.eye(n)]

    def add_stability(self, margin=1e-3, cross_pairs=True):
        """
        Asymptotic stability LMIs (Tanaka-Wang PDC).
    
        Parameters
        ----------
        margin : float
            Stability margin. Continuous: Re(eig) <= -margin.
            Discrete: |eig| <= 1 - margin.
        cross_pairs : bool, default True
            If True (Tanaka-Wang PDC, strict): include symmetrized cross-pair
            constraints for every (r, s) with r != s. This guarantees
            stability under arbitrary convex combinations of the vertex
            gains.
            If False (diagonal / quadratic stability, loose): include only
            the diagonal r constraints. This is a strictly weaker condition
            guaranteeing only vertex-by-vertex stability; it is the most
            permissive convex relaxation in this family.
        """
        if margin < 0:
            raise ValueError(f"margin must be >= 0, got {margin}")
        slack = 1e-6 * np.eye(self.n)
    
        if not self.discrete:
            # ----- continuous-time -----
            for r in range(self.R):
                Ar, Br, Mr = self.A[r], self.B[r], self.M[r]
                expr = (-self.X @ Ar.T - Ar @ self.X
                        + Br @ Mr + Mr.T @ Br.T
                        - 2.0 * margin * self.X)
                self.constraints.append(expr >> slack)
            if cross_pairs:
                for r in range(self.R):
                    for s in range(r + 1, self.R):
                        Ar, As = self.A[r], self.A[s]
                        Br, Bs = self.B[r], self.B[s]
                        Mr, Ms = self.M[r], self.M[s]
                        expr = (-self.X @ Ar.T - Ar @ self.X
                                - self.X @ As.T - As @ self.X
                                + Br @ Ms + Ms.T @ Br.T
                                + Bs @ Mr + Mr.T @ Bs.T
                                - 4.0 * margin * self.X)
                        self.constraints.append(expr >> 0)
        else:
            # ----- discrete-time -----
            scale = (1.0 - margin) ** 2 if margin > 0 else 1.0
            for r in range(self.R):
                Ar, Br, Mr = self.A[r], self.B[r], self.M[r]
                tr = Ar @ self.X + Br @ Mr
                block = cp.bmat([[scale * self.X, tr.T], [tr, self.X]])
                self.constraints.append(block >> 1e-6 * np.eye(2 * self.n))
            if cross_pairs:
                for r in range(self.R):
                    for s in range(r + 1, self.R):
                        Ar, As = self.A[r], self.A[s]
                        Br, Bs = self.B[r], self.B[s]
                        Mr, Ms = self.M[r], self.M[s]
                        cross = 0.5 * (Ar @ self.X + Br @ Ms
                                    + As @ self.X + Bs @ Mr)
                        block = cp.bmat([[scale * self.X, cross.T], [cross, self.X]])
                        self.constraints.append(block >> 0)
        return self

    def add_decay(self, alpha):
        """Decay-rate LMIs: Re(eig) <= -alpha (CT) or |eig| <= exp(-alpha) (DT)."""
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        slack = 1e-6 * np.eye(self.n)
        if not self.discrete:
            for r in range(self.R):
                Ar, Br, Mr = self.A[r], self.B[r], self.M[r]
                expr = (-self.X @ Ar.T - Ar @ self.X
                        + Br @ Mr + Mr.T @ Br.T
                        - 2.0 * alpha * self.X)
                self.constraints.append(expr >> slack)
            for r in range(self.R):
                for s in range(r + 1, self.R):
                    Ar, As = self.A[r], self.A[s]
                    Br, Bs = self.B[r], self.B[s]
                    Mr, Ms = self.M[r], self.M[s]
                    expr = (-self.X @ Ar.T - Ar @ self.X
                            - self.X @ As.T - As @ self.X
                            + Br @ Ms + Ms.T @ Br.T
                            + Bs @ Mr + Mr.T @ Bs.T
                            - 4.0 * alpha * self.X)
                    self.constraints.append(expr >> 0)
        else:
            scale = np.exp(-2.0 * alpha)
            for r in range(self.R):
                Ar, Br, Mr = self.A[r], self.B[r], self.M[r]
                tr = Ar @ self.X + Br @ Mr
                block = cp.bmat([[scale * self.X, tr.T], [tr, self.X]])
                self.constraints.append(block >> 1e-6 * np.eye(2 * self.n))
            for r in range(self.R):
                for s in range(r + 1, self.R):
                    Ar, As = self.A[r], self.A[s]
                    Br, Bs = self.B[r], self.B[s]
                    Mr, Ms = self.M[r], self.M[s]
                    cross = 0.5 * (Ar @ self.X + Br @ Ms
                                   + As @ self.X + Bs @ Mr)
                    block = cp.bmat([[scale * self.X, cross.T], [cross, self.X]])
                    self.constraints.append(block >> 0)
        return self

    def add_input_constraint(self, umax, phi):
        """Input-bound LMIs: ||u|| <= umax given ||x(0)|| <= phi."""
        if umax <= 0 or phi <= 0:
            raise ValueError("umax and phi must be positive")
        self.constraints.append(
            self.X - (phi ** 2) * np.eye(self.n) >> 1e-6 * np.eye(self.n))
        for r in range(self.R):
            block = cp.bmat([[self.X, self.M[r].T],
                             [self.M[r], (umax**2) * np.eye(self.m)]])
            self.constraints.append(block >> 0)
        return self

    def solve(self, solver=DEFAULT_SOLVER, verbose=False, **solver_opts):
        """
        Solve the SDP and return (K, P).

        K : ndarray, shape (*sizes, m, n)  -- per-vertex feedback gains
        P : ndarray, shape (n, n)          -- Lyapunov matrix inv(X)
        """
        obj = (cp.Minimize(0) if self.discrete
               else cp.Minimize(cp.trace(self.X)))
        prob = cp.Problem(obj, self.constraints)
        try:
            prob.solve(solver=solver, verbose=verbose, **solver_opts)
        except Exception as e:
            raise RuntimeError(f"LMI solve failed: {e}") from e

        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(
                f"LMI infeasible or solver failure: status='{prob.status}'. "
                "Try a different hull, increase margin slack, or relax bounds.")

        X_val = self.X.value
        if X_val is None:
            raise RuntimeError("Solver returned no X value")

        P = np.linalg.inv(X_val)
        K_flat = np.zeros((self.R, self.m, self.n))
        for r in range(self.R):
            Mr_val = self.M[r].value
            if Mr_val is None:
                raise RuntimeError(f"Solver returned no M[{r}] value")
            K_flat[r] = Mr_val @ P

        return K_flat.reshape((*self.sizes, self.m, self.n)), P

    def __repr__(self):
        return (f"LMIProblem(sizes={self.sizes}, n={self.n}, m={self.m}, "
                f"p={self.p}, R={self.R}, discrete={self.discrete}, "
                f"n_constraints={len(self.constraints)})")


# ===================================================================
# Observer LMI (dual problem)
# ===================================================================


class LMIObserverProblem:
    """
    LMI synthesis problem for observer gain L.

    Uses the same Schur structure as LMIProblem but with the substitution:
        A_r  ->  A_r          (unchanged)
        B_r  ->  -C^T         (broadcast, same for all vertices)
        K_r  ->  L_r^T        (transpose of observer gain)
        M_r  ->  Z N_r^T      where N_r = L_r^T Z, Z = Lyapunov matrix

    The Schur block [Z, (A_r Z - N_r C)^T; A_r Z - N_r C, Z] >> 0
    guarantees rho(A_r - L_r C) < 1 - margin.

    Variables: Z (Lyapunov, symmetric n x n), N_r = L_r^T Z (shape p x n).
    Gain extraction: L_r = (N_r @ inv(Z)).T = Z^{-1} N_r^T.

    Parameters
    ----------
    S : ndarray, shape (R, n+p, n+m)  TP core tensor (A vertices extracted)
    C : ndarray, shape (p, n)          Output matrix (constant across vertices)
    n : int                            State dimension
    discrete : bool                    Discrete-time if True
    W : ndarray, shape (n, n), optional  Process noise — minimize tr(W Z) if given
    """

    def __init__(self, S, C, n, discrete=False, W=None):
        if not _HAS_CVXPY:
            raise ImportError("tptool.lmi requires cvxpy.")
        self.discrete = discrete
        S = np.asarray(S)
        sizes = S.shape[:-2]
        R = int(np.prod(sizes))
        n_plus_p, n_plus_m = S.shape[-2], S.shape[-1]

        S_flat = S.reshape(R, n_plus_p, n_plus_m)
        self.sizes = sizes
        self.R = R
        self.n = n
        C_arr = np.asarray(C)
        self.C = C_arr
        self.p = C_arr.shape[0]
        self.m = n_plus_m - n  # kept for compatibility
        self.W = np.asarray(W) if W is not None else None

        # A vertices from core tensor
        self.A = S_flat[:, :n, :n].copy()   # (R, n, n)

        # Lyapunov variable Z and auxiliary N_r = L_r^T Z, shape (p, n)
        self.Z = cp.Variable((n, n), symmetric=True)
        self.N = [cp.Variable((self.p, n)) for _ in range(R)]

        self.constraints = [self.Z >> 1e-3 * np.eye(n)]

    def add_stability(self, margin=1e-3, cross_pairs=True):
        """
        Observer stability LMIs.
    
        Parameters
        ----------
        margin : float
        cross_pairs : bool, default True
            If False, drop the cross-pair (r, s) constraints; equivalent to
            diagonal-only / quadratic stability of the observer.
        """
        if margin < 0:
            raise ValueError(f"margin must be >= 0, got {margin}")
    
        if not self.discrete:
            for r in range(self.R):
                Ar, Mr = self.A[r], self.N[r]
                C = self.C
                expr = (-self.Z @ Ar.T - Ar @ self.Z
                        - C.T @ Mr - Mr.T @ C
                        - 2.0 * margin * self.Z)
                self.constraints.append(expr >> 1e-6 * np.eye(self.n))
            if cross_pairs:
                for r in range(self.R):
                    for s in range(r + 1, self.R):
                        Ar, As = self.A[r], self.A[s]
                        Mr, Ms = self.N[r], self.N[s]
                        C = self.C
                        expr = (-self.Z @ Ar.T - Ar @ self.Z
                                - self.Z @ As.T - As @ self.Z
                                - C.T @ Mr - Mr.T @ C
                                - C.T @ Ms - Ms.T @ C
                                - 4.0 * margin * self.Z)
                        self.constraints.append(expr >> 0)
        else:
            scale = (1.0 - margin) ** 2 if margin > 0 else 1.0
            C = self.C
            for r in range(self.R):
                Ar, Mr = self.A[r], self.N[r]
                tr = Ar.T @ self.Z - C.T @ Mr
                block = cp.bmat([[scale * self.Z, tr.T], [tr, self.Z]])
                self.constraints.append(block >> 1e-6 * np.eye(2 * self.n))
            if cross_pairs:
                for r in range(self.R):
                    for s in range(r + 1, self.R):
                        Ar, As = self.A[r], self.A[s]
                        Mr, Ms = self.N[r], self.N[s]
                        cross = 0.5 * (Ar.T @ self.Z - C.T @ Mr
                                    + As.T @ self.Z - C.T @ Ms)
                        block = cp.bmat([[scale * self.Z, cross.T], [cross, self.Z]])
                        self.constraints.append(block >> 0)
        return self

    def add_decay(self, alpha):
        """Decay-rate LMIs for observer poles."""
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        C = self.C
        if not self.discrete:
            for r in range(self.R):
                Ar, Mr = self.A[r], self.N[r]
                expr = (-self.Z @ Ar.T - Ar @ self.Z
                        - C.T @ Mr - Mr.T @ C
                        - 2.0 * alpha * self.Z)
                self.constraints.append(expr >> 1e-6 * np.eye(self.n))
            for r in range(self.R):
                for s in range(r + 1, self.R):
                    Ar, As = self.A[r], self.A[s]
                    Mr, Ms = self.N[r], self.N[s]
                    expr = (-self.Z @ Ar.T - Ar @ self.Z
                            - self.Z @ As.T - As @ self.Z
                            - C.T @ Mr - Mr.T @ C
                            - C.T @ Ms - Ms.T @ C
                            - 4.0 * alpha * self.Z)
                    self.constraints.append(expr >> 0)
        else:
            scale = np.exp(-2.0 * alpha)
            for r in range(self.R):
                Ar, Mr = self.A[r], self.N[r]
                tr = Ar.T @ self.Z - C.T @ Mr
                block = cp.bmat([[scale * self.Z, tr.T], [tr, self.Z]])
                self.constraints.append(block >> 1e-6 * np.eye(2 * self.n))
            for r in range(self.R):
                for s in range(r + 1, self.R):
                    Ar, As = self.A[r], self.A[s]
                    Mr, Ms = self.N[r], self.N[s]
                    cross = 0.5 * (Ar.T @ self.Z - C.T @ Mr
                                   + As.T @ self.Z - C.T @ Ms)
                    block = cp.bmat([[scale * self.Z, cross.T], [cross, self.Z]])
                    self.constraints.append(block >> 0)
        return self

    def solve(self, solver=DEFAULT_SOLVER, verbose=False, **solver_opts):
        """
        Solve the dual SDP and return (L, Pi).

        L  : ndarray, shape (*sizes, n, p)  per-vertex observer gains
        Pi : ndarray, shape (n, n)          inv(Z)
        """
        if self.W is not None:
            obj = cp.Minimize(cp.trace(self.W @ self.Z))
        else:
            obj = (cp.Minimize(0) if self.discrete
                   else cp.Minimize(cp.trace(self.Z)))

        prob = cp.Problem(obj, self.constraints)
        try:
            prob.solve(solver=solver, verbose=verbose, **solver_opts)
        except Exception as e:
            raise RuntimeError(f"LMI observer solve failed: {e}") from e

        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(
                f"LMI observer infeasible: status='{prob.status}'. "
                "Try a different hull or relax margin.")

        Z_val = self.Z.value
        if Z_val is None:
            raise RuntimeError("Solver returned no Z value")

        Pi = np.linalg.inv(Z_val)
        L_flat = np.zeros((self.R, self.n, self.p))
        for r in range(self.R):
            Nr_val = self.N[r].value
            if Nr_val is None:
                raise RuntimeError(f"Solver returned no N[{r}] value")
            # N_r = L_r^T Z  =>  L_r^T = N_r Z^{-1} = N_r Pi
            # L_r = (N_r Pi).T
            L_flat[r] = (Nr_val @ Pi).T

        return L_flat.reshape((*self.sizes, self.n, self.p)), Pi

    def __repr__(self):
        return (f"LMIObserverProblem(sizes={self.sizes}, n={self.n}, "
                f"p={self.p}, R={self.R}, discrete={self.discrete}, "
                f"n_constraints={len(self.constraints)})")


# ===================================================================
# MATLAB-style functional interface
# ===================================================================

def lmi_struct(S, n, discrete=False):
    """Initialize LMIProblem from TP core tensor (mirrors lmistruct.m)."""
    return LMIProblem(S, n, discrete=discrete)

def lmi_asym(lmi):
    """Add asymptotic stability LMIs (mirrors lmi_asym.m)."""
    return lmi.add_stability()

def lmi_asym_decay(lmi, alpha):
    """Add decay-rate LMIs (mirrors lmi_asym_decay.m)."""
    return lmi.add_decay(alpha)

def lmi_input(lmi, umax, phi):
    """Add input-bound LMIs (mirrors lmi_input.m)."""
    return lmi.add_input_constraint(umax, phi)

def lmi_solve(lmi, **kwargs):
    """Solve and extract gains (mirrors lmi_solve.m)."""
    return lmi.solve(**kwargs)