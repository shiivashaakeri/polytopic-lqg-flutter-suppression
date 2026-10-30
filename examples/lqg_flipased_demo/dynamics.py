"""
FliPASED P-FLEX plant for the LQG demo, with modal-block subselection.

Loads modal-form (A, B, C) on the airspeed grid, then keeps a chosen
subset of modal blocks for synthesis. The default keeps the 14 states
that matter for closed-loop output-feedback stability:

    blocks 1..7 (sym):
      1: 1.91 Hz short period
      2: 3.05 Hz flexible
      3: 6.52 Hz flexible
      4: 7.78 Hz FLUTTER mode
      5: 12.4 Hz flexible
      6: 14.9 Hz flexible (lightly damped)
      7: 28.8 Hz actuator complex pair

We DROP:
    block 0:    phugoid (0.04 Hz, well-damped, untouchable from aileron)
    blocks 8-9: two fast actuator real poles (>1500 rad/s, irrelevant at 200 Hz)

The actuator complex pair (block 7) MUST be kept because it carries the
dominant input coupling: ||B_block7|| ~ 1400, larger than any structural
mode. Without it the observer gets confused by unmodeled actuator dynamics
showing up in the IMU output, and the closed loop diverges at high V.

Why modal slicing rather than Kalman decomposition: see project history.
Modal blocks of the real modal form are 2x2 (or 1x1) A_v-invariant by
construction at every vertex, so slicing them produces a well-defined
polytope. SVD-based "shared controllable subspaces" generally aren't
A_v-invariant and corrupt the polytope.

Public API:
    flipased_polytope(channel='sym')                    -> full plant
    flipased_polytope_reduced(channel='sym')            -> 14-state synthesis plant
    flipased_ab_dep(poly)                               -> (AB, dep) for tptrans
    flipased_tpform(poly, hull, gridsize, **kw)         -> TPForm (proper HOSVD hull)
    project_state(x_full, slice)                        -> reduced-state slice
    lift_state(x_red, slice, n_full)                    -> full-state with zeros on dropped
    discretize_polytope(poly, dt)                       -> ZOH-discretized polytope
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


_DATA_DIR = Path(__file__).parent / "data"
_NPZ = {
    "sym":  _DATA_DIR / "flipased_modal_sym.npz",
    "asym": _DATA_DIR / "flipased_modal_asym.npz",
}

V_MIN = 40.0
V_MAX = 61.0

# Default modal blocks to keep for synthesis: drop phugoid (block 0) and the
# two fast actuator real poles (blocks 8, 9). Keep the actuator complex pair
# (block 7) because it carries dominant input coupling.
_DEFAULT_KEEP_BLOCKS_SYM  = list(range(1, 8))
_DEFAULT_KEEP_BLOCKS_ASYM = list(range(1, 8))


@dataclass
class FlipasedPolytope:
    channel: str
    n: int
    m: int
    p: int
    R: int
    V_grid: np.ndarray
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    weights: np.ndarray
    mode_freqs_hz: np.ndarray = field(default_factory=lambda: np.array([]))
    mode_zetas:    np.ndarray = field(default_factory=lambda: np.array([]))
    is_reduced: bool = False
    state_indices: Optional[np.ndarray] = None
    full_n: Optional[int] = None


def _trap_weights(V_grid: np.ndarray) -> np.ndarray:
    V = np.asarray(V_grid, dtype=float)
    n = len(V)
    if n == 1:
        return np.array([1.0])
    w = np.zeros(n)
    w[0]  = 0.5 * (V[1]  - V[0])
    w[-1] = 0.5 * (V[-1] - V[-2])
    for k in range(1, n - 1):
        w[k] = 0.5 * (V[k + 1] - V[k - 1])
    return w / w.sum()


def _state_indices_from_blocks(starts, ends, blocks):
    idx = []
    for b in blocks:
        idx.extend(range(int(starts[b]), int(ends[b])))
    return np.array(idx, dtype=int)


def flipased_polytope(channel: str = "sym") -> FlipasedPolytope:
    """Load the full 18-state (sym) or 17-state (asym) modal-form polytope."""
    if channel not in _NPZ:
        raise ValueError(f"channel must be 'sym' or 'asym'; got {channel!r}")
    path = _NPZ[channel]
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}. Place flipased_modal_{channel}.npz in {_DATA_DIR}/."
        )
    d = np.load(path, allow_pickle=True)
    A = np.transpose(d["A_modal"], (2, 0, 1))
    B = np.transpose(d["B_modal"], (2, 0, 1))
    C = np.transpose(d["C_modal"], (2, 0, 1))
    D = np.transpose(d["D_modal"], (2, 0, 1))
    V_grid = d["V_grid"].astype(float).ravel()
    return FlipasedPolytope(
        channel=channel,
        n=A.shape[1], m=B.shape[2], p=C.shape[1], R=len(V_grid),
        V_grid=V_grid, A=A, B=B, C=C, D=D,
        weights=_trap_weights(V_grid),
        mode_freqs_hz=d.get("mode_freqs_hz", np.array([])),
        mode_zetas=d.get("mode_zetas", np.array([])),
        is_reduced=False,
    )


def flipased_polytope_reduced(
    channel: str = "sym",
    keep_blocks: Optional[list] = None,
) -> FlipasedPolytope:
    """Load and slice: returns the modal-block-reduced polytope."""
    poly_full = flipased_polytope(channel)
    d = np.load(_NPZ[channel], allow_pickle=True)
    starts = d["mode_starts"]
    ends   = d["mode_ends"]
    if keep_blocks is None:
        keep_blocks = (_DEFAULT_KEEP_BLOCKS_SYM if channel == "sym"
                       else _DEFAULT_KEEP_BLOCKS_ASYM)
    state_idx = _state_indices_from_blocks(starts, ends, keep_blocks)
    n_red = len(state_idx)
    A_r = poly_full.A[:, state_idx, :][:, :, state_idx]
    B_r = poly_full.B[:, state_idx, :]
    C_r = poly_full.C[:, :, state_idx]
    D_r = poly_full.D.copy()
    keep_mask = np.zeros(len(starts), dtype=bool)
    keep_mask[keep_blocks] = True
    freqs_r = poly_full.mode_freqs_hz[keep_mask] if poly_full.mode_freqs_hz.size else np.array([])
    zetas_r = poly_full.mode_zetas[keep_mask]    if poly_full.mode_zetas.size    else np.array([])
    return FlipasedPolytope(
        channel=channel,
        n=n_red, m=poly_full.m, p=poly_full.p, R=poly_full.R,
        V_grid=poly_full.V_grid.copy(),
        A=A_r, B=B_r, C=C_r, D=D_r,
        weights=poly_full.weights.copy(),
        mode_freqs_hz=freqs_r, mode_zetas=zetas_r,
        is_reduced=True,
        state_indices=state_idx,
        full_n=poly_full.n,
    )


def project_state(x_full: np.ndarray, state_indices: np.ndarray) -> np.ndarray:
    return x_full[state_indices]


def lift_state(x_red: np.ndarray, state_indices: np.ndarray, n_full: int) -> np.ndarray:
    out = np.zeros(n_full)
    out[state_indices] = x_red
    return out


def discretize_polytope(poly: FlipasedPolytope, dt: float) -> FlipasedPolytope:
    from scipy.linalg import expm
    R, n, _ = poly.A.shape
    m = poly.B.shape[2]
    A_d = np.zeros_like(poly.A)
    B_d = np.zeros_like(poly.B)
    for r in range(R):
        M = np.zeros((n + m, n + m))
        M[:n, :n] = poly.A[r]
        M[:n, n:] = poly.B[r]
        eM = expm(M * dt)
        A_d[r] = eM[:n, :n]
        B_d[r] = eM[:n, n:]
    return FlipasedPolytope(
        channel=poly.channel,
        n=poly.n, m=poly.m, p=poly.p, R=poly.R,
        V_grid=poly.V_grid.copy(),
        A=A_d, B=B_d, C=poly.C.copy(), D=poly.D.copy(),
        weights=poly.weights.copy(),
        mode_freqs_hz=poly.mode_freqs_hz.copy() if poly.mode_freqs_hz.size else np.array([]),
        mode_zetas=poly.mode_zetas.copy()       if poly.mode_zetas.size    else np.array([]),
        is_reduced=poly.is_reduced,
        state_indices=poly.state_indices.copy() if poly.state_indices is not None else None,
        full_n=poly.full_n,
    )


def flipased_lpv_at(V: float, channel: str = "sym",
                     poly: Optional[FlipasedPolytope] = None):
    if poly is None:
        poly = flipased_polytope(channel)
    V_clamped = float(np.clip(V, V_MIN, V_MAX))
    Vg = poly.V_grid
    if V_clamped <= Vg[0]:
        return poly.A[0].copy(), poly.B[0].copy(), poly.C[0].copy(), poly.D[0].copy()
    if V_clamped >= Vg[-1]:
        return (poly.A[-1].copy(), poly.B[-1].copy(),
                poly.C[-1].copy(), poly.D[-1].copy())
    k1 = int(np.searchsorted(Vg, V_clamped) - 1)
    k1 = max(0, min(k1, poly.R - 2))
    k2 = k1 + 1
    frac = (V_clamped - Vg[k1]) / (Vg[k2] - Vg[k1])
    A = (1 - frac) * poly.A[k1] + frac * poly.A[k2]
    B = (1 - frac) * poly.B[k1] + frac * poly.B[k2]
    C = (1 - frac) * poly.C[k1] + frac * poly.C[k2]
    D = (1 - frac) * poly.D[k1] + frac * poly.D[k2]
    return A, B, C, D


def alpha_at(V: float, V_grid: np.ndarray) -> np.ndarray:
    R = len(V_grid)
    a = np.zeros(R)
    V_clamped = float(np.clip(V, V_grid[0], V_grid[-1]))
    if V_clamped <= V_grid[0]:
        a[0] = 1.0; return a
    if V_clamped >= V_grid[-1]:
        a[-1] = 1.0; return a
    k1 = int(np.searchsorted(V_grid, V_clamped) - 1)
    k1 = max(0, min(k1, R - 2))
    k2 = k1 + 1
    frac = (V_clamped - V_grid[k1]) / (V_grid[k2] - V_grid[k1])
    a[k1] = 1 - frac
    a[k2] = frac
    return a


# ---------------------------------------------------------------------------
# Proper TP model transformation (HOSVD + convex hull)
# ---------------------------------------------------------------------------


def flipased_ab_dep(poly: FlipasedPolytope):
    """
    Convert a FlipasedPolytope into the (AB, dep) callable format expected
    by tptool's tptrans.

    The LPV is parameterized by a single scheduling variable V in
    [V_MIN, V_MAX]. At each V the plant is interpolated linearly from the
    R grid vertices. tptrans will then sample this callable on its own
    internal grid and run HOSVD + hull compression.

    Returns
    -------
    AB : list of lists of callables, shape (n, n+m)
        AB[i][j](p) returns the (i,j) entry of [A(V) | B(V)] at p=[V].
    dep : ndarray, shape (n, n+m, 1), int
        dep[i,j,0] = 1 if entry (i,j) depends on V, else 0.
    """
    n, m = poly.n, poly.m
    V_grid = poly.V_grid

    # Cache the interpolated AB matrix at a given V
    def _ab_at(V):
        a = alpha_at(float(V), V_grid)
        AB_full = np.concatenate([poly.A, poly.B], axis=2)  # (R, n, n+m)
        return np.einsum("r,rij->ij", a, AB_full)

    def _make_entry(i, j):
        def entry(p):
            return float(_ab_at(float(p[0]))[i, j])
        return entry

    AB = [[_make_entry(i, j) for j in range(n + m)] for i in range(n)]

    # Dependency mask: entry depends on V iff it varies across the grid
    AB_lo = np.concatenate([poly.A[0:1], poly.B[0:1]], axis=2)[0]
    AB_hi = np.concatenate([poly.A[-1:], poly.B[-1:]], axis=2)[0]
    diff = np.abs(AB_hi - AB_lo)
    mag  = np.abs(AB_lo) + np.abs(AB_hi)
    dep = np.zeros((n, n + m, 1), dtype=int)
    dep[:, :, 0] = (diff > 1e-9 * (1.0 + mag)).astype(int)

    return AB, dep


def flipased_tpform(
    poly: FlipasedPolytope,
    hull: str = "cno",
    gridsize: int = 51,
    svtol: float = 1e-5,
    verbose: bool = False,
):
    """
    Build a proper TP model form for a FlipasedPolytope using HOSVD + hull.

    This replaces the old FlipasedTPForm.from_polytope() which used an
    identity weighting matrix (no HOSVD, no compression, no hull guarantee).

    Parameters
    ----------
    poly : FlipasedPolytope
        Typically the discretized reduced polytope (from discretize_polytope).
    hull : str
        Hull type passed to tptrans: 'cno', 'irno', 'ino', 'snnn', 'close'.
    gridsize : int
        Number of points for tptrans to sample the LPV over [V_MIN, V_MAX].
    svtol : float
        Singular value truncation tolerance for HOSVD.
    verbose : bool

    Returns
    -------
    tp : TPForm
        Object with .S (core tensor) and .U (weighting matrices) attributes,
        compatible with PRHPGProblem and ObserverPRHPGProblem.
    """
    from tptool.lpv import tptrans

    AB, dep = flipased_ab_dep(poly)
    domain = np.array([[V_MIN, V_MAX]])

    tp = tptrans(
        AB,
        dep,
        domain,
        [gridsize],
        hull=hull,
        svtol=svtol,
    )

    if verbose:
        vc = [u.shape[1] for u in tp.U]
        print(f"  flipased_tpform({hull}): core {tp.S.shape}, vertices={vc}")

    return tp


if __name__ == "__main__":
    print("=" * 70)
    print("FliPASED dynamics self-test (modal-block reduction, includes actuator)")
    print("=" * 70)

    for channel in ("sym", "asym"):
        print(f"\n--- {channel} ---")
        try:
            poly_full = flipased_polytope(channel)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        print(f"  full plant: n={poly_full.n}, R={poly_full.R}")
        print(f"  modal blocks (idx, freq, zeta, ||B||):")
        d = np.load(_NPZ[channel], allow_pickle=True)
        for k in range(len(d["mode_starts"])):
            s = int(d['mode_starts'][k]); e = int(d['mode_ends'][k])
            B_norm = np.linalg.norm(poly_full.B[5][s:e, :])  # at V_mean
            kept = "KEPT  " if k in _DEFAULT_KEEP_BLOCKS_SYM else "drop  "
            print(f"    {kept} block {k}: states {s}..{e-1}  "
                  f"f={poly_full.mode_freqs_hz[k]:7.3f} Hz  "
                  f"zeta={poly_full.mode_zetas[k]:+.4f}  "
                  f"||B[block]|| at V_mean = {B_norm:8.2f}")

        poly_red = flipased_polytope_reduced(channel)
        print(f"\n  reduced plant: n={poly_red.n} states "
              f"(kept blocks {_DEFAULT_KEEP_BLOCKS_SYM if channel == 'sym' else _DEFAULT_KEEP_BLOCKS_ASYM})")

        from numpy.linalg import matrix_rank
        ranks = []
        for r in range(poly_red.R):
            Wc = np.hstack([np.linalg.matrix_power(poly_red.A[r], k) @ poly_red.B[r]
                            for k in range(poly_red.n)])
            ranks.append(matrix_rank(Wc, tol=1e-9))
        print(f"  per-vertex ctrb rank: min={min(ranks)}, max={max(ranks)} / {poly_red.n}")

        print(f"\n  open-loop continuous max Re(eig) per V:")
        for r in range(poly_red.R):
            re_max = float(np.linalg.eigvals(poly_red.A[r]).real.max())
            print(f"    V={poly_red.V_grid[r]:5.2f}  max Re(eig) = {re_max:+8.4f}")

        # Test the new tpform path
        print(f"\n  testing flipased_tpform (cno, gridsize=21) ...")
        from examples.lqg_flipased_demo.dynamics import discretize_polytope, flipased_tpform
        poly_red_d = discretize_polytope(poly_red, dt=0.005)
        tp = flipased_tpform(poly_red_d, hull="cno", gridsize=21, verbose=True)
        print(f"  OK: S.shape={tp.S.shape}, U[0].shape={tp.U[0].shape}")
