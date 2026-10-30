"""
LQG synthesis smoke test on FliPASED P-FLEX reduced symmetric channel.

Uses the modal-block-reduced 12-state plant (flutter-relevant modes only).

Run from project root:
    python -m examples.lqg_flipased_demo.smoke_test_lqg
"""

from __future__ import annotations

import numpy as np

from examples.lqg_flipased_demo.dynamics import (
    FlipasedTPForm,
    discretize_polytope,
    flipased_polytope_reduced,
)
from tptool.prhpg import ObserverPRHPGProblem, PRHPGProblem


CHANNEL = "sym"
DT = 0.005
HORIZON = 200


def build_Q(n: int, channel: str) -> np.ndarray:
    """Heavy weight on flutter mode states.

    For sym reduced plant (12 states, blocks 1..6 of original):
        block 1 (1.91 Hz)  -> reduced states 0,1
        block 2 (3.05 Hz)  -> reduced states 2,3
        block 3 (6.52 Hz)  -> reduced states 4,5
        block 4 (7.78 Hz)  -> reduced states 6,7   <-- flutter
        block 5 (12.4 Hz)  -> reduced states 8,9
        block 6 (14.9 Hz)  -> reduced states 10,11

    For asym reduced plant (11 states, blocks 1..6 of original):
        block 1 (real)     -> reduced state  0
        block 2 (6.52 Hz)  -> reduced states 1,2
        block 3 (7.42 Hz)  -> reduced states 3,4   <-- lightly damped (flutter-leading)
        block 4 (9.12 Hz)  -> reduced states 5,6
        block 5 (15.8 Hz)  -> reduced states 7,8
        block 6 (19.4 Hz)  -> reduced states 9,10
    """
    Q = np.eye(n)
    if channel == "sym":
        Q[6, 6] = 1e4
        Q[7, 7] = 1e4
    elif channel == "asym":
        Q[3, 3] = 1e4
        Q[4, 4] = 1e4
    return Q


R_CTRL = np.array([[1.0]])

W_PROCESS = lambda n: np.eye(n) * 1.0
V_NOISE = (1e-3) ** 2 * np.eye(2)


def _spectral_radius(M):
    return float(np.max(np.abs(np.linalg.eigvals(M))))


def main():
    print("=" * 70)
    print(f"FliPASED LQG smoke test (reduced)  channel={CHANNEL}  dt={DT}  N={HORIZON}")
    print("=" * 70)

    poly = flipased_polytope_reduced(CHANNEL)
    poly_d = discretize_polytope(poly, dt=DT)
    n, m, p, R = poly_d.n, poly_d.m, poly_d.p, poly_d.R
    print(f"plant: n={n} (reduced from {poly_d.full_n}), m={m}, p={p}, R={R}")

    print(f"\nopen-loop discrete spectral radii per vertex:")
    for r in range(R):
        rho = _spectral_radius(poly_d.A[r])
        marker = "  <-- unstable" if rho > 1.0 else ""
        print(f"  V={poly_d.V_grid[r]:5.2f}  rho={rho:7.4f}{marker}")

    tp_d = FlipasedTPForm.from_polytope(poly_d)

    # state-feedback K
    print(f"\n--- state-feedback P-RHPG ---")
    Q = build_Q(n, CHANNEL)
    sf = PRHPGProblem(tp_d, n=n)
    sf.set_stage_cost(Q=Q, R=R_CTRL)
    sf.set_horizon(N=HORIZON)
    K_prhpg, J_K = sf.solve()
    K_perv = K_prhpg.reshape(R, m, n)
    K_dep_perv = -K_perv
    print(f"  J_K = {J_K:.4e}")
    print(f"  ||K_dep||_F: min={min(np.linalg.norm(K_dep_perv[r]) for r in range(R)):.2f}, "
          f"max={max(np.linalg.norm(K_dep_perv[r]) for r in range(R)):.2f}")

    print(f"  closed-loop spectral radii per vertex:")
    rhos_K = []
    for r in range(R):
        rho = _spectral_radius(poly_d.A[r] - poly_d.B[r] @ K_dep_perv[r])
        rhos_K.append(rho)
        marker = " ✓" if rho < 1.0 else "  ✗"
        print(f"    V={poly_d.V_grid[r]:5.2f}  rho={rho:7.4f}{marker}")

    if max(rhos_K) >= 1.0:
        print("\n*** state feedback failed at some vertex.")
        return

    # observer L
    print(f"\n--- observer P-RHPG ---")
    obs = ObserverPRHPGProblem(tp=tp_d, C=poly_d.C, n=n)
    obs.set_noise_covariances(W=W_PROCESS(n), V=V_NOISE)
    obs.set_horizon(N=HORIZON)
    L_flat, J_L = obs.solve()
    L_perv = L_flat.reshape(R, n, p)
    print(f"  J_L = {J_L:.4e}")
    print(f"  ||L||_F: min={min(np.linalg.norm(L_perv[r]) for r in range(R)):.2f}, "
          f"max={max(np.linalg.norm(L_perv[r]) for r in range(R)):.2f}")

    print(f"  observer-error spectral radii per vertex:")
    rhos_L = []
    for r in range(R):
        rho = _spectral_radius(poly_d.A[r] - L_perv[r] @ poly_d.C[r])
        rhos_L.append(rho)
        marker = " ✓" if rho < 1.0 else "  ✗"
        print(f"    V={poly_d.V_grid[r]:5.2f}  rho={rho:7.4f}{marker}")

    if max(rhos_L) >= 1.0:
        print("\n*** observer failed at some vertex.")
        return

    # combined LQG
    print(f"\n--- combined LQG closed loop ---")
    rhos_LQG = []
    for r in range(R):
        Av, Bv, Cv = poly_d.A[r], poly_d.B[r], poly_d.C[r]
        Kv, Lv = K_dep_perv[r], L_perv[r]
        ABK = Av - Bv @ Kv
        ALC = Av - Lv @ Cv
        M = np.block([[ABK, Bv @ Kv],
                      [np.zeros_like(Av), ALC]])
        rho = _spectral_radius(M)
        rhos_LQG.append(rho)
        marker = " ✓" if rho < 1.0 else "  ✗"
        print(f"    V={poly_d.V_grid[r]:5.2f}  rho={rho:7.4f}{marker}")

    print("\n" + "=" * 70)
    if max(rhos_LQG) < 1.0:
        print(f"✓ PASS")
        print(f"   max rho_K   = {max(rhos_K):.4f}")
        print(f"   max rho_L   = {max(rhos_L):.4f}")
        print(f"   max rho_LQG = {max(rhos_LQG):.4f}")
    else:
        print(f"✗ FAIL  --  combined LQG unstable at some vertex.")
    print("=" * 70)


if __name__ == "__main__":
    main()
