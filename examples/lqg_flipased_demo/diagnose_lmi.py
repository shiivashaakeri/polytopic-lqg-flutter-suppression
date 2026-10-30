"""
LMI correctness diagnostic.

Verifies:
1. LMI K: all vertices of (A_r - B_r K_dep_r) have |eig| < 1
2. LMI L: all vertices of (A_r - L_r C) have |eig| < 1
3. Sign convention: K_dep = -K_lmi (negated for deployment)
4. Cross-check: closed-loop simulation converges

Run from project root:
    python -m examples.lqg_flipased_demo.diagnose_lmi
"""

from __future__ import annotations
import numpy as np
from examples.lqg_flipased_demo.dynamics import (
    discretize_polytope, flipased_polytope_reduced, flipased_tpform)
from tptool.lmi import LMIProblem, LMIObserverProblem

CHANNEL = "sym"
HULL    = "cno"
DT      = 0.005
MARGIN  = 1e-3


def main():
    print("=" * 70)
    print(f"LMI correctness diagnostic  (channel={CHANNEL}, hull={HULL})")
    print("=" * 70)

    poly_red   = flipased_polytope_reduced(CHANNEL)
    poly_red_d = discretize_polytope(poly_red, dt=DT)
    n, m, p    = poly_red_d.n, poly_red_d.m, poly_red_d.p
    C_mid      = poly_red_d.C[poly_red_d.R // 2]
    tp_d       = flipased_tpform(poly_red_d, hull=HULL, verbose=True)
    R          = int(np.prod([u.shape[1] for u in tp_d.U]))

    print(f"\nplant: n={n}, m={m}, p={p}, R={R} TP vertices")
    print(f"C_mid shape: {C_mid.shape}")

    # ---- 1. LMI state-feedback K ----
    print("\n--- LMI state-feedback K ---")
    prob_K = LMIProblem(tp_d.S, n=n, discrete=True)
    prob_K.add_stability(margin=MARGIN)
    K_lmi, P = prob_K.solve(solver="MOSEK")
    K_lmi = K_lmi.reshape(R, m, n)

    print(f"P eigenvalues (should all be > 0): "
          f"{np.linalg.eigvalsh(P)}")

    print(f"\nPer-vertex spectral radii of (A + B K_lmi)  "
          f"[LMI convention, should be < 1]:")
    for r in range(R):
        Ar = tp_d.S[r, :n, :n]
        Br = tp_d.S[r, :n, n:]
        A_cl = Ar + Br @ K_lmi[r]
        rho = float(np.max(np.abs(np.linalg.eigvals(A_cl))))
        print(f"  vertex {r}: rho(A + B K) = {rho:.6f}  "
              f"{'✓' if rho < 1 else '✗ UNSTABLE'}")

    K_dep = -K_lmi   # deployment: u = -K_dep x
    print(f"\nPer-vertex spectral radii of (A - B K_dep)  "
          f"[deployment, should be < 1]:")
    for r in range(R):
        Ar = tp_d.S[r, :n, :n]
        Br = tp_d.S[r, :n, n:]
        A_cl = Ar - Br @ K_dep[r]
        rho = float(np.max(np.abs(np.linalg.eigvals(A_cl))))
        print(f"  vertex {r}: rho(A - B K_dep) = {rho:.6f}  "
              f"{'✓' if rho < 1 else '✗ UNSTABLE'}")

    # ---- 2. LMI observer L ----
    print("\n--- LMI observer L ---")
    from tptool.lmi import LMIObserverProblem
    W = np.eye(n)
    prob_L = LMIObserverProblem(tp_d.S, C=C_mid, n=n, discrete=True, W=W)
    prob_L.add_stability(margin=MARGIN)
    L_lmi, Pi = prob_L.solve(solver="MOSEK")
    L_lmi = L_lmi.reshape(R, n, p)

    print(f"Pi eigenvalues (should all be > 0): "
          f"{np.linalg.eigvalsh(Pi)}")

    print(f"\nPer-vertex spectral radii of (A - L C)  "
          f"[observer, should be < 1]:")
    for r in range(R):
        Ar = tp_d.S[r, :n, :n]
        A_obs = Ar - L_lmi[r] @ C_mid
        rho = float(np.max(np.abs(np.linalg.eigvals(A_obs))))
        print(f"  vertex {r}: rho(A - L C) = {rho:.6f}  "
              f"{'✓' if rho < 1 else '✗ UNSTABLE'}")

    # ---- 3. Combined LQG closed-loop ----
    print("\n--- Combined LQG spectral radius at TP vertices ---")
    print("Full LQG matrix: [A-BK, BK; LC, A-LC-BK+LCB K]")
    for r in range(R):
        Ar = tp_d.S[r, :n, :n]
        Br = tp_d.S[r, :n, n:]
        Kd = K_dep[r]   # (m, n)
        Lr = L_lmi[r]   # (n, p)
        A_sf  = Ar - Br @ Kd
        A_obs = Ar - Lr @ C_mid
        # augmented: z = [x; e] where e = x - xhat
        # x+ = (A-BK)x + BK e
        # e+ = (A-LC) e
        M = np.block([
            [A_sf,          Br @ Kd],
            [np.zeros_like(A_sf), A_obs],
        ])
        rho = float(np.max(np.abs(np.linalg.eigvals(M))))
        print(f"  vertex {r}: rho(LQG augmented) = {rho:.6f}  "
              f"{'✓' if rho < 1 else '✗ UNSTABLE'}")

    # ---- 4. Gain norm comparison ----
    print("\n--- Gain norm summary ---")
    print(f"{'vertex':>8}  {'||K_dep||':>10}  {'||L||':>8}")
    for r in range(R):
        print(f"  {r:>6}  {np.linalg.norm(K_dep[r]):>10.4f}  "
              f"{np.linalg.norm(L_lmi[r]):>8.4f}")

    print("\n✓ diagnostic complete.")


if __name__ == "__main__":
    main()
