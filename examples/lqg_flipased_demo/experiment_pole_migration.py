"""
Pole migration: open-loop vs closed-loop eigenvalues across V ∈ [40, 61] m/s.

For each V on a fine grid, computes:
  - open-loop eigenvalues of A_full(V)
  - closed-loop eigenvalues of A_full(V) - B_full(V) K(V)  (state feedback)
  - observer eigenvalues of A_red(V) - L(V) C_red          (observer poles)

Plots a 2-panel figure:
  (a) continuous-time s-plane: Re vs Im for all modes, color-coded by V,
      with flutter mode trajectory highlighted
  (b) flutter mode only: Re(eig) vs V, open-loop vs closed-loop

Run from project root:
    python -m examples.lqg_flipased_demo.experiment_pole_migration
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import logm

from examples.lqg_flipased_demo.dynamics import (
    flipased_polytope,
    flipased_polytope_reduced,
)
from examples.lqg_flipased_demo.pipeline import (
    _blend_poly,
    _blend_tp,
    synthesize,
)


CHANNELS = ["sym", "asym"]
HULL = "cno"
N_V_FINE = 61          # fine grid for smooth migration curves
DT = 0.005             # must match synthesis dt for discrete->continuous conversion

# Flutter mode index in continuous-time full plant eigenvalues
# (sorted by Im(eig), flutter is the pair with Im ~ 2*pi*7.78 rad/s)
FLUTTER_FREQ_HZ = 7.78
FLUTTER_FREQ_RAD = 2 * np.pi * FLUTTER_FREQ_HZ

# Bang Wong palette
COL_OL      = "#D55E00"   # open-loop
COL_CL_SF   = "#0072B2"   # closed-loop state feedback
COL_OBS     = "#009E73"   # observer poles
COL_FLUTTER = "#CC79A7"   # flutter mode highlight
COL_ONSET   = "#999999"


def discrete_to_continuous_eigs(eigs_d: np.ndarray, dt: float) -> np.ndarray:
    """Convert discrete-time eigenvalues z to continuous-time s = log(z)/dt."""
    return np.log(eigs_d.astype(complex)) / dt


def find_flutter_pair(eigs_s: np.ndarray,
                       prev_eig: complex = None) -> np.ndarray:
    """Return indices of the eigenvalue pair closest to the flutter frequency.

    If prev_eig is given, tracks continuously by nearest-neighbor in the
    complex plane (avoids branch-cut jumps in log(z)/dt conversion).
    """
    if prev_eig is not None:
        diffs = np.abs(eigs_s - prev_eig)
    else:
        freqs = np.abs(eigs_s.imag)
        diffs = np.abs(freqs - FLUTTER_FREQ_RAD)
    idx = np.argmin(diffs)
    conj_idx = np.argmin(np.abs(eigs_s - eigs_s[idx].conjugate()))
    return np.array([idx, conj_idx])


def run_channel(channel: str, fig_dir: Path, res_dir: Path):
    print(f"\n{'=' * 70}")
    print(f"Pole migration  (channel={channel}, hull={HULL})")
    print("=" * 70)

    # synthesize
    synth = synthesize(channel=channel, hull=HULL, verbose=True)
    tp     = synth.tp
    domain = tp.domain if hasattr(tp, "domain") else np.array([[40.0, 61.0]])

    poly_full = flipased_polytope(channel)
    poly_full_d = synth.poly_full_d
    poly_red_d  = synth.poly_red_d

    V_fine = np.linspace(40.0, 61.0, N_V_FINE)

    # storage: continuous-time eigenvalues per V
    eigs_ol  = []   # open-loop full plant
    eigs_cl  = []   # closed-loop: A_full - B_full K (K lifted to full)
    eigs_obs = []   # observer: A_red - L C_red

    # flutter-mode Re(s) trajectories
    re_flutter_ol = np.zeros(N_V_FINE)
    re_flutter_cl = np.zeros(N_V_FINE)

    state_idx = synth.state_indices   # maps reduced indices -> full indices

    print(f"\n  {'V':>5}  {'Re(flutter) OL':>16}  {'Re(flutter) CL':>16}")
    prev_eig_ol = None
    prev_eig_cl = None
    for i, V in enumerate(V_fine):
        # open-loop: continuous-time A_full
        A_full_c = _blend_poly(poly_full.A, V, poly_full.V_grid)
        eigs_ol_c = np.linalg.eigvals(A_full_c)
        eigs_ol.append(eigs_ol_c)

        # closed-loop state feedback: discrete A_full - B_full K, then -> continuous
        A_full_d = _blend_poly(poly_full_d.A, V, poly_full_d.V_grid)
        B_full_d = _blend_poly(poly_full_d.B, V, poly_full_d.V_grid)
        K_red    = _blend_tp(synth.K, V, tp, domain)   # (m, n_red)

        # lift K from reduced to full state space
        n_full = poly_full_d.n
        K_full = np.zeros((K_red.shape[0], n_full))
        for j, idx in enumerate(state_idx):
            K_full[:, idx] = K_red[:, j]

        A_cl_d = A_full_d - B_full_d @ K_full
        eigs_cl_d = np.linalg.eigvals(A_cl_d)
        eigs_cl_c = discrete_to_continuous_eigs(eigs_cl_d, DT)
        eigs_cl.append(eigs_cl_c)

        # observer poles: A_red - L C_red (discrete -> continuous)
        A_red_d = _blend_poly(poly_red_d.A, V, poly_red_d.V_grid)
        C_red   = poly_red_d.C[poly_red_d.R // 2]   # C_mid (constant)
        L       = _blend_tp(synth.L, V, tp, domain)  # (n_red, p)
        A_obs_d = A_red_d - L @ C_red
        eigs_obs_d = np.linalg.eigvals(A_obs_d)
        eigs_obs_c = discrete_to_continuous_eigs(eigs_obs_d, DT)
        eigs_obs.append(eigs_obs_c)

        # flutter mode tracking (continuous: nearest-neighbor)
        fl_ol = find_flutter_pair(eigs_ol_c, prev_eig_ol)
        re_flutter_ol[i] = float(eigs_ol_c[fl_ol[0]].real)
        prev_eig_ol = eigs_ol_c[fl_ol[0]]

        fl_cl = find_flutter_pair(eigs_cl_c, prev_eig_cl)
        re_flutter_cl[i] = float(eigs_cl_c[fl_cl[0]].real)
        prev_eig_cl = eigs_cl_c[fl_cl[0]]

        if i % 10 == 0 or abs(V - 50.2) < 0.5:
            print(f"  {V:5.2f}  {re_flutter_ol[i]:>16.4f}  "
                  f"{re_flutter_cl[i]:>16.4f}")

    # find flutter onset
    flutter_onset = None
    for i in range(N_V_FINE - 1):
        if re_flutter_ol[i] < 0 < re_flutter_ol[i + 1]:
            frac = -re_flutter_ol[i] / (re_flutter_ol[i+1] - re_flutter_ol[i])
            flutter_onset = V_fine[i] + frac * (V_fine[i+1] - V_fine[i])
            break
    print(f"\n  flutter onset (open-loop): V_F = {flutter_onset:.2f} m/s")

    # ---- save numerics ----
    np.savez(res_dir / f"pole_migration_{channel}.npz",
             V=V_fine,
             re_flutter_ol=re_flutter_ol,
             re_flutter_cl=re_flutter_cl,
             flutter_onset=flutter_onset if flutter_onset else np.nan)

    # ---- plot ----
    plt.rcParams.update({
        "text.usetex": True, "font.family": "serif",
        "font.size": 9, "pdf.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))

    # Panel (a): s-plane scatter, all modes, colored by V
    ax = axes[0]
    cmap = plt.cm.viridis
    norm = plt.Normalize(V_fine[0], V_fine[-1])

    for i, V in enumerate(V_fine):
        color = cmap(norm(V))
        eol = eigs_ol[i]
        # plot only modes within a reasonable window
        mask = (np.abs(eol.imag) < 300) & (np.abs(eol.real) < 50)
        ax.scatter(eol[mask].real, eol[mask].imag / (2*np.pi),
                   c=[color], s=4, alpha=0.4, linewidths=0)

    # overlay closed-loop flutter mode trajectory
    fl_re_cl = np.array([eigs_cl[i][find_flutter_pair(eigs_cl[i])[0]].real
                         for i in range(N_V_FINE)])
    fl_im_cl = np.array([eigs_cl[i][find_flutter_pair(eigs_cl[i])[0]].imag
                         for i in range(N_V_FINE)])
    ax.plot(fl_re_cl, fl_im_cl / (2*np.pi), "-", color=COL_CL_SF,
            lw=2.0, label="closed-loop flutter mode")
    ax.plot(re_flutter_ol, np.full(N_V_FINE, FLUTTER_FREQ_HZ), ":",
            color=COL_OL, lw=1.4, label="open-loop flutter mode (approx Im)")

    # direction arrows at ~25%, 50%, 75% of V range
    for frac in [0.25, 0.55, 0.80]:
        i0 = int(frac * (N_V_FINE - 2))
        i1 = i0 + 3
        # closed-loop arrow
        dx = fl_re_cl[i1] - fl_re_cl[i0]
        dy = (fl_im_cl[i1] - fl_im_cl[i0]) / (2*np.pi)
        ax.annotate("", xy=(fl_re_cl[i1], fl_im_cl[i1]/(2*np.pi)),
                    xytext=(fl_re_cl[i0], fl_im_cl[i0]/(2*np.pi)),
                    arrowprops=dict(arrowstyle="->", color=COL_CL_SF, lw=1.5))
        # open-loop arrow (horizontal, Im fixed)
        x0_ol = re_flutter_ol[i0]
        x1_ol = re_flutter_ol[i1]
        ax.annotate("", xy=(x1_ol, FLUTTER_FREQ_HZ),
                    xytext=(x0_ol, FLUTTER_FREQ_HZ),
                    arrowprops=dict(arrowstyle="->", color=COL_OL, lw=1.5))

    ax.axvline(0, color="k", lw=0.8, alpha=0.5)
    ax.axhline(0, color="k", lw=0.4, alpha=0.3)
    ax.set_xlabel(r"Re$(s)$  (rad/s)")
    ax.set_ylabel(r"Im$(s)$ / $2\pi$  (Hz)")
    ax.set_title("(a) $s$-plane pole migration  (color = airspeed V)")
    ax.set_xlim(-30, 10)
    ax.set_ylim(-5, 35)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="V (m/s)", shrink=0.8)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in")

    # Panel (b): Re(flutter eig) vs V
    ax = axes[1]
    ax.plot(V_fine, re_flutter_ol, "-", color=COL_OL, lw=1.8,
            label="open-loop")
    ax.plot(V_fine, re_flutter_cl, "-", color=COL_CL_SF, lw=1.8,
            label="closed-loop (state feedback)")
    ax.axhline(0, color="k", lw=0.8, alpha=0.6, linestyle="--")
    if flutter_onset is not None:
        ax.axvline(flutter_onset, color=COL_ONSET, lw=0.9, linestyle=":",
                   label=f"flutter onset $V_F$ = {flutter_onset:.1f} m/s")
    ax.fill_between(V_fine, re_flutter_ol, 0,
                    where=(re_flutter_ol > 0),
                    color=COL_OL, alpha=0.12, label="unstable region (OL)")
    ax.set_xlabel("airspeed $V$ (m/s)")
    ax.set_ylabel(r"Re$(s_{\rm flutter})$  (rad/s)")
    ax.set_title("(b) flutter mode growth rate vs airspeed")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in")

    fig.suptitle(
        rf"FliPASED [{channel}]: pole migration across $V \in [40, 61]$ m/s  "
        rf"(hull={HULL}, $\Delta t={DT*1e3:.0f}$ ms)",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(fig_dir / f"pole_migration_{channel}.png",
                dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved pole_migration_{channel}.png")


def main():
    out_dir = Path("examples/lqg_flipased_demo")
    fig_dir = out_dir / "figures"
    res_dir = out_dir / "results"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    for ch in CHANNELS:
        run_channel(ch, fig_dir, res_dir)


if __name__ == "__main__":
    main()
