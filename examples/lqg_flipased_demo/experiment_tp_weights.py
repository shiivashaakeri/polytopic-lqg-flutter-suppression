#!/usr/bin/env python3
"""
experiment_tp_weights.py
========================
Plot the TP weighting functions w_r(V) for all five convex hull types
(CNO, IRNO, INO, SNNN, CLOSE) for both symmetric and asymmetric channels.

Produces a 2x5 figure: top row = symmetric, bottom row = asymmetric.

Usage:
    python experiment_tp_weights.py
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- paths ----
SCRIPT_DIR = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(SCRIPT_DIR.parent.parent))

from examples.lqg_flipased_demo.dynamics import (
    flipased_polytope_reduced,
    discretize_polytope,
    flipased_tpform,
    V_MIN,
    V_MAX,
)

# ---- config ----
HULLS = ["cno", "irno", "ino", "snnn", "close"]
HULL_LABELS = ["CNO", "IRNO", "INO", "SNNN", "CLOSE"]
CHANNELS = ["sym", "asym"]
CHANNEL_LABELS = ["Symmetric ($n=14$)", "Asymmetric ($n=13$)"]
GRIDSIZE = 51
DT = 0.005

# Bang Wong colorblind-safe palette
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
          "#56B4E9", "#F0E442", "#000000"]

# ---- matplotlib style ----
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def main():
    V_plot = np.linspace(V_MIN, V_MAX, GRIDSIZE)

    fig, axes = plt.subplots(2, 5, figsize=(16, 6.5), sharey=True, sharex=True)

    for row, (channel, ch_label) in enumerate(zip(CHANNELS, CHANNEL_LABELS)):
        print(f"\n--- {channel} channel ---")
        poly_red = flipased_polytope_reduced(channel)
        poly_red_d = discretize_polytope(poly_red, dt=DT)

        for col, (hull, hull_label) in enumerate(zip(HULLS, HULL_LABELS)):
            ax = axes[row, col]
            print(f"  hull={hull} ...", end=" ", flush=True)
            try:
                tp = flipased_tpform(poly_red_d, hull=hull, gridsize=GRIDSIZE,
                                     verbose=False)
            except Exception as e:
                print(f"FAILED: {e}")
                ax.set_title(f"{hull_label}\nfailed", fontsize=9)
                ax.set_xlim(V_MIN, V_MAX)
                continue

            U = tp.U[0]  # shape (GRIDSIZE, R_tilde)
            R_tilde = U.shape[1]
            print(f"R={R_tilde}")

            for r in range(R_tilde):
                ax.plot(V_plot, U[:, r], color=COLORS[r % len(COLORS)],
                        lw=1.6, label=f"$w_{{{r+1}}}$")

            if row == 0:
                ax.set_title(f"{hull_label}, $\\tilde R = {R_tilde}$",
                             fontsize=16)
            else:
                ax.set_title(f"$\\tilde R = {R_tilde}$", fontsize=16)

            ax.set_xlim(V_MIN, V_MAX)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, lw=0.3, alpha=0.4)
            ax.tick_params(direction="in")

            if row == 1:
                ax.set_xlabel(r"$V_{\mathrm{TAS}}$ (m/s)")

        # Row label on leftmost plot
        axes[row, 0].set_ylabel(
            f"{ch_label}\n$w_r(V)$", fontsize=16
        )

    # Shared legend from the first subplot (all have same R_tilde=4)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=14,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        r"TP weighting functions $w_r(V)$ for five convex hull types",
        fontsize=18,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))

    # ---- save ----
    fig_dir = SCRIPT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "tp_weights_all_hulls.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()