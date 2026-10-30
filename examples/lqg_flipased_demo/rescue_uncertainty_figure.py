"""
Rescue script: re-render the §6.8 figure from the saved Monte Carlo .npz
without re-running the (slow) MC. Fixes the LaTeX % escaping bug that
crashed the original plotting at the end.

Run from project root:
    python -m examples.lqg_flipased_demo.rescue_uncertainty_figure
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


EPSILON_LEVELS = [0.05, 0.10, 0.20]
N_SAMPLES = 1000

COLORS = {
    "pqrnpo":  "#0072B2",
    "lmi":     "#D55E00",
}


def main():
    out_dir = Path("examples/lqg_flipased_demo")
    fig_dir = out_dir / "paper_figures"
    res_dir = out_dir / "results"
    channel = "sym"

    # ---- load saved MC data ----
    data = np.load(res_dir / f"uncertainty_mc_{channel}.npz")

    # rebuild results dict from saved arrays
    results = {}
    for eps in EPSILON_LEVELS:
        results[eps] = {}
        for method in ["pqrnpo", "lmi"]:
            tag = f"{method}_eps{int(eps * 100)}"
            results[eps][method] = {
                "stable": data[f"{tag}_stable"],
                "worst_MS": data[f"{tag}_worstMS"],
            }

    # ---- print summary again from saved data ----
    print("=" * 72)
    print("Summary from saved MC data")
    print("=" * 72)
    print(f"{'eps':>5} | {'method':>10} | {'P[stable]':>10} | "
          f"{'median M_S':>10} | {'p95 M_S':>8} | {'max M_S':>10}")
    print("-" * 72)
    for eps in EPSILON_LEVELS:
        for method in ["pqrnpo", "lmi"]:
            stab = results[eps][method]["stable"]
            ms = results[eps][method]["worst_MS"]
            ms_stable = ms[stab]
            p_stable = float(np.mean(stab))
            if len(ms_stable) > 0:
                med = float(np.median(ms_stable))
                p95 = float(np.percentile(ms_stable, 95))
                mx = float(np.max(ms_stable))
            else:
                med = p95 = mx = float("nan")
            method_label = "P-QRNPO" if method == "pqrnpo" else "LMI"
            print(f"{eps:>5.0%} | {method_label:>10} | "
                  f"{p_stable:>9.3f} | "
                  f"{med:>10.3f} | {p95:>8.3f} | {mx:>10.3f}")

    # ---- plot ----
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0),
                                gridspec_kw={"wspace": 0.32})

    # Panel (a): stability probability vs eps (bar chart)
    ax = axes[0]
    eps_arr = np.array(EPSILON_LEVELS)
    pq_p = np.array([float(np.mean(results[e]["pqrnpo"]["stable"])) for e in EPSILON_LEVELS])
    lmi_p = np.array([float(np.mean(results[e]["lmi"]["stable"])) for e in EPSILON_LEVELS])
    width = 0.012
    ax.bar(eps_arr - width / 2, pq_p, width=width, color=COLORS["pqrnpo"],
             label="P-QRNPO LPV", edgecolor="white", lw=0.5)
    ax.bar(eps_arr + width / 2, lmi_p, width=width, color=COLORS["lmi"],
             label="LMI LPV (loose)", edgecolor="white", lw=0.5,
             hatch="///", alpha=0.8)
    # annotate exact percentages on top of each bar
    for x, v in zip(eps_arr - width / 2, pq_p):
        ax.text(x, v + 0.02, f"{v:.1%}", ha="center", fontsize=8,
                  color=COLORS["pqrnpo"])
    for x, v in zip(eps_arr + width / 2, lmi_p):
        ax.text(x, v + 0.02, f"{v:.1%}", ha="center", fontsize=8,
                  color=COLORS["lmi"])
    ax.set_xticks(eps_arr)
    ax.set_xticklabels([rf"{e * 100:.0f}\%" for e in EPSILON_LEVELS])
    ax.set_xlabel(r"perturbation level $\varepsilon$")
    ax.set_ylabel(r"P[stable across $V \in [40, 61]$]")
    ax.set_title("(a) Stability probability", fontsize=10)
    ax.set_ylim(0, 1.10)
    ax.axhline(1.0, color="k", lw=0.4, alpha=0.4)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.grid(True, axis="y", lw=0.3, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in")

    # Panel (b): empirical CDF of worst-case M_S at eps = 10% (FIXED: escape % in mathmode)
    ax = axes[1]
    eps_focus = 0.10
    pq_stab_mask = results[eps_focus]["pqrnpo"]["stable"]
    lmi_stab_mask = results[eps_focus]["lmi"]["stable"]
    pq_ms_finite = results[eps_focus]["pqrnpo"]["worst_MS"][pq_stab_mask]
    lmi_ms_finite = results[eps_focus]["lmi"]["worst_MS"][lmi_stab_mask]
    pq_sorted = np.sort(pq_ms_finite)
    lmi_sorted = np.sort(lmi_ms_finite)
    pq_cdf = np.linspace(1.0 / N_SAMPLES, float(np.mean(pq_stab_mask)), len(pq_sorted))
    lmi_cdf = np.linspace(1.0 / N_SAMPLES, float(np.mean(lmi_stab_mask)), len(lmi_sorted))
    ax.plot(pq_sorted, pq_cdf, "-", color=COLORS["pqrnpo"], lw=1.8,
              label="P-QRNPO LPV")
    ax.plot(lmi_sorted, lmi_cdf, "--", color=COLORS["lmi"], lw=1.5,
              label="LMI LPV (loose)")
    ax.axvline(2.0, color="k", lw=0.4, alpha=0.4)
    ax.text(2.05, 0.05, r"$M_S = 2$ (acceptable)", fontsize=8, color="#666666")
    ax.set_xlabel(r"worst-case $M_S$ across envelope")
    ax.set_ylabel("empirical CDF")
    ax.set_title(r"(b) CDF of worst-case $M_S$ at $\varepsilon = 10\%$",
                   fontsize=10)
    ax.set_xlim(left=1.0, right=20.0)   # truncate tail for readability
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="best", fontsize=9)
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in")

    # Panel (c): box plots of worst-case M_S at all three eps levels
    ax = axes[2]
    box_data = []
    box_labels = []
    box_colors = []
    for eps in EPSILON_LEVELS:
        pq_stab_mask = results[eps]["pqrnpo"]["stable"]
        lmi_stab_mask = results[eps]["lmi"]["stable"]
        pq_ms = results[eps]["pqrnpo"]["worst_MS"][pq_stab_mask]
        lmi_ms = results[eps]["lmi"]["worst_MS"][lmi_stab_mask]
        box_data.append(pq_ms)
        box_data.append(lmi_ms)
        # FIXED: escape % in mathmode
        box_labels.append(rf"P-Q" + "\n" + rf"$\varepsilon={int(eps*100)}\%$")
        box_labels.append(rf"LMI" + "\n" + rf"$\varepsilon={int(eps*100)}\%$")
        box_colors.append(COLORS["pqrnpo"])
        box_colors.append(COLORS["lmi"])
    bp = ax.boxplot(box_data, tick_labels=box_labels, widths=0.55,
                       patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
        patch.set_edgecolor(color)
    for median in bp["medians"]:
        median.set_color("k")
        median.set_linewidth(1.0)
    for whisker in bp["whiskers"] + bp["caps"]:
        whisker.set_color("#444444")
    ax.axhline(2.0, color="k", lw=0.4, alpha=0.4)
    ax.text(0.6, 2.1, r"$M_S = 2$ (acceptable)", fontsize=8, color="#666666")
    ax.set_ylabel(r"worst-case $M_S$")
    ax.set_title("(c) $M_S$ distribution by perturbation level", fontsize=10)
    ax.grid(True, axis="y", lw=0.3, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in")
    # cap y-axis so outliers don't crush the box visualization
    ax.set_ylim(0, 50)

    fig.suptitle(rf"FliPASED [sym, hull = close] structured uncertainty: "
                 rf"{N_SAMPLES} Monte Carlo samples per $\varepsilon$, "
                 rf"vertex-level relative perturbations, "
                 rf"$V$ swept $[40, 61]$ m/s",
                 fontsize=10, y=1.02)

    out_path = fig_dir / f"uncertainty_mc_{channel}.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out_path.name}")


if __name__ == "__main__":
    main()
