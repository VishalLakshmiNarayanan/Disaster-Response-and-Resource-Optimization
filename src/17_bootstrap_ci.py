"""
17_bootstrap_ci.py
Bootstrap 95% confidence intervals on all key policy metrics across the
46 test storms.  Shows statistical reliability of policy comparisons.

Output: outputs/figures/bootstrap_ci.png
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ALLOC_DIR = Path("outputs/allocations")
OUT_FIG   = Path("outputs/figures")
OUT_FIG.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_BOOTSTRAP = 10_000
RNG_SEED    = 42
POLICIES    = ["population_proportional", "svi_weighted", "model_lp_equity"]
POLICY_LABELS = {
    "population_proportional": "pop-prop",
    "svi_weighted":            "svi-weighted",
    "model_lp_equity":         "model_lp_equity",
}
METRICS = ["pct_unmet", "coverage_overall", "coverage_high_svi",
           "coverage_low_svi", "equity_gap"]

# Design tokens
BG    = "#fafafa"
DARK  = "#0f172a"
MID   = "#64748b"
LIGHT = "#cbd5e1"
GRID  = "#e2e8f0"
WHITE = "#ffffff"
PC = {
    "population_proportional": "#94a3b8",
    "svi_weighted":            "#f59e0b",
    "model_lp_equity":         "#10b981",
}

# ---------------------------------------------------------------------------
# Global matplotlib style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   BG,
    "axes.edgecolor":   LIGHT,
    "axes.labelcolor":  DARK,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "xtick.color":      MID,
    "ytick.color":      MID,
    "xtick.labelsize":  8.5,
    "ytick.labelsize":  8.5,
    "grid.color":       GRID,
    "grid.linestyle":   "--",
    "grid.linewidth":   0.6,
    "legend.fontsize":  8,
    "font.family":      "sans-serif",
    "text.color":       DARK,
})

# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------

def bootstrap_mean(values, n_boot=N_BOOTSTRAP, rng=None):
    """Return (mean, ci_lo, ci_hi) for 95% CI via percentile bootstrap."""
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    arr = np.asarray(values, dtype=float)
    n   = len(arr)
    boot_means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = arr[rng.integers(0, n, size=n)]
        boot_means[i] = sample.mean()
    lo = float(np.percentile(boot_means, 2.5))
    hi = float(np.percentile(boot_means, 97.5))
    return float(arr.mean()), lo, hi


def bootstrap_diff(vals_a, vals_b, n_boot=N_BOOTSTRAP, rng=None):
    """
    Bootstrap CI for mean(vals_a) - mean(vals_b).
    Pairs are sampled together (same row indices) to preserve within-storm
    correlation.
    """
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    a = np.asarray(vals_a, dtype=float)
    b = np.asarray(vals_b, dtype=float)
    n = len(a)
    boot_diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_diffs[i] = a[idx].mean() - b[idx].mean()
    lo = float(np.percentile(boot_diffs, 2.5))
    hi = float(np.percentile(boot_diffs, 97.5))
    return float((a - b).mean()), lo, hi


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading all_storms_comparison.parquet ...")
    comp = pd.read_parquet(ALLOC_DIR / "all_storms_comparison.parquet")

    # Confirm all three policies present
    present = comp["policy"].unique().tolist()
    print(f"Policies found: {present}")
    for p in POLICIES:
        if p not in present:
            raise ValueError(f"Policy '{p}' not found in comparison file.")

    storm_ids = comp["storm_id"].unique()
    n_storms  = len(storm_ids)
    print(f"Storms: {n_storms}")
    print(f"Bootstrap iterations: {N_BOOTSTRAP:,}")

    rng = np.random.default_rng(RNG_SEED)

    # ------------------------------------------------------------------
    # Build per-policy, per-storm arrays (shape: n_storms x n_metrics)
    # ------------------------------------------------------------------
    policy_data = {}
    for p in POLICIES:
        sub = comp[comp["policy"] == p].set_index("storm_id")
        # Align to a consistent storm order
        sub = sub.reindex(storm_ids)
        policy_data[p] = sub[METRICS].astype(float)

    # ------------------------------------------------------------------
    # Compute bootstrap CIs for each policy x metric
    # ------------------------------------------------------------------
    ci_results = {}   # {policy: {metric: (mean, lo, hi)}}
    for p in POLICIES:
        ci_results[p] = {}
        for m in METRICS:
            vals = policy_data[p][m].fillna(0.0).values
            mn, lo, hi = bootstrap_mean(vals, n_boot=N_BOOTSTRAP, rng=rng)
            ci_results[p][m] = (mn, lo, hi)

    # ------------------------------------------------------------------
    # Pairwise improvement: model_lp vs pop-prop (for each metric)
    # Positive difference = model_lp is better.
    # For pct_unmet: improvement = pop_prop - model_lp (lower is better)
    # For coverage/equity_gap: improvement = model_lp - pop_prop
    # ------------------------------------------------------------------
    lp_policy   = "model_lp_equity"
    base_policy = "population_proportional"

    diff_ci = {}
    for m in METRICS:
        lp_vals   = policy_data[lp_policy][m].fillna(0.0).values
        base_vals = policy_data[base_policy][m].fillna(0.0).values
        if m == "pct_unmet":
            # improvement means lower unmet -> base - lp
            mn, lo, hi = bootstrap_diff(base_vals, lp_vals, n_boot=N_BOOTSTRAP, rng=rng)
        else:
            mn, lo, hi = bootstrap_diff(lp_vals, base_vals, n_boot=N_BOOTSTRAP, rng=rng)
        diff_ci[m] = (mn, lo, hi)

    # ------------------------------------------------------------------
    # Print formatted table
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print("BOOTSTRAP 95% CI RESULTS (n=10,000 resamples, 46 storms)")
    print("=" * 80)
    col_w = 20
    hdr = (
        f"{'Policy':<25} | "
        f"{'pct_unmet':<{col_w}} | "
        f"{'hi-SVI cov':<{col_w}} | "
        f"{'equity_gap':<{col_w}}"
    )
    print(hdr)
    print("-" * len(hdr))
    for p in POLICIES:
        lbl  = POLICY_LABELS[p]
        un   = ci_results[p]["pct_unmet"]
        hi   = ci_results[p]["coverage_high_svi"]
        eg   = ci_results[p]["equity_gap"]
        row  = (
            f"{lbl:<25} | "
            f"{un[0]:.3f} [{un[1]:.3f}-{un[2]:.3f}]".ljust(col_w + 3) + " | "
            f"{hi[0]:.3f} [{hi[1]:.3f}-{hi[2]:.3f}]".ljust(col_w + 3) + " | "
            f"{eg[0]:.3f} [{eg[1]:.3f}-{eg[2]:.3f}]"
        )
        print(row)
    print("=" * 80)

    print()
    print("Improvement of model_lp_equity vs pop-prop (95% CI)")
    print("  For pct_unmet: positive = model_lp has lower unmet demand")
    print("  For others:    positive = model_lp has higher value")
    print("-" * 60)
    for m in METRICS:
        mn, lo, hi = diff_ci[m]
        sig = "(*)" if not (lo <= 0 <= hi) else "   "
        print(f"  {m:<25} delta={mn:+.4f}  95% CI=[{lo:+.4f}, {hi:+.4f}]  {sig}")
    print("-" * 60)

    # ------------------------------------------------------------------
    # Full CI table for all metrics
    # ------------------------------------------------------------------
    print()
    print("FULL BOOTSTRAP CI TABLE")
    print("=" * 110)
    metrics_short = {
        "pct_unmet":          "pct_unmet",
        "coverage_overall":   "cov_overall",
        "coverage_high_svi":  "cov_high_svi",
        "coverage_low_svi":   "cov_low_svi",
        "equity_gap":         "equity_gap",
    }
    header_parts = [f"{'Policy':<25}"]
    for m in METRICS:
        header_parts.append(f"{metrics_short[m]:>22}")
    print(" | ".join(header_parts))
    print("-" * 110)
    for p in POLICIES:
        row_parts = [f"{POLICY_LABELS[p]:<25}"]
        for m in METRICS:
            mn, lo, hi = ci_results[p][m]
            cell = f"{mn:.3f} [{lo:.3f}-{hi:.3f}]"
            row_parts.append(f"{cell:>22}")
        print(" | ".join(row_parts))
    print("=" * 110)

    # ------------------------------------------------------------------
    # Figure: 2x3 grid of horizontal bar charts
    # ------------------------------------------------------------------
    # Panel layout: (row, col) -> metric / content
    PANEL_METRICS = [
        ("pct_unmet",         "Pct Unmet Demand\n(lower is better)"),
        ("coverage_high_svi", "Coverage - High SVI\n(higher is better)"),
        ("equity_gap",        "Equity Gap (cov_low - cov_high)\n(lower = more equitable)"),
        ("coverage_overall",  "Coverage - Overall\n(higher is better)"),
        ("coverage_low_svi",  "Coverage - Low SVI\n(higher is better)"),
        ("improvement",       "model_lp Improvement vs pop-prop\n(positive = model_lp better)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "Bootstrap 95% Confidence Intervals on Policy Performance\n"
        "(n=10,000 resamples, 46 storms)",
        fontsize=13, fontweight="bold", color=DARK, y=0.99
    )

    axes_flat = axes.flatten()

    for panel_idx, (metric_key, panel_title) in enumerate(PANEL_METRICS):
        ax = axes_flat[panel_idx]
        ax.set_facecolor(BG)
        ax.grid(True, axis="x", zorder=0)
        ax.set_axisbelow(True)

        if metric_key == "improvement":
            # Special panel: model_lp improvement over pop-prop for each metric
            imp_metrics = [
                ("pct_unmet",         "pct_unmet (unmet reduction)"),
                ("coverage_overall",  "cov_overall"),
                ("coverage_high_svi", "cov_high_svi"),
                ("coverage_low_svi",  "cov_low_svi"),
                ("equity_gap",        "equity_gap"),
            ]
            y_positions = np.arange(len(imp_metrics))
            for j, (m, label) in enumerate(imp_metrics):
                mn, lo, hi = diff_ci[m]
                err_lo = mn - lo
                err_hi = hi - mn
                ax.barh(
                    j, mn,
                    xerr=[[err_lo], [err_hi]],
                    color=PC[lp_policy], alpha=0.85, height=0.55,
                    error_kw=dict(ecolor=DARK, capsize=4, lw=1.2),
                    zorder=2
                )
                ax.text(
                    mn + (hi - mn) * 0.15,
                    j,
                    f"{mn:+.3f}",
                    va="center", ha="left", fontsize=7.5, color=DARK
                )

            ax.set_yticks(y_positions)
            ax.set_yticklabels([lbl for _, lbl in imp_metrics], fontsize=8)
            ax.axvline(0, color=DARK, linewidth=1.0, linestyle="--", alpha=0.6, zorder=3)
            ax.set_title(panel_title, fontsize=10, pad=5)
            ax.set_xlabel("Delta (model_lp minus pop-prop)", fontsize=9)
        else:
            y_positions = np.arange(len(POLICIES))
            for j, p in enumerate(POLICIES):
                mn, lo, hi = ci_results[p][metric_key]
                err_lo = mn - lo
                err_hi = hi - mn
                ax.barh(
                    j, mn,
                    xerr=[[err_lo], [err_hi]],
                    color=PC[p], alpha=0.85, height=0.55,
                    error_kw=dict(ecolor=DARK, capsize=4, lw=1.2),
                    zorder=2
                )
                ax.text(
                    mn + max(err_hi, 0.002),
                    j,
                    f"{mn:.3f}",
                    va="center", ha="left", fontsize=7.5, color=DARK
                )

            ax.set_yticks(y_positions)
            ax.set_yticklabels([POLICY_LABELS[p] for p in POLICIES], fontsize=8.5)
            ax.set_title(panel_title, fontsize=10, pad=5)
            ax.set_xlabel("Value (mean +/- 95% CI)", fontsize=9)

            # Vertical reference line at 0 for equity_gap
            if metric_key == "equity_gap":
                ax.axvline(0, color=MID, linewidth=1.0, linestyle="--", alpha=0.7, zorder=3)

    # Legend for policy colors (place in top-right of figure)
    legend_patches = [
        mpatches.Patch(color=PC[p], label=POLICY_LABELS[p]) for p in POLICIES
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=3,
        fontsize=9,
        framealpha=0.9,
        bbox_to_anchor=(0.5, 0.00),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    out_path = OUT_FIG / "bootstrap_ci.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"\nFigure saved -> {out_path}")


if __name__ == "__main__":
    main()
