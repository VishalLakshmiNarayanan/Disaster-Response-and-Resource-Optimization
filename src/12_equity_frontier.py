"""
12_equity_frontier.py
Sweeps EQUITY_FRAC from 0.0 to 0.80 in steps of 0.04, runs the LP on all 46
test storms for each value, averages metrics, and plots the equity-efficiency
Pareto frontier.

Output: outputs/figures/equity_frontier.png
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import pulp
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROC     = Path("data/processed")
RAW_SVI  = Path("data/raw/svi/svi_2020.parquet")
RAW_ACS  = Path("data/raw/acs/acs_county.parquet")
OUT_FIG  = Path("outputs/figures")
OUT_FIG.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SVI_THRESHOLD   = 0.75
SUPPLY_FRACTION = 0.80
CURRENT_EQUITY  = 0.40

# Design tokens
BG    = "#fafafa"
DARK  = "#0f172a"
MID   = "#64748b"
LIGHT = "#cbd5e1"
GRID  = "#e2e8f0"
WHITE = "#ffffff"
COLOR_LOW_EQ  = "#94a3b8"
COLOR_HIGH_EQ = "#10b981"

# ---------------------------------------------------------------------------
# Global matplotlib style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   BG,
    "axes.edgecolor":   LIGHT,
    "axes.labelcolor":  DARK,
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "xtick.color":      MID,
    "ytick.color":      MID,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "grid.color":       GRID,
    "grid.linestyle":   "--",
    "grid.linewidth":   0.7,
    "legend.fontsize":  9,
    "font.family":      "sans-serif",
    "text.color":       DARK,
})


# ---------------------------------------------------------------------------
# LP allocation (same structure as 09_optimize.py)
# ---------------------------------------------------------------------------
def lp_allocate(demands, svi_vals, total_supply, equity_frac, svi_threshold=SVI_THRESHOLD):
    """
    Solve LP to minimise weighted unmet demand subject to:
      - sum(alloc) <= total_supply
      - sum(alloc[high_svi]) >= equity_frac * total_supply
    Returns DataFrame with columns: allocated, unmet  (index = fips).
    """
    counties  = demands.index.tolist()
    svi_dict  = svi_vals.to_dict()
    high_svi  = [i for i in counties if float(svi_dict.get(i, 0.0)) >= svi_threshold]

    weights = {
        i: (2.0 if float(svi_dict.get(i, 0.0)) >= svi_threshold else 1.0)
        for i in counties
    }

    prob  = pulp.LpProblem("allocation", pulp.LpMinimize)
    alloc = pulp.LpVariable.dicts("a", counties, lowBound=0)
    unmet = pulp.LpVariable.dicts("u", counties, lowBound=0)

    prob += pulp.lpSum(weights[i] * unmet[i] for i in counties)

    for i in counties:
        prob += unmet[i] >= demands[i] - alloc[i]

    prob += pulp.lpSum(alloc[i] for i in counties) <= total_supply

    if high_svi and equity_frac > 0:
        prob += (
            pulp.lpSum(alloc[i] for i in high_svi)
            >= equity_frac * total_supply
        )

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    return pd.DataFrame({
        "fips":      counties,
        "allocated": [max(0.0, pulp.value(alloc[i]) or 0.0) for i in counties],
        "unmet":     [max(0.0, pulp.value(unmet[i])  or 0.0) for i in counties],
    }).set_index("fips")


# ---------------------------------------------------------------------------
# Per-storm metric computation
# ---------------------------------------------------------------------------
def compute_storm_metrics(subset, svi_ser, pop_ser, equity_frac):
    """
    subset  : DataFrame indexed by fips, contains pred_demand, svi_overall
    svi_ser : Series indexed by fips
    pop_ser : Series indexed by fips
    Returns dict of metrics or None if storm is skipped.
    """
    demands = subset["pred_demand"].clip(lower=0).astype(float)
    S = demands.sum() * SUPPLY_FRACTION
    if S <= 0:
        return None

    # Align SVI
    svi_vals = svi_ser.reindex(demands.index).fillna(0.5)

    result = lp_allocate(demands, svi_vals, S, equity_frac)

    allocated = result["allocated"]
    unmet     = result["unmet"]

    high_mask = svi_vals >= SVI_THRESHOLD
    low_mask  = ~high_mask

    def safe_coverage(dem, alloc, mask):
        d = dem[mask].sum()
        if d <= 0:
            return 1.0
        return float(alloc[mask].clip(upper=dem[mask]).sum() / d)

    cov_overall   = safe_coverage(demands, allocated, pd.Series(True, index=demands.index))
    cov_high      = safe_coverage(demands, allocated, high_mask)
    cov_low       = safe_coverage(demands, allocated, low_mask)
    equity_gap    = float(cov_low - cov_high)   # positive means low-SVI favored
    total_demand  = demands.sum()
    pct_unmet     = float(unmet.sum() / total_demand) if total_demand > 0 else 0.0

    return {
        "coverage_overall":   cov_overall,
        "coverage_high_svi":  cov_high,
        "coverage_low_svi":   cov_low,
        "equity_gap":         equity_gap,
        "pct_unmet":          pct_unmet,
    }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def main():
    print("Loading data...")
    preds   = pd.read_parquet(PROC / "predictions.parquet")
    svi_raw = pd.read_parquet(RAW_SVI)

    # SVI: prefer 'fips' column if present, otherwise use FIPS
    if "fips" in svi_raw.columns:
        svi_df = svi_raw.set_index("fips")
    else:
        svi_raw["fips"] = svi_raw["FIPS"].astype(str).str.zfill(5)
        svi_df = svi_raw.set_index("fips")

    acs_df = pd.read_parquet(RAW_ACS)
    pop_ser = (
        acs_df.sort_values("year")
              .drop_duplicates("fips", keep="last")
              .set_index("fips")["total_pop"]
    )

    svi_ser = svi_df["svi_overall"].astype(float)

    storm_ids = preds["storm_id"].unique()
    print(f"Storms in test set: {len(storm_ids)}")

    equity_fracs = np.arange(0.0, 0.81, 0.04)
    results = []

    print(f"\nSweeping equity_frac over {len(equity_fracs)} values...")
    print(f"{'equity_frac':>12} | {'storms_run':>10} | {'cov_overall':>11} | "
          f"{'cov_high':>8} | {'cov_low':>8} | {'equity_gap':>10} | {'pct_unmet':>9}")
    print("-" * 80)

    for ef in equity_fracs:
        storm_metrics = []
        for sid in storm_ids:
            subset = preds[preds["storm_id"] == sid].copy().set_index("fips")
            m = compute_storm_metrics(subset, svi_ser, pop_ser, ef)
            if m is not None:
                storm_metrics.append(m)

        if not storm_metrics:
            continue

        agg = {k: float(np.mean([m[k] for m in storm_metrics])) for k in storm_metrics[0]}
        agg["equity_frac"]  = float(ef)
        agg["n_storms"]     = len(storm_metrics)
        results.append(agg)

        print(f"{ef:>12.2f} | {len(storm_metrics):>10d} | "
              f"{agg['coverage_overall']:>11.4f} | "
              f"{agg['coverage_high_svi']:>8.4f} | "
              f"{agg['coverage_low_svi']:>8.4f} | "
              f"{agg['equity_gap']:>10.4f} | "
              f"{agg['pct_unmet']:>9.4f}")

    df = pd.DataFrame(results).set_index("equity_frac")
    print("\n")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("=" * 80)
    print("EQUITY FRONTIER SUMMARY TABLE")
    print("=" * 80)
    print(f"{'eq_frac':>8} | {'cov_overall':>11} | {'cov_high_svi':>12} | "
          f"{'cov_low_svi':>11} | {'equity_gap':>10} | {'pct_unmet':>9}")
    print("-" * 80)
    for ef, row in df.iterrows():
        print(f"{ef:>8.2f} | "
              f"{row['coverage_overall']:>11.4f} | "
              f"{row['coverage_high_svi']:>12.4f} | "
              f"{row['coverage_low_svi']:>11.4f} | "
              f"{row['equity_gap']:>10.4f} | "
              f"{row['pct_unmet']:>9.4f}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Color gradient setup (low equity -> high equity)
    # ------------------------------------------------------------------
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "eq_gradient", [COLOR_LOW_EQ, COLOR_HIGH_EQ]
    )
    norm     = mcolors.Normalize(vmin=0.0, vmax=0.80)
    ef_vals  = df.index.values

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "Equity-Efficiency Trade-off: LP Allocation Frontier (avg over 46 storms)",
        fontsize=14, fontweight="bold", color=DARK, y=0.97
    )

    # ------------------------------------------------------------------
    # Left panel: Pareto frontier (cov_low_svi vs cov_high_svi)
    # ------------------------------------------------------------------
    ax1 = axes[0]
    ax1.set_facecolor(BG)
    ax1.grid(True, zorder=0)

    x_vals = df["coverage_low_svi"].values
    y_vals = df["coverage_high_svi"].values

    # Colored line segments
    for i in range(len(ef_vals) - 1):
        mid_ef = 0.5 * (ef_vals[i] + ef_vals[i + 1])
        c = cmap(norm(mid_ef))
        ax1.plot(
            [x_vals[i], x_vals[i + 1]],
            [y_vals[i], y_vals[i + 1]],
            color=c, linewidth=2.5, solid_capstyle="round", zorder=2
        )

    # Scatter dots colored by equity_frac
    sc = ax1.scatter(
        x_vals, y_vals,
        c=ef_vals, cmap=cmap, norm=norm,
        s=50, zorder=3, edgecolors=DARK, linewidths=0.5
    )

    # Equality line
    lim_min = min(x_vals.min(), y_vals.min()) - 0.02
    lim_max = max(x_vals.max(), y_vals.max()) + 0.02
    eq_pts  = np.linspace(lim_min, lim_max, 100)
    ax1.plot(eq_pts, eq_pts, linestyle="--", color=MID, linewidth=1.0,
             alpha=0.7, label="Equality line (cov_high = cov_low)", zorder=1)

    # Current operating point
    if CURRENT_EQUITY in df.index:
        cx = float(df.loc[CURRENT_EQUITY, "coverage_low_svi"])
        cy = float(df.loc[CURRENT_EQUITY, "coverage_high_svi"])
        ax1.scatter([cx], [cy], marker="*", s=350, color="#f59e0b",
                    zorder=5, edgecolors=DARK, linewidths=0.7,
                    label=f"Current operating point (eq_frac={CURRENT_EQUITY})")
        ax1.annotate(
            f"Current\n(ef={CURRENT_EQUITY})",
            xy=(cx, cy), xytext=(cx + 0.015, cy - 0.018),
            fontsize=8, color=DARK,
            arrowprops=dict(arrowstyle="->", color=MID, lw=0.8)
        )

    # Annotations for selected equity_frac values
    annotate_at = [0.0, 0.20, 0.40, 0.60, 0.80]
    for ef in annotate_at:
        if ef not in df.index:
            continue
        xi = float(df.loc[ef, "coverage_low_svi"])
        yi = float(df.loc[ef, "coverage_high_svi"])
        if ef == CURRENT_EQUITY:
            continue  # already annotated above
        offset = (-0.05, 0.01) if xi > 0.5 else (0.01, 0.01)
        ax1.annotate(
            f"ef={ef:.2f}",
            xy=(xi, yi), xytext=(xi + offset[0], yi + offset[1]),
            fontsize=7.5, color=MID,
            arrowprops=dict(arrowstyle="->", color=LIGHT, lw=0.6),
            bbox=dict(boxstyle="round,pad=0.2", fc=WHITE, ec=LIGHT, alpha=0.8)
        )

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1, fraction=0.04, pad=0.02)
    cbar.set_label("equity_frac", fontsize=9, color=DARK)
    cbar.ax.yaxis.set_tick_params(color=MID, labelsize=8)

    ax1.set_xlabel("Coverage - Low SVI Counties (Efficiency)", fontsize=11)
    ax1.set_ylabel("Coverage - High SVI Counties (Equity)", fontsize=11)
    ax1.set_title("Pareto Frontier: Equity vs Efficiency Trade-off", fontsize=13, pad=8)
    ax1.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax1.set_xlim(lim_min, lim_max)
    ax1.set_ylim(lim_min, lim_max)

    # ------------------------------------------------------------------
    # Right panel: pct_unmet vs equity_frac
    # ------------------------------------------------------------------
    ax2 = axes[1]
    ax2.set_facecolor(BG)
    ax2.grid(True, zorder=0)

    pct_vals = df["pct_unmet"].values

    # Gradient-colored line segments
    for i in range(len(ef_vals) - 1):
        mid_ef = 0.5 * (ef_vals[i] + ef_vals[i + 1])
        c = cmap(norm(mid_ef))
        ax2.plot(
            [ef_vals[i], ef_vals[i + 1]],
            [pct_vals[i], pct_vals[i + 1]],
            color=c, linewidth=2.5, solid_capstyle="round", zorder=2
        )

    ax2.scatter(
        ef_vals, pct_vals,
        c=ef_vals, cmap=cmap, norm=norm,
        s=60, zorder=3, edgecolors=DARK, linewidths=0.5
    )

    # Current point
    if CURRENT_EQUITY in df.index:
        cy2 = float(df.loc[CURRENT_EQUITY, "pct_unmet"])
        ax2.scatter(
            [CURRENT_EQUITY], [cy2],
            marker="*", s=350, color="#f59e0b",
            zorder=5, edgecolors=DARK, linewidths=0.7,
            label=f"Current (ef={CURRENT_EQUITY})"
        )
        ax2.annotate(
            f"Current\n(ef={CURRENT_EQUITY})\npct_unmet={cy2:.3f}",
            xy=(CURRENT_EQUITY, cy2),
            xytext=(CURRENT_EQUITY + 0.06, cy2 + 0.003),
            fontsize=8, color=DARK,
            arrowprops=dict(arrowstyle="->", color=MID, lw=0.8)
        )

    ax2.set_xlabel("Equity Fraction (fraction of supply to high-SVI)", fontsize=11)
    ax2.set_ylabel("Pct Unmet Demand (avg across 46 storms)", fontsize=11)
    ax2.set_title("Cost of Equity: Unmet Demand vs Equity Fraction", fontsize=13, pad=8)
    ax2.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # Colorbar
    sm2 = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm2.set_array([])
    cbar2 = plt.colorbar(sm2, ax=ax2, fraction=0.04, pad=0.02)
    cbar2.set_label("equity_frac", fontsize=9, color=DARK)
    cbar2.ax.yaxis.set_tick_params(color=MID, labelsize=8)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = OUT_FIG / "equity_frontier.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Figure saved -> {out_path}")


if __name__ == "__main__":
    main()
