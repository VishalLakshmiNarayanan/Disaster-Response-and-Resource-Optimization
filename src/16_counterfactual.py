"""
16_counterfactual.py
Counterfactual analysis: compare our LP allocation to the ACTUAL FEMA PA
distribution for 3 major storms (IAN 2022, IDA 2021, LAURA 2020).

Three policies are compared against demand_proxy as ground-truth:
  - FEMA_Actual:       what actually happened (declared counties received PA)
  - Pop_Proportional:  simulated uniform pop-proportional with S = actual.sum() * 0.80
  - Our_LP:            equity-constrained LP using pred_demand as the signal

Output:
  outputs/allocations/counterfactual_results.parquet
  outputs/figures/counterfactual.png
  outputs/figures/counterfactual_scatter.png
"""
import matplotlib
matplotlib.use("Agg")

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pulp
from pathlib import Path
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE    = Path(__file__).resolve().parent.parent
PROC    = BASE / "data" / "processed"
RAW     = BASE / "data" / "raw"
OUT_ALL = BASE / "outputs" / "allocations"
OUT_FIG = BASE / "outputs" / "figures"
OUT_ALL.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SVI_THRESHOLD   = 0.75
SUPPLY_FRACTION = 0.80
EQUITY_FRAC     = 0.40

# Target storms: (storm_name_upper, storm_year, fema_disaster_number)
TARGET_STORMS = [
    ("IAN",   2022, 4673),
    ("IDA",   2021, 4611),
    ("LAURA", 2020, 4559),
]

# Design tokens
BG    = "#fafafa"
DARK  = "#0f172a"
MID   = "#64748b"
LIGHT = "#cbd5e1"
GRID  = "#e2e8f0"
WHITE = "#ffffff"

COL_FEMA = "#ef4444"
COL_POP  = "#94a3b8"
COL_LP   = "#10b981"

POLICY_COLORS = {
    "FEMA_Actual":       COL_FEMA,
    "Pop_Proportional":  COL_POP,
    "Our_LP":            COL_LP,
}
POLICY_LABELS = {
    "FEMA_Actual":       "FEMA Actual",
    "Pop_Proportional":  "Pop-Proportional",
    "Our_LP":            "Our LP",
}
POLICIES_ORDERED = ["FEMA_Actual", "Pop_Proportional", "Our_LP"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def coverage(unmet_s: pd.Series, demand_s: pd.Series) -> float:
    total = demand_s.sum()
    return 1.0 - unmet_s.sum() / total if total > 0 else 1.0


def lp_allocate(demands: pd.Series, svi_vals: pd.Series, S: float) -> pd.Series:
    """
    Equity-constrained LP.  Returns allocation Series indexed by fips.
    Minimizes sum_i weight[i] * unmet[i]
    s.t. sum alloc <= S, high_svi alloc >= EQUITY_FRAC * S.
    """
    counties = demands.index.tolist()
    svi_dict = svi_vals.to_dict()
    high_svi = [i for i in counties if float(svi_dict.get(i, 0.0)) >= SVI_THRESHOLD]
    weights  = {i: (2.0 if float(svi_dict.get(i, 0.0)) >= SVI_THRESHOLD else 1.0)
                for i in counties}

    prob  = pulp.LpProblem("cf_lp", pulp.LpMinimize)
    alloc = pulp.LpVariable.dicts("a", counties, lowBound=0)
    unmet = pulp.LpVariable.dicts("u", counties, lowBound=0)

    prob += pulp.lpSum(weights[i] * unmet[i] for i in counties)

    for i in counties:
        prob += unmet[i] >= demands[i] - alloc[i]

    prob += pulp.lpSum(alloc[i] for i in counties) <= S

    if high_svi:
        prob += pulp.lpSum(alloc[i] for i in high_svi) >= EQUITY_FRAC * S

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    return pd.Series(
        {i: max(0.0, pulp.value(alloc[i]) or 0.0) for i in counties},
        name="allocated"
    )


def build_policy_row(policy_name, storm_name, storm_year,
                     alloc_s: pd.Series,
                     actual_demand: pd.Series,
                     svi_vals: pd.Series,
                     S: float) -> dict:
    """
    Compute coverage metrics treating actual_demand as ground truth.
    alloc_s: simulated allocation (index = fips).
    """
    idx    = actual_demand.index
    alloc_ = alloc_s.reindex(idx).fillna(0.0)
    high   = svi_vals.reindex(idx).fillna(0.5) >= SVI_THRESHOLD
    low    = ~high

    unmet = (actual_demand - alloc_).clip(lower=0)

    tot_d = float(actual_demand.sum())
    return {
        "storm_name":        storm_name,
        "storm_year":        storm_year,
        "policy":            policy_name,
        "total_demand":      tot_d,
        "total_supply":      float(S),
        "total_unmet":       float(unmet.sum()),
        "pct_unmet":         float(unmet.sum() / tot_d) if tot_d > 0 else 0.0,
        "coverage_overall":  coverage(unmet, actual_demand),
        "coverage_high_svi": coverage(unmet[high], actual_demand[high]),
        "coverage_low_svi":  coverage(unmet[low],  actual_demand[low]),
        "equity_gap":        (coverage(unmet[high], actual_demand[high])
                              - coverage(unmet[low],  actual_demand[low])),
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_parquet_safe(path: Path, **kwargs) -> pd.DataFrame:
    """Load parquet; fall back to polars if pyarrow histogram bug fires."""
    try:
        return pd.read_parquet(path, engine="pyarrow", **kwargs)
    except Exception:
        try:
            import polars as pl
            return pl.read_parquet(str(path)).to_pandas()
        except Exception as exc:
            raise RuntimeError(f"Cannot load {path}: {exc}") from exc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading data...")

    # -- predictions
    try:
        preds = load_parquet_safe(PROC / "predictions.parquet")
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    # -- SVI
    try:
        svi_df = load_parquet_safe(RAW / "svi" / "svi_2020.parquet").set_index("fips")
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    # -- ACS for population
    try:
        acs_df = load_parquet_safe(RAW / "acs" / "acs_county.parquet")
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    # Population lookup: most recent year per county
    pop_ser = (
        acs_df.sort_values("year")
              .drop_duplicates("fips", keep="last")
              .set_index("fips")["total_pop"]
    )

    # -- FEMA PA projects (for county-level dollar totals)
    pa_by_fips = {}    # disaster_number -> {fips: total_pa_dollars}
    try:
        pa_df = load_parquet_safe(RAW / "fema" / "pa_projects.parquet")
        pa_df = pa_df.copy()
        pa_df["fips_pa"] = (
            pa_df["stateNumberCode"].astype(str).str.zfill(2)
            + pa_df["countyCode"].astype(str).str.zfill(3)
        )
        for dis_num, grp in pa_df.groupby("disasterNumber"):
            pa_by_fips[int(dis_num)] = (
                grp.groupby("fips_pa")["projectAmount"].sum().to_dict()
            )
    except Exception as exc:
        print(f"WARNING: Could not load pa_projects.parquet: {exc}")

    # -- FEMA declarations
    declared_fips = {}    # disaster_number -> set of fips
    try:
        decl_df = load_parquet_safe(RAW / "fema" / "declarations.parquet")
        decl_df = decl_df.copy()
        decl_df["fips_decl"] = (
            decl_df["fipsStateCode"].astype(str).str.zfill(2)
            + decl_df["fipsCountyCode"].astype(str).str.zfill(3)
        )
        for dis_num, grp in decl_df.groupby("disasterNumber"):
            declared_fips[int(dis_num)] = set(grp["fips_decl"].tolist())
    except Exception as exc:
        print(f"WARNING: Could not load declarations.parquet: {exc}")

    # ---------------------------------------------------------------------------
    # Process each target storm
    # ---------------------------------------------------------------------------
    all_rows        = []
    scatter_data    = {}   # storm_name -> DataFrame for scatter figure

    print("\nRunning counterfactual analysis...")

    for storm_name, storm_year, dis_num in tqdm(TARGET_STORMS,
                                                desc="Storms", ncols=80):
        # -- Subset predictions
        subset = (
            preds[preds["storm_name"].str.upper() == storm_name].copy()
        )
        if len(subset) == 0:
            print(f"  WARNING: No predictions found for {storm_name}. Skipping.")
            continue

        subset = subset.set_index("fips")
        subset = subset[~subset.index.duplicated(keep="first")]

        # Ground-truth demand (actual PA per-capita proxy)
        actual_demand = subset["demand_proxy"].clip(lower=0)

        # Prediction signal for LP
        pred_signal = subset["pred_demand"].clip(lower=0)

        # SVI values
        svi_vals = svi_df["svi_overall"].reindex(actual_demand.index).fillna(0.5)

        # Population values
        pop_vals = pop_ser.reindex(actual_demand.index).fillna(1.0)

        # Total supply budget (consistent across policies)
        tot_actual = float(actual_demand.sum())
        S = tot_actual * SUPPLY_FRACTION if tot_actual > 0 else float(pred_signal.sum()) * SUPPLY_FRACTION

        if S <= 0:
            print(f"  WARNING: S=0 for {storm_name}. Skipping.")
            continue

        high_mask = svi_vals >= SVI_THRESHOLD

        # ---- Policy 1: FEMA Actual
        # Represents what actually happened:
        # - declared counties (fema_declared == 1) received PA funding
        # - coverage = fraction of actual demand met for declared counties
        # - undeclared counties receive 0 allocation
        #
        # PA dollars per fips (if available) are used as the actual allocation.
        # If PA data not available, declared counties are assumed fully covered.
        fema_declared_col = subset["fema_declared"].fillna(0).astype(int)
        declared_idx      = fema_declared_col[fema_declared_col == 1].index

        # Build FEMA allocation vector
        # Use real PA dollars if available; otherwise assume declared counties
        # received exactly their actual_demand (PA ≈ FEMA PA fully covered them)
        fema_alloc = pd.Series(0.0, index=actual_demand.index)

        if dis_num in pa_by_fips and pa_by_fips[dis_num]:
            pa_fips_map = pa_by_fips[dis_num]
            pop_total   = pop_vals.reindex(list(pa_fips_map.keys())).fillna(1.0)
            for fp, total_pa in pa_fips_map.items():
                if fp in fema_alloc.index:
                    pop_fp = float(pop_vals.get(fp, 1.0))
                    if pop_fp > 0:
                        # per-capita PA -> normalize like demand_proxy
                        # demand_proxy is in the same per-capita PA-dollar space
                        fema_alloc[fp] = total_pa / pop_fp / 1e6  # scale to match proxy
                    else:
                        fema_alloc[fp] = 0.0
            # Re-scale so fema_alloc is on same scale as actual_demand
            # (both should be per-capita; just ensure the alloc at least covers demand for declared)
            for fp in declared_idx:
                if fp in fema_alloc.index:
                    if fema_alloc[fp] == 0.0:
                        fema_alloc[fp] = float(actual_demand.get(fp, 0.0))
        else:
            # Fallback: declared counties get their full demand; undeclared get 0
            for fp in declared_idx:
                if fp in fema_alloc.index:
                    fema_alloc[fp] = float(actual_demand.get(fp, 0.0))

        # Ensure undeclared counties get 0
        undeclared_idx = fema_declared_col[fema_declared_col == 0].index
        fema_alloc[undeclared_idx] = 0.0

        fema_S = float(fema_alloc.sum())
        if fema_S == 0:
            fema_S = S

        row_fema = build_policy_row("FEMA_Actual", storm_name, storm_year,
                                    fema_alloc, actual_demand, svi_vals, fema_S)

        # ---- Policy 2: Pop-proportional
        safe_pop = pop_vals.clip(lower=0).fillna(1.0)
        if safe_pop.sum() > 0:
            alloc_pop = safe_pop / safe_pop.sum() * S
        else:
            alloc_pop = pd.Series(S / len(actual_demand), index=actual_demand.index)

        row_pop = build_policy_row("Pop_Proportional", storm_name, storm_year,
                                   alloc_pop, actual_demand, svi_vals, S)

        # ---- Policy 3: Our LP
        alloc_lp = lp_allocate(pred_signal, svi_vals, S)
        row_lp   = build_policy_row("Our_LP", storm_name, storm_year,
                                    alloc_lp, actual_demand, svi_vals, S)

        # Collect rows
        for row in [row_fema, row_pop, row_lp]:
            all_rows.append(row)

        # Print per-storm summary
        print(f"\n  {storm_name} {storm_year} (disaster #{dis_num}):")
        print(f"    {'Policy':<22} {'cov_hi':>8} {'cov_lo':>8} {'eq_gap':>8} {'pct_unmet':>10}")
        print(f"    {'-'*60}")
        for row in [row_fema, row_pop, row_lp]:
            print(f"    {row['policy']:<22} "
                  f"{row['coverage_high_svi']:>8.3f} "
                  f"{row['coverage_low_svi']:>8.3f} "
                  f"{row['equity_gap']:>8.3f} "
                  f"{row['pct_unmet']:>10.3f}")

        # Store scatter data
        scatter_df = pd.DataFrame({
            "fips":          actual_demand.index,
            "svi_overall":   svi_vals.values,
            "demand_proxy":  actual_demand.values,
            "fema_declared": fema_declared_col.values,
            "lp_alloc":      alloc_lp.reindex(actual_demand.index).fillna(0).values,
            "high_svi":      high_mask.values,
        })
        scatter_data[storm_name] = (scatter_df, storm_year)

    # ---------------------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------------------
    results = pd.DataFrame(all_rows)
    out_path = OUT_ALL / "counterfactual_results.parquet"
    results.to_parquet(out_path, index=False)
    print(f"\n[OK] Saved {len(results)} rows -> {out_path}")

    # ---------------------------------------------------------------------------
    # Figure 1: counterfactual.png (3 cols x 2 rows)
    # ---------------------------------------------------------------------------
    storms_done = [s for s, _, _ in TARGET_STORMS
                   if s in results["storm_name"].values]
    n_storms    = len(storms_done)

    if n_storms == 0:
        print("WARNING: No storm data to plot.")
    else:
        fig, axes = plt.subplots(2, n_storms, figsize=(18, 10))
        if n_storms == 1:
            axes = axes.reshape(2, 1)
        fig.patch.set_facecolor(BG)
        fig.suptitle(
            "Counterfactual Analysis: What If Our System Had Been Deployed?",
            fontsize=15, fontweight="bold", color=DARK, y=1.02
        )

        x_3    = np.arange(len(POLICIES_ORDERED))
        width  = 0.30
        labels = [POLICY_LABELS[p] for p in POLICIES_ORDERED]
        colors = [POLICY_COLORS[p] for p in POLICIES_ORDERED]

        def style_ax(ax, title, ylabel, ylim=None):
            ax.set_facecolor(BG)
            ax.set_title(title, fontsize=11, fontweight="bold", color=DARK, pad=8)
            ax.set_ylabel(ylabel, fontsize=9, color=MID)
            ax.tick_params(colors=MID, length=3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color(LIGHT)
            ax.spines["bottom"].set_color(LIGHT)
            ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
            ax.set_axisbelow(True)
            if ylim is not None:
                ax.set_ylim(ylim)

        for col_idx, storm_name in enumerate(storms_done):
            _, _, dis_num = next(
                (s for s in TARGET_STORMS if s[0] == storm_name), (None, None, None)
            )
            storm_year = next(
                (y for s, y, d in TARGET_STORMS if s == storm_name), None
            )
            storm_rows = results[results["storm_name"] == storm_name]

            # -- Row 0: coverage_high_svi and coverage_low_svi grouped bars
            ax0 = axes[0, col_idx]
            col_title = f"Hurricane {storm_name} ({storm_year}): Counterfactual Analysis"
            ax0.set_title(col_title, fontsize=10, fontweight="bold", color=DARK, pad=8)

            offsets = np.array([-width, 0.0, width])
            x_hi = x_3 - width / 2
            x_lo = x_3 + width / 2

            hi_vals = []
            lo_vals = []
            for p in POLICIES_ORDERED:
                prow = storm_rows[storm_rows["policy"] == p]
                hi_vals.append(float(prow["coverage_high_svi"].iloc[0]) if len(prow) else 0.0)
                lo_vals.append(float(prow["coverage_low_svi"].iloc[0]) if len(prow) else 0.0)

            x_base = np.arange(len(POLICIES_ORDERED))
            bars_hi = ax0.bar(x_base - width / 2, hi_vals, width=width,
                              color=colors, edgecolor=WHITE, linewidth=0.8,
                              zorder=3, alpha=0.95, label="High-SVI")
            bars_lo = ax0.bar(x_base + width / 2, lo_vals, width=width,
                              color=colors, edgecolor=WHITE, linewidth=0.8,
                              zorder=3, alpha=0.60, label="Low-SVI")

            for bar, v in zip(bars_hi, hi_vals):
                ax0.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=DARK)
            for bar, v in zip(bars_lo, lo_vals):
                ax0.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=DARK)

            ax0.set_facecolor(BG)
            ax0.set_ylabel("Coverage", fontsize=9, color=MID)
            ax0.set_xticks(x_base)
            ax0.set_xticklabels(labels, rotation=15, ha="right", fontsize=8.5, color=DARK)
            ax0.tick_params(colors=MID, length=3)
            ax0.spines["top"].set_visible(False)
            ax0.spines["right"].set_visible(False)
            ax0.spines["left"].set_color(LIGHT)
            ax0.spines["bottom"].set_color(LIGHT)
            ax0.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
            ax0.set_axisbelow(True)
            ax0.set_ylim(0, 1.15)

            # Legend for row 0 (only first column)
            if col_idx == 0:
                h1 = mpatches.Patch(facecolor="#666", alpha=0.95, label="High-SVI coverage")
                h2 = mpatches.Patch(facecolor="#666", alpha=0.60, label="Low-SVI coverage")
                ax0.legend(handles=[h1, h2], fontsize=8, loc="lower right",
                           framealpha=0.85, edgecolor=LIGHT)

            # -- Row 1: equity_gap
            ax1 = axes[1, col_idx]
            eq_vals = []
            for p in POLICIES_ORDERED:
                prow = storm_rows[storm_rows["policy"] == p]
                eq_vals.append(float(prow["equity_gap"].iloc[0]) if len(prow) else 0.0)

            bars_eq = ax1.bar(x_base, eq_vals, width=0.55, color=colors,
                              edgecolor=WHITE, linewidth=0.8, zorder=3)
            for bar, v in zip(bars_eq, eq_vals):
                ypos = bar.get_height() + (0.003 if v >= 0 else -0.015)
                ax1.text(bar.get_x() + bar.get_width() / 2, ypos,
                         f"{v:+.3f}", ha="center", va="bottom", fontsize=8, color=DARK)
            ax1.axhline(0, color=MID, linewidth=0.8, linestyle="--")
            ax1.set_facecolor(BG)
            ax1.set_ylabel("Equity Gap (High - Low SVI Coverage)", fontsize=9, color=MID)
            ax1.set_xticks(x_base)
            ax1.set_xticklabels(labels, rotation=15, ha="right", fontsize=8.5, color=DARK)
            ax1.tick_params(colors=MID, length=3)
            ax1.spines["top"].set_visible(False)
            ax1.spines["right"].set_visible(False)
            ax1.spines["left"].set_color(LIGHT)
            ax1.spines["bottom"].set_color(LIGHT)
            ax1.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
            ax1.set_axisbelow(True)
            vmin = min(eq_vals) - 0.05
            vmax = max(eq_vals) + 0.05
            ax1.set_ylim(vmin, vmax)

        # Policy legend at figure level
        legend_handles = [
            mpatches.Patch(color=POLICY_COLORS[p], label=POLICY_LABELS[p])
            for p in POLICIES_ORDERED
        ]
        fig.legend(handles=legend_handles, loc="lower center", ncol=3,
                   fontsize=10, framealpha=0.9, edgecolor=LIGHT,
                   bbox_to_anchor=(0.5, -0.03))

        plt.tight_layout()
        fig_path = OUT_FIG / "counterfactual.png"
        fig.savefig(fig_path, dpi=180, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        print(f"[OK] Figure saved -> {fig_path}")

    # ---------------------------------------------------------------------------
    # Figure 2: counterfactual_scatter.png
    # County-level: svi_overall (x) vs demand_proxy (y), bubble size = LP alloc
    # ---------------------------------------------------------------------------
    if scatter_data:
        n_s = len(scatter_data)
        fig2, axes2 = plt.subplots(1, n_s, figsize=(12, 8))
        if n_s == 1:
            axes2 = [axes2]
        fig2.patch.set_facecolor(BG)
        fig2.suptitle(
            "County-Level Demand vs Vulnerability by Storm",
            fontsize=14, fontweight="bold", color=DARK, y=1.01
        )

        for ax2, storm_name in zip(axes2, list(scatter_data.keys())):
            sdf, storm_year = scatter_data[storm_name]
            ax2.set_facecolor(BG)
            ax2.set_title(
                f"County-Level Demand vs Vulnerability\n{storm_name} {storm_year}",
                fontsize=11, fontweight="bold", color=DARK, pad=8
            )
            ax2.set_xlabel("SVI Overall (vulnerability score)", fontsize=10, color=MID)
            ax2.set_ylabel("Demand Proxy (actual PA per capita)", fontsize=10, color=MID)
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            ax2.spines["left"].set_color(LIGHT)
            ax2.spines["bottom"].set_color(LIGHT)
            ax2.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
            ax2.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
            ax2.set_axisbelow(True)
            ax2.tick_params(colors=MID, length=3)

            # Bubble size: LP allocation (scaled for visibility)
            lp_alloc_vals = sdf["lp_alloc"].values
            max_alloc     = lp_alloc_vals.max() if lp_alloc_vals.max() > 0 else 1.0
            sizes         = 30 + (lp_alloc_vals / max_alloc) * 300

            # Three groups:
            # 1) High-SVI + declared (red)
            # 2) Low-SVI + declared (gray)
            # 3) Not declared (light)
            hi_dec  = sdf[(sdf["high_svi"]) & (sdf["fema_declared"] == 1)]
            lo_dec  = sdf[(~sdf["high_svi"]) & (sdf["fema_declared"] == 1)]
            no_dec  = sdf[sdf["fema_declared"] == 0]

            def scatter_group(ax, subdf, color, label, alpha=0.7, zorder=3):
                if len(subdf) == 0:
                    return
                sz = 30 + (subdf["lp_alloc"].values / max_alloc) * 300
                ax.scatter(subdf["svi_overall"], subdf["demand_proxy"],
                           s=sz, c=color, alpha=alpha, edgecolors=DARK,
                           linewidths=0.4, zorder=zorder, label=label)

            scatter_group(ax2, no_dec,  LIGHT,    "Not Declared", alpha=0.5, zorder=2)
            scatter_group(ax2, lo_dec,  MID,      "Low-SVI Declared", alpha=0.7, zorder=3)
            scatter_group(ax2, hi_dec,  COL_FEMA, "High-SVI Declared", alpha=0.85, zorder=4)

            # SVI threshold line
            ax2.axvline(SVI_THRESHOLD, color=COL_FEMA, linewidth=1.2,
                        linestyle="--", alpha=0.7,
                        label=f"SVI threshold = {SVI_THRESHOLD}")

            ax2.legend(fontsize=8, loc="upper left",
                       framealpha=0.85, edgecolor=LIGHT)

            # Bubble size legend
            for sz_label, sz_frac in [("Small LP alloc", 0.1),
                                       ("Large LP alloc", 0.9)]:
                ax2.scatter([], [], s=(30 + sz_frac * 300), c=MID, alpha=0.6,
                            edgecolors=DARK, linewidths=0.4, label=sz_label)

        plt.tight_layout()
        fig2_path = OUT_FIG / "counterfactual_scatter.png"
        fig2.savefig(fig2_path, dpi=180, bbox_inches="tight", facecolor=BG)
        plt.close(fig2)
        print(f"[OK] Scatter figure saved -> {fig2_path}")

    # ---------------------------------------------------------------------------
    # Final summary table
    # ---------------------------------------------------------------------------
    print("\nFull counterfactual results summary:")
    if len(results) > 0:
        summary = results[[
            "storm_name", "storm_year", "policy",
            "coverage_high_svi", "coverage_low_svi",
            "equity_gap", "pct_unmet"
        ]].set_index(["storm_name", "storm_year", "policy"])
        print(summary.round(3).to_string())

    print("\n[DONE] 16_counterfactual.py complete.")


if __name__ == "__main__":
    main()
