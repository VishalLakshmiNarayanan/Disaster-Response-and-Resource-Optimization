"""
13_stochastic_lp.py
Implements two advanced LP formulations using quantile uncertainty bands:
  - LP A: Stochastic LP (here-and-now, 3 equal-weight scenarios: q10/q50/q90)
  - LP B: Two-Stage LP with Recourse (pre-position + reactive allocation)
Compares against deterministic LP and pop-proportional baselines.
Output: outputs/allocations/stochastic_comparison.parquet
        outputs/figures/stochastic_comparison.png
"""
import matplotlib
matplotlib.use("Agg")

import warnings
warnings.filterwarnings("ignore")

import os
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
ALPHA           = 0.60   # fraction pre-positioned in two-stage LP

# Design tokens
BG    = "#fafafa"
DARK  = "#0f172a"
MID   = "#64748b"
LIGHT = "#cbd5e1"
GRID  = "#e2e8f0"
WHITE = "#ffffff"

# Policy colors
COL_POP   = "#94a3b8"
COL_DET   = "#10b981"
COL_STOCH = "#6366f1"
COL_TS    = "#ec4899"

POLICY_LABELS = {
    "pop_proportional": "Pop-Proportional",
    "deterministic_lp": "Deterministic LP",
    "stochastic_lp":    "Stochastic LP",
    "twostage_lp":      "Two-Stage LP",
}
POLICY_COLORS = {
    "pop_proportional": COL_POP,
    "deterministic_lp": COL_DET,
    "stochastic_lp":    COL_STOCH,
    "twostage_lp":      COL_TS,
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def coverage(unmet_s: pd.Series, demand_s: pd.Series) -> float:
    total = demand_s.sum()
    return 1.0 - unmet_s.sum() / total if total > 0 else 1.0


def compute_row(policy_name, storm_name, storm_year, storm_id,
                alloc_s, demand_eval, svi_vals, S):
    high = svi_vals >= SVI_THRESHOLD
    low  = ~high
    unmet = (demand_eval - alloc_s).clip(lower=0)
    return {
        "storm_id":          storm_id,
        "storm_name":        storm_name,
        "storm_year":        storm_year,
        "policy":            policy_name,
        "total_demand":      float(demand_eval.sum()),
        "total_supply":      float(S),
        "total_unmet":       float(unmet.sum()),
        "pct_unmet":         float(unmet.sum() / demand_eval.sum()) if demand_eval.sum() > 0 else 0.0,
        "coverage_overall":  coverage(unmet, demand_eval),
        "coverage_high_svi": coverage(unmet[high], demand_eval[high]),
        "coverage_low_svi":  coverage(unmet[low],  demand_eval[low]),
        "equity_gap":        (coverage(unmet[high], demand_eval[high])
                              - coverage(unmet[low],  demand_eval[low])),
    }


# ---------------------------------------------------------------------------
# Deterministic LP (single-scenario, reuse 09_optimize pattern)
# ---------------------------------------------------------------------------
def lp_deterministic(demands: pd.Series, svi_vals: pd.Series, S: float) -> pd.Series:
    """Equity-constrained LP on a single demand vector. Returns alloc series."""
    counties  = demands.index.tolist()
    high_svi  = [i for i in counties if float(svi_vals.get(i, 0.0)) >= SVI_THRESHOLD]
    weights   = {i: (2.0 if float(svi_vals.get(i, 0.0)) >= SVI_THRESHOLD else 1.0)
                 for i in counties}

    prob  = pulp.LpProblem("det_lp", pulp.LpMinimize)
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


# ---------------------------------------------------------------------------
# LP A: Stochastic LP (here-and-now, 3 scenarios)
# ---------------------------------------------------------------------------
def lp_stochastic(q10_d: pd.Series, q50_d: pd.Series, q90_d: pd.Series,
                  svi_vals: pd.Series, S: float) -> pd.Series:
    """
    Variables: alloc[i] >= 0,  unmet[i,s] >= 0  for s in {0=q10, 1=q50, 2=q90}
    Minimize:  (1/3) * sum_s sum_i weight[i] * unmet[i,s]
    Subject to:
      unmet[i,s] >= demand[i,s] - alloc[i]   for all i, s
      sum_i alloc[i] <= S
      sum_{i in high_svi} alloc[i] >= EQUITY_FRAC * S
      alloc[i] >= 0,  unmet[i,s] >= 0
    """
    counties = q50_d.index.tolist()
    high_svi = [i for i in counties if float(svi_vals.get(i, 0.0)) >= SVI_THRESHOLD]
    weights  = {i: (2.0 if float(svi_vals.get(i, 0.0)) >= SVI_THRESHOLD else 1.0)
                for i in counties}

    scenarios = {0: q10_d, 1: q50_d, 2: q90_d}
    sc_ids    = list(scenarios.keys())

    prob  = pulp.LpProblem("stoch_lp", pulp.LpMinimize)
    alloc = pulp.LpVariable.dicts("a",  counties, lowBound=0)
    unmet = {(i, s): pulp.LpVariable(f"u_{i}_{s}", lowBound=0)
             for i in counties for s in sc_ids}

    # Objective: (1/3) * sum_s sum_i w[i] * unmet[i,s]
    prob += (1.0 / 3.0) * pulp.lpSum(
        weights[i] * unmet[(i, s)]
        for i in counties for s in sc_ids
    )

    # Unmet constraints
    for i in counties:
        for s in sc_ids:
            prob += unmet[(i, s)] >= float(scenarios[s].get(i, 0.0)) - alloc[i]

    # Budget
    prob += pulp.lpSum(alloc[i] for i in counties) <= S

    # Equity
    if high_svi:
        prob += pulp.lpSum(alloc[i] for i in high_svi) >= EQUITY_FRAC * S

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    return pd.Series(
        {i: max(0.0, pulp.value(alloc[i]) or 0.0) for i in counties},
        name="allocated"
    )


# ---------------------------------------------------------------------------
# LP B: Two-Stage LP with Recourse
# ---------------------------------------------------------------------------
def lp_twostage(q10_d: pd.Series, q50_d: pd.Series, q90_d: pd.Series,
                svi_vals: pd.Series, S: float) -> pd.Series:
    """
    alpha = 0.60  (pre-positioned fraction)
    Variables: pre[i] >= 0,  react[i,s] >= 0,  unmet[i,s] >= 0
    Minimize:  (1/3) * sum_s sum_i weight[i] * unmet[i,s]
    Subject to:
      unmet[i,s] >= demand[i,s] - pre[i] - react[i,s]    for all i, s
      sum_i pre[i] <= alpha * S
      sum_i react[i,s] <= (1 - alpha) * S                 for each s
      sum_{i in high_svi} (pre[i] + react[i,s]) >= EQUITY_FRAC * S  for each s
      pre[i] >= 0,  react[i,s] >= 0,  unmet[i,s] >= 0
    Returns total allocation = pre + react under q50 scenario.
    """
    counties  = q50_d.index.tolist()
    high_svi  = [i for i in counties if float(svi_vals.get(i, 0.0)) >= SVI_THRESHOLD]
    weights   = {i: (2.0 if float(svi_vals.get(i, 0.0)) >= SVI_THRESHOLD else 1.0)
                 for i in counties}

    scenarios = {0: q10_d, 1: q50_d, 2: q90_d}
    sc_ids    = list(scenarios.keys())

    S_pre   = ALPHA * S
    S_react = (1.0 - ALPHA) * S

    prob  = pulp.LpProblem("twostage_lp", pulp.LpMinimize)
    pre   = pulp.LpVariable.dicts("pre", counties, lowBound=0)
    react = {(i, s): pulp.LpVariable(f"react_{i}_{s}", lowBound=0)
             for i in counties for s in sc_ids}
    unmet = {(i, s): pulp.LpVariable(f"u_{i}_{s}", lowBound=0)
             for i in counties for s in sc_ids}

    # Objective
    prob += (1.0 / 3.0) * pulp.lpSum(
        weights[i] * unmet[(i, s)]
        for i in counties for s in sc_ids
    )

    # Unmet constraints
    for i in counties:
        for s in sc_ids:
            prob += (unmet[(i, s)] >= float(scenarios[s].get(i, 0.0))
                     - pre[i] - react[(i, s)])

    # Pre-positioning budget
    prob += pulp.lpSum(pre[i] for i in counties) <= S_pre

    # Reactive budget per scenario
    for s in sc_ids:
        prob += pulp.lpSum(react[(i, s)] for i in counties) <= S_react

    # Equity per scenario
    if high_svi:
        for s in sc_ids:
            prob += (pulp.lpSum(pre[i] + react[(i, s)] for i in high_svi)
                     >= EQUITY_FRAC * S)

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    # Return pre + react under q50 scenario (s=1) as realized allocation
    alloc_ts = pd.Series(
        {i: max(0.0, (pulp.value(pre[i]) or 0.0) + (pulp.value(react[(i, 1)]) or 0.0))
         for i in counties},
        name="allocated"
    )
    return alloc_ts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading data...")

    # -- predictions
    try:
        preds = pd.read_parquet(PROC / "predictions.parquet",
                                engine="pyarrow",
                                columns=["fips", "storm_id", "storm_name", "storm_year",
                                         "demand_proxy", "pred_demand",
                                         "pred_q10", "pred_q50", "pred_q90",
                                         "svi_overall", "peak_wind_kt", "min_dist_km",
                                         "log_total_pop", "median_hh_income_10k",
                                         "fema_declared", "abs_error"])
    except Exception:
        # pyarrow histogram mismatch bug -- fall back to polars -> pandas
        try:
            import polars as pl
            preds = pl.read_parquet(str(PROC / "predictions.parquet")).to_pandas()
        except Exception as exc:
            print(f"ERROR loading predictions.parquet: {exc}")
            sys.exit(1)

    # -- SVI
    try:
        svi_df = pd.read_parquet(RAW / "svi" / "svi_2020.parquet",
                                 engine="pyarrow").set_index("fips")
    except Exception:
        try:
            import polars as pl
            svi_df = pl.read_parquet(
                str(RAW / "svi" / "svi_2020.parquet")).to_pandas().set_index("fips")
        except Exception as exc:
            print(f"ERROR loading svi_2020.parquet: {exc}")
            sys.exit(1)

    # -- ACS
    try:
        acs_df = pd.read_parquet(RAW / "acs" / "acs_county.parquet",
                                 engine="pyarrow")
    except Exception:
        try:
            import polars as pl
            acs_df = pl.read_parquet(
                str(RAW / "acs" / "acs_county.parquet")).to_pandas()
        except Exception as exc:
            print(f"ERROR loading acs_county.parquet: {exc}")
            sys.exit(1)

    # Population lookup: latest ACS year per county
    pop_ser = (
        acs_df.sort_values("year")
              .drop_duplicates("fips", keep="last")
              .set_index("fips")["total_pop"]
    )

    storm_ids = preds["storm_id"].unique()
    print(f"Running stochastic & two-stage LP for {len(storm_ids)} storms...")

    all_rows    = []
    vss_records = []   # for Value of Stochastic Solution

    for sid in tqdm(storm_ids, desc="Storms", ncols=80):
        subset = preds[preds["storm_id"] == sid].copy().set_index("fips")

        # Remove duplicate fips (keep first)
        subset = subset[~subset.index.duplicated(keep="first")]

        q10_d = subset["pred_q10"].clip(lower=0)
        q50_d = subset["pred_q50"].clip(lower=0)
        q90_d = subset["pred_q90"].clip(lower=0)
        det_d = subset["pred_demand"].clip(lower=0)

        svi_vals = svi_df["svi_overall"].reindex(q50_d.index).fillna(0.5)
        pop_vals = pop_ser.reindex(q50_d.index).fillna(1.0)

        S = float(q50_d.sum()) * SUPPLY_FRACTION
        if S <= 0 or len(q50_d) < 2:
            continue

        storm_name = subset["storm_name"].iloc[0]
        storm_year = int(subset["storm_year"].iloc[0])

        # ---- Policy 1: Pop-proportional baseline
        safe_pop  = pop_vals.clip(lower=0).fillna(0)
        denom_pop = safe_pop.sum()
        if denom_pop > 0:
            alloc_pop = safe_pop / denom_pop * S
        else:
            alloc_pop = pd.Series(S / len(q50_d), index=q50_d.index)

        # ---- Policy 2: Deterministic LP (uses pred_demand as signal)
        alloc_det = lp_deterministic(det_d, svi_vals, S)

        # ---- Policy 3: Stochastic LP
        alloc_stoch = lp_stochastic(q10_d, q50_d, q90_d, svi_vals, S)

        # ---- Policy 4: Two-Stage LP with Recourse
        alloc_ts = lp_twostage(q10_d, q50_d, q90_d, svi_vals, S)

        # ---- Evaluate all 4 using q50 as realized demand
        demand_eval = q50_d

        for policy_name, alloc_s in [
            ("pop_proportional", alloc_pop),
            ("deterministic_lp", alloc_det),
            ("stochastic_lp",    alloc_stoch),
            ("twostage_lp",      alloc_ts),
        ]:
            all_rows.append(
                compute_row(policy_name, storm_name, storm_year, sid,
                            alloc_s, demand_eval, svi_vals, S)
            )

        # VSS: deterministic objective cost vs stochastic objective cost
        # objective = (1/3) * sum_s sum_i w[i] * unmet[i,s]
        weights_arr = np.where(svi_vals >= SVI_THRESHOLD, 2.0, 1.0)
        q_scenarios = [q10_d.values, q50_d.values, q90_d.values]

        def obj_cost(alloc_arr):
            total = 0.0
            for q in q_scenarios:
                unmet_q = np.maximum(0, q - alloc_arr)
                total += np.dot(weights_arr, unmet_q)
            return total / 3.0

        vss_det   = obj_cost(alloc_det.values)
        vss_stoch = obj_cost(alloc_stoch.values)
        vss_records.append({
            "storm_id":   sid,
            "storm_name": storm_name,
            "vss":        vss_det - vss_stoch,
            "det_obj":    vss_det,
            "stoch_obj":  vss_stoch,
        })

    # ---------------------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------------------
    results = pd.DataFrame(all_rows)
    out_path = OUT_ALL / "stochastic_comparison.parquet"
    results.to_parquet(out_path, index=False)
    print(f"\n[OK] Saved {len(results)} rows -> {out_path}")

    # ---------------------------------------------------------------------------
    # VSS summary
    # ---------------------------------------------------------------------------
    vss_df = pd.DataFrame(vss_records)
    total_vss  = vss_df["vss"].sum()
    mean_vss   = vss_df["vss"].mean()
    print(f"\nValue of Stochastic Solution (VSS):")
    print(f"  VSS = det_unmet_cost - stoch_unmet_cost (objective units)")
    print(f"  Mean VSS across storms : {mean_vss:+.4f}")
    print(f"  Total VSS across storms: {total_vss:+.4f}")
    print(f"  Positive VSS means stochastic LP reduces weighted unmet demand.")

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    summary = (
        results.groupby("policy")[[
            "pct_unmet", "coverage_overall",
            "coverage_high_svi", "coverage_low_svi", "equity_gap"
        ]].mean().round(4)
    )
    print("\nPolicy Comparison (averages across all test storms):")
    print(summary.to_string())

    # ---------------------------------------------------------------------------
    # Figure: 2x2 grid
    # ---------------------------------------------------------------------------
    policies_ordered = ["pop_proportional", "deterministic_lp",
                        "stochastic_lp", "twostage_lp"]
    policy_means = (
        results.groupby("policy")[
            ["pct_unmet", "coverage_high_svi", "coverage_low_svi",
             "coverage_overall", "equity_gap"]
        ].mean()
    )

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Stochastic and Two-Stage LP vs Deterministic Baseline",
                 fontsize=15, fontweight="bold", color=DARK, y=1.01)

    x      = np.arange(len(policies_ordered))
    width  = 0.55
    labels = [POLICY_LABELS[p] for p in policies_ordered]
    colors = [POLICY_COLORS[p] for p in policies_ordered]

    def style_ax(ax, title, ylabel, ylim=None):
        ax.set_facecolor(BG)
        ax.set_title(title, fontsize=12, fontweight="bold", color=DARK, pad=8)
        ax.set_ylabel(ylabel, fontsize=10, color=MID)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9, color=DARK)
        ax.tick_params(colors=MID, length=3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(LIGHT)
        ax.spines["bottom"].set_color(LIGHT)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        if ylim is not None:
            ax.set_ylim(ylim)

    # --- Top-left: avg pct_unmet
    ax = axes[0, 0]
    vals = [policy_means.loc[p, "pct_unmet"] if p in policy_means.index else 0.0
            for p in policies_ordered]
    bars = ax.bar(x, vals, width=width, color=colors, edgecolor=WHITE, linewidth=0.8, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=DARK)
    style_ax(ax, "Avg % Unmet Demand", "pct_unmet", ylim=(0, max(vals) * 1.25 + 0.01))

    # --- Top-right: avg coverage_high_svi
    ax = axes[0, 1]
    vals = [policy_means.loc[p, "coverage_high_svi"] if p in policy_means.index else 0.0
            for p in policies_ordered]
    bars = ax.bar(x, vals, width=width, color=colors, edgecolor=WHITE, linewidth=0.8, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=DARK)
    style_ax(ax, "Avg Coverage: High-SVI Counties", "coverage_high_svi",
             ylim=(max(0, min(vals) - 0.05), 1.05))

    # --- Bottom-left: avg equity_gap
    ax = axes[1, 0]
    vals = [policy_means.loc[p, "equity_gap"] if p in policy_means.index else 0.0
            for p in policies_ordered]
    bars = ax.bar(x, vals, width=width, color=colors, edgecolor=WHITE, linewidth=0.8, zorder=3)
    for bar, v in zip(bars, vals):
        ypos = bar.get_height() + (0.001 if v >= 0 else -0.012)
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{v:+.3f}", ha="center", va="bottom", fontsize=8, color=DARK)
    ax.axhline(0, color=MID, linewidth=0.8, linestyle="--")
    vmin = min(vals) - 0.03
    vmax = max(vals) + 0.03
    style_ax(ax, "Avg Equity Gap (High - Low SVI Coverage)", "equity_gap",
             ylim=(vmin, vmax))

    # --- Bottom-right: scatter coverage_low_svi vs coverage_high_svi
    ax = axes[1, 1]
    ax.set_facecolor(BG)
    ax.set_title("Coverage: High-SVI vs Low-SVI (per storm)",
                 fontsize=12, fontweight="bold", color=DARK, pad=8)
    ax.set_xlabel("coverage_low_svi", fontsize=10, color=MID)
    ax.set_ylabel("coverage_high_svi", fontsize=10, color=MID)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(LIGHT)
    ax.spines["bottom"].set_color(LIGHT)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # Equality line
    lims = [0.0, 1.0]
    ax.plot(lims, lims, color=MID, linewidth=1.0, linestyle="--",
            label="x = y (equal coverage)", zorder=1)

    for p in policies_ordered:
        sub = results[results["policy"] == p]
        if len(sub) == 0:
            continue
        ax.scatter(sub["coverage_low_svi"], sub["coverage_high_svi"],
                   color=POLICY_COLORS[p], alpha=0.35, s=20, zorder=2,
                   label=f"_{POLICY_LABELS[p]}")
        # Mean diamond
        mx = sub["coverage_low_svi"].mean()
        my = sub["coverage_high_svi"].mean()
        ax.scatter([mx], [my], color=POLICY_COLORS[p], marker="D", s=100,
                   edgecolors=DARK, linewidths=0.8, zorder=5,
                   label=POLICY_LABELS[p] + " (mean)")

    ax.tick_params(colors=MID, length=3)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    legend_handles = [
        mpatches.Patch(color=POLICY_COLORS[p], label=POLICY_LABELS[p])
        for p in policies_ordered
    ]
    legend_handles.append(
        plt.Line2D([0], [0], color=MID, linestyle="--", linewidth=1.0,
                   label="x = y (equal coverage)")
    )
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right",
              framealpha=0.85, edgecolor=LIGHT)

    plt.tight_layout()
    fig_path = OUT_FIG / "stochastic_comparison.png"
    fig.savefig(fig_path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"\n[OK] Figure saved -> {fig_path}")

    # Print VSS in objective units
    print(f"\nVSS (Value of Stochastic Solution):")
    print(f"  Mean det objective cost  : {vss_df['det_obj'].mean():.4f}")
    print(f"  Mean stoch objective cost: {vss_df['stoch_obj'].mean():.4f}")
    print(f"  VSS = det_cost - stoch_cost = {mean_vss:+.4f}  (avg per storm)")


if __name__ == "__main__":
    main()
