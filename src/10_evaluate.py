"""
10_evaluate.py
Produces all final evaluation figures and summary tables:
  - Policy comparison bar charts
  - Equity gap analysis
  - Robustness (CVaR under demand shocks)
  - County allocation map for the largest test storm
Output: outputs/figures/
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

FIGS    = Path("outputs/figures")
ALLOC   = Path("outputs/allocations")
PROC    = Path("data/processed")
FIGS.mkdir(parents=True, exist_ok=True)

POLICY_LABELS = {
    "population_proportional": "Population\nProportional",
    "svi_weighted":            "SVI-Weighted",
    "model_lp_equity":         "Model + LP\nEquity",
}
POLICY_COLORS = {
    "population_proportional": "#6baed6",
    "svi_weighted":            "#fd8d3c",
    "model_lp_equity":         "#31a354",
}


def gini(x: np.ndarray) -> float:
    x = np.sort(np.abs(x[~np.isnan(x)]))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return (2 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum())) - (n + 1) / n


if __name__ == "__main__":
    results = pd.read_parquet(ALLOC / "all_storms_comparison.parquet")
    preds   = pd.read_parquet(PROC / "predictions.parquet")

    # Attach lat/lon from panel if missing (they were dropped in feature engineering)
    if "county_lon" not in preds.columns or "county_lat" not in preds.columns:
        panel = pd.read_parquet(PROC / "panel.parquet")[
            ["storm_id", "fips", "county_lat", "county_lon"]
        ].drop_duplicates()
        preds = preds.merge(panel, on=["storm_id", "fips"], how="left")

    policies = ["population_proportional", "svi_weighted", "model_lp_equity"]

    # - Figure 1: Policy comparison bar charts ----------------
    summary = results.groupby("policy")[[
        "pct_unmet", "coverage_high_svi", "coverage_low_svi", "equity_gap"
    ]].mean()

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Policy Comparison (avg. across test storms)", fontsize=13, y=1.02)

    metrics = [
        ("pct_unmet",         "Avg % Unmet Demand",          "lower is better", False),
        ("coverage_high_svi", "High-SVI Coverage Rate",       "higher is better", True),
        ("equity_gap",        "Equity Gap\n(High−Low SVI coverage)", "higher is better", True),
    ]

    for ax, (metric, title, subtitle, higher_better) in zip(axes, metrics):
        vals   = [summary.loc[p, metric] if p in summary.index else 0 for p in policies]
        colors = [POLICY_COLORS[p] for p in policies]
        bars   = ax.bar([POLICY_LABELS[p] for p in policies], vals, color=colors,
                        edgecolor="white", linewidth=0.5)
        ax.set_title(f"{title}\n({subtitle})", fontsize=10)
        ax.set_ylim(0, max(vals) * 1.25 if max(vals) > 0 else 1)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.02,
                    f"{val:.1%}", ha="center", va="bottom", fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    plt.tight_layout()
    plt.savefig(FIGS / "policy_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] Saved policy_comparison.png")

    # - Figure 2: Robustness / CVaR ---------------------─
    lp_only = results[results["policy"] == "model_lp_equity"].copy()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(lp_only.sort_values("storm_year")["storm_year"],
            lp_only.sort_values("storm_year")["pct_unmet"],
            marker="o", label="Nominal demand", color="#31a354")
    ax.plot(lp_only.sort_values("storm_year")["storm_year"],
            lp_only.sort_values("storm_year")["unmet_shock_plus20"] /
            lp_only.sort_values("storm_year")["total_demand"],
            marker="^", linestyle="--", label="+20% demand shock", color="#de2d26", alpha=0.7)
    ax.set_xlabel("Storm Year")
    ax.set_ylabel("% Unmet Demand")
    ax.set_title("Robustness: Model+LP Policy under Demand Shocks")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGS / "robustness.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] Saved robustness.png")

    # - Figure 3: County scatter — predicted demand vs SVI ----------
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(
        preds["svi_overall"],
        preds["pred_demand"],
        c=preds["pred_demand"],
        cmap="YlOrRd",
        alpha=0.4,
        s=8,
    )
    plt.colorbar(sc, ax=ax, label="Predicted demand (supply units)")
    ax.set_xlabel("SVI Score (0=least vulnerable, 1=most vulnerable)")
    ax.set_ylabel("Predicted Demand (supply units per capita × 1000)")
    ax.set_title("Predicted Demand vs Social Vulnerability")
    plt.tight_layout()
    plt.savefig(FIGS / "demand_vs_svi.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] Saved demand_vs_svi.png")

    # - Figure 4: Allocation map (largest test storm) ------------─
    # Use a simple scatter plot on lat/lon since geopandas shapefile not required
    biggest_storm = (
        preds.groupby("storm_id")["pred_demand"].sum().idxmax()
    )
    storm_subset = preds[preds["storm_id"] == biggest_storm].copy()
    storm_name   = storm_subset["storm_name"].iloc[0]
    storm_year   = storm_subset["storm_year"].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (col, title) in zip(axes, [
        ("pred_demand",  "Predicted Demand"),
        ("svi_overall",  "SVI Score"),
    ]):
        vals = storm_subset[col].fillna(0)
        sc = ax.scatter(
            storm_subset["county_lon"],
            storm_subset["county_lat"],
            c=vals,
            cmap="YlOrRd" if "demand" in col else "Blues",
            s=25,
            alpha=0.8,
            vmin=vals.quantile(0.05),
            vmax=vals.quantile(0.95),
        )
        plt.colorbar(sc, ax=ax, shrink=0.8)
        ax.set_title(f"{title}\n{storm_name} ({storm_year})", fontsize=10)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        # Rough US coast bounding box
        ax.set_xlim(-100, -65)
        ax.set_ylim(20, 50)

    plt.tight_layout()
    plt.savefig(FIGS / f"map_{storm_name}_{storm_year}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved map for {storm_name} ({storm_year})")

    # - Final summary table -------------------------─
    print("\n" + "=" * 65)
    print("FINAL RESULTS SUMMARY")
    print("=" * 65)
    full_summary = results.groupby("policy")[[
        "pct_unmet", "coverage_high_svi", "coverage_low_svi", "equity_gap"
    ]].agg(["mean", "std"]).round(3)
    print(full_summary.to_string())

    lp_row  = results[results["policy"] == "model_lp_equity"]
    pop_row = results[results["policy"] == "population_proportional"]
    if len(lp_row) > 0 and len(pop_row) > 0:
        pct_improvement = (
            (pop_row["pct_unmet"].mean() - lp_row["pct_unmet"].mean()) /
            pop_row["pct_unmet"].mean() * 100
        )
        print(f"\n> Model+LP reduces unmet demand by {pct_improvement:.1f}% "
              f"vs. population-proportional baseline")

        cov_improvement = (
            lp_row["coverage_high_svi"].mean() -
            pop_row["coverage_high_svi"].mean()
        ) * 100
        print(f"> High-SVI coverage improves by {cov_improvement:.1f}pp "
              f"vs. population-proportional baseline")
