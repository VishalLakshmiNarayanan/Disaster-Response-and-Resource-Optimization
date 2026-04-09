"""
14_fairness_analysis.py
-----------------------
Analyze equity across multiple demographic dimensions for the hurricane disaster
resource allocation capstone project.

Dimensions analyzed:
  1. Minority Status  -- EP_MINRTY from svi_2020.parquet
  2. Income           -- median_hh_income_10k from predictions.parquet
  3. Rural / Urban    -- log_total_pop from predictions.parquet as proxy

Outputs:
  outputs/figures/fairness_analysis.png  (18x12 in, dpi=180)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import os
import sys
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = r"C:\Users\shubb\Pytorch\Disaster-Response-and-Resource-Optimization"
PREDICTIONS_PATH   = os.path.join(BASE_DIR, "data", "processed", "predictions.parquet")
SVI_PATH           = os.path.join(BASE_DIR, "data", "raw", "svi", "svi_2020.parquet")
ALLOCATIONS_PATH   = os.path.join(BASE_DIR, "outputs", "allocations",
                                  "all_storms_comparison.parquet")
FIGURE_PATH        = os.path.join(BASE_DIR, "outputs", "figures",
                                  "fairness_analysis.png")

os.makedirs(os.path.join(BASE_DIR, "outputs", "figures"), exist_ok=True)

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
BG    = "#fafafa"
DARK  = "#0f172a"
MID   = "#64748b"
LIGHT = "#cbd5e1"
GRID  = "#e2e8f0"
WHITE = "#ffffff"
GREEN = "#10b981"
AMBER = "#f59e0b"
SLATE = "#94a3b8"

SVI_THRESHOLD = 0.75
HOLDOUT_YEAR  = 2018
TARGET        = "demand_proxy"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_load(path, **kwargs):
    """Load a parquet file; exit with a clear message if missing."""
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)
    return pd.read_parquet(path, **kwargs)


def quartile_labels():
    return ["Q1", "Q2", "Q3", "Q4"]


def assign_quartiles(series, col_name="quartile"):
    """Return a quartile-label Series (Q1..Q4)."""
    labels = quartile_labels()
    return pd.qcut(series, q=4, labels=labels, duplicates="drop")


def bar_chart(ax, x_labels, values, colors, ylabel, title, edge_color=DARK,
              annotation=True):
    """Generic horizontal-bar helper for the fairness grid."""
    bars = ax.bar(x_labels, values, color=colors, edgecolor=edge_color,
                  linewidth=0.6, width=0.6)
    ax.set_title(title, fontsize=9, fontweight="bold", color=DARK, pad=6)
    ax.set_ylabel(ylabel, fontsize=8, color=MID)
    ax.set_facecolor(BG)
    ax.tick_params(axis="x", labelsize=8, colors=DARK)
    ax.tick_params(axis="y", labelsize=7, colors=MID)
    for spine in ax.spines.values():
        spine.set_edgecolor(LIGHT)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    if annotation:
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.01,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=7, color=DARK)
    return bars


def green_gradient(n=4):
    """Return n shades from light to dark green."""
    base = mcolors.to_rgb(GREEN)
    white = (1.0, 1.0, 1.0)
    return [
        tuple(white[i] * (1 - t) + base[i] * t for i in range(3))
        for t in np.linspace(0.3, 1.0, n)
    ]


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("[1/5] Loading predictions.parquet ...")
pred = safe_load(PREDICTIONS_PATH)
print(f"      Loaded {len(pred):,} rows, columns: {list(pred.columns)}")

print("[2/5] Loading svi_2020.parquet ...")
svi_raw = safe_load(SVI_PATH)
# reset_index so fips is a column (it may be the index)
if "fips" not in svi_raw.columns:
    svi_raw = svi_raw.reset_index()

# Keep only what we need
svi_cols = ["fips"]
for col in ["EP_MINRTY", "svi_overall", "EP_POV150", "EP_NOVEH",
            "EP_AGE65", "EP_DISABL", "EP_UNINSUR", "EP_MOBILE",
            "EP_CROWD", "EP_UNEMP", "EP_NOHSDP"]:
    if col in svi_raw.columns:
        svi_cols.append(col)

svi_small = svi_raw[svi_cols].drop_duplicates(subset=["fips"])
print(f"      SVI columns available: {svi_cols}")

# Merge EP_MINRTY into predictions
print("[3/5] Merging SVI data into predictions ...")
pred = pred.merge(svi_small, on="fips", how="left", suffixes=("", "_svi"))

# Resolve potential duplicate columns (e.g., svi_overall_svi)
for col in ["svi_overall", "EP_POV150", "EP_NOVEH", "EP_AGE65",
            "EP_DISABL", "EP_UNINSUR", "EP_MOBILE", "EP_CROWD"]:
    dup = col + "_svi"
    if dup in pred.columns:
        pred[col] = pred[col].fillna(pred[dup])
        pred.drop(columns=[dup], inplace=True)

print(f"      After merge: {len(pred):,} rows")

# Check EP_MINRTY availability
has_minrty = "EP_MINRTY" in pred.columns and pred["EP_MINRTY"].notna().sum() > 0
print(f"      EP_MINRTY available: {has_minrty}")

# ---------------------------------------------------------------------------
# 2. Build quartile bins
# ---------------------------------------------------------------------------
print("[4/5] Creating quartile bins ...")

# -- Minority quartile
if has_minrty:
    try:
        pred["minority_q"] = assign_quartiles(pred["EP_MINRTY"])
    except Exception as e:
        print(f"      [WARN] Could not bin EP_MINRTY: {e}. Using uniform bins.")
        pred["minority_q"] = pd.cut(pred["EP_MINRTY"], bins=4,
                                    labels=quartile_labels())
else:
    print("      [WARN] EP_MINRTY not found; minority dimension will be skipped/synthetic.")
    pred["minority_q"] = "Q1"  # fallback

# -- Income quartile (Q1=lowest)
if "median_hh_income_10k" in pred.columns:
    try:
        pred["income_q"] = assign_quartiles(pred["median_hh_income_10k"])
    except Exception as e:
        print(f"      [WARN] Could not bin income: {e}.")
        pred["income_q"] = pd.cut(pred["median_hh_income_10k"], bins=4,
                                  labels=quartile_labels())
else:
    print("      [WARN] median_hh_income_10k not found; synthesising from svi_overall.")
    pred["median_hh_income_10k"] = 1.0 - pred["svi_overall"].fillna(0.5)
    pred["income_q"] = assign_quartiles(pred["median_hh_income_10k"])

# -- Rural/Urban via log_total_pop
if "log_total_pop" in pred.columns:
    p25 = pred["log_total_pop"].quantile(0.25)
    p75 = pred["log_total_pop"].quantile(0.75)
    def assign_urban(val):
        if pd.isna(val):
            return "Suburban"
        if val <= p25:
            return "Rural"
        if val >= p75:
            return "Urban"
        return "Suburban"
    pred["urban_cat"] = pred["log_total_pop"].apply(assign_urban)
else:
    print("      [WARN] log_total_pop not found; assigning all as Suburban.")
    pred["urban_cat"] = "Suburban"

# fema_declared may be boolean or 0/1
if "fema_declared" in pred.columns:
    pred["fema_declared"] = pred["fema_declared"].astype(float)

print("      Quartile bins created.")

# ---------------------------------------------------------------------------
# 3. Aggregate statistics by dimension
# ---------------------------------------------------------------------------
print("[5/5] Computing aggregated statistics ...")

agg_cols = {}
for col in ["pred_demand", "p_nonzero", "demand_proxy", "fema_declared", "svi_overall"]:
    if col in pred.columns:
        agg_cols[col] = "mean"

agg_cols["fips"] = "count"

# -- Minority dimension
minority_grp = (
    pred.groupby("minority_q", observed=True)
    .agg(agg_cols)
    .rename(columns={"fips": "count"})
    .reindex(quartile_labels())
)
minority_grp.index.name = "quartile"
print("\n  Minority Status by Quartile:")
print(minority_grp.to_string())

# -- Income dimension
income_grp = (
    pred.groupby("income_q", observed=True)
    .agg(agg_cols)
    .rename(columns={"fips": "count"})
    .reindex(quartile_labels())
)
income_grp.index.name = "quartile"
print("\n  Income by Quartile (Q1=lowest):")
print(income_grp.to_string())

# -- Urban dimension
urban_order = ["Rural", "Suburban", "Urban"]
urban_grp = (
    pred.groupby("urban_cat", observed=True)
    .agg(agg_cols)
    .rename(columns={"fips": "count"})
    .reindex(urban_order)
)
urban_grp.index.name = "urban_cat"
print("\n  Rural/Suburban/Urban:")
print(urban_grp.to_string())

# -- Intersectionality: high-SVI + high-minority + low-income
print("\n  Intersectionality Analysis:")
has_svi = "svi_overall" in pred.columns
high_svi_mask  = pred["svi_overall"] >= SVI_THRESHOLD if has_svi else pd.Series(False, index=pred.index)
high_min_mask  = pred["minority_q"] == "Q4"
low_inc_mask   = pred["income_q"]   == "Q1"

intersection   = pred[high_svi_mask & high_min_mask & low_inc_mask]
n_intersect    = len(intersection)
print(f"    Counties that are high-SVI (>={SVI_THRESHOLD}), high-minority (Q4), "
      f"AND low-income (Q1): {n_intersect:,} rows")

if n_intersect > 0:
    avg_pred_demand_int  = intersection["pred_demand"].mean()  if "pred_demand"  in intersection.columns else float("nan")
    avg_actual_demand_int = intersection["demand_proxy"].mean() if "demand_proxy" in intersection.columns else float("nan")
    print(f"    Avg pred_demand  (intersection): {avg_pred_demand_int:.4f}")
    print(f"    Avg demand_proxy (intersection): {avg_actual_demand_int:.4f}")
else:
    avg_pred_demand_int   = 0.0
    avg_actual_demand_int = 0.0

# -- Load allocations for equity-policy context (optional)
if os.path.exists(ALLOCATIONS_PATH):
    try:
        alloc = pd.read_parquet(ALLOCATIONS_PATH)
        equity_alloc = alloc[alloc["policy"] == "model_lp_equity"] if "policy" in alloc.columns else alloc
        print(f"\n  Allocation summary (model_lp_equity policy):")
        if len(equity_alloc) > 0:
            for col in ["coverage_overall", "coverage_high_svi", "coverage_low_svi", "equity_gap"]:
                if col in equity_alloc.columns:
                    print(f"    {col}: mean={equity_alloc[col].mean():.4f}")
        else:
            print("    No rows for model_lp_equity policy.")
    except Exception as e:
        print(f"  [WARN] Could not load allocations: {e}")
else:
    print(f"  [WARN] Allocations file not found: {ALLOCATIONS_PATH}")

# ---------------------------------------------------------------------------
# 4. Helper: safe value extraction from aggregated groups
# ---------------------------------------------------------------------------

def safe_vals(grp, col, index):
    """Return array of values from group; fill with 0 where missing."""
    vals = []
    for idx in index:
        try:
            v = grp.loc[idx, col]
            vals.append(float(v) if pd.notna(v) else 0.0)
        except Exception:
            vals.append(0.0)
    return np.array(vals)

# ---------------------------------------------------------------------------
# 5. Build figure
# ---------------------------------------------------------------------------
print("\nBuilding fairness_analysis.png ...")

fig = plt.figure(figsize=(18, 12), facecolor=BG)
fig.suptitle("Fairness Analysis: Demand Prediction Across Demographic Groups",
             fontsize=14, fontweight="bold", color=DARK, y=0.98)

# Row labels as side-text axes
gs = fig.add_gridspec(3, 3, hspace=0.52, wspace=0.38,
                      left=0.08, right=0.97, top=0.93, bottom=0.06)

axes = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(3)]

q_labels   = quartile_labels()
colors_g   = green_gradient(4)               # light -> dark green (4)
colors_g3  = green_gradient(3)               # for 3 urban categories
amber_g    = [mcolors.to_rgba(AMBER, alpha=a) for a in [0.4, 0.6, 0.8, 1.0]]
slate_g    = [mcolors.to_rgba(SLATE, alpha=a) for a in [0.4, 0.6, 0.8, 1.0]]

# -------- Row 0: Minority Status --------
ax = axes[0][0]
vals = safe_vals(minority_grp, "demand_proxy", q_labels)
bar_chart(ax, q_labels, vals, colors_g,
          ylabel="Avg demand_proxy",
          title="Minority Quartile -> Avg Demand\n(Q1=least minority, Q4=most)")

ax = axes[0][1]
vals = safe_vals(minority_grp, "p_nonzero", q_labels)
bar_chart(ax, q_labels, vals, amber_g,
          ylabel="Avg p_nonzero",
          title="Minority Quartile -> P(any demand)")

ax = axes[0][2]
if "fema_declared" in minority_grp.columns:
    vals = safe_vals(minority_grp, "fema_declared", q_labels)
else:
    vals = np.zeros(4)
bar_chart(ax, q_labels, vals, slate_g,
          ylabel="FEMA declared rate",
          title="Minority Quartile -> FEMA Declaration Rate")

# -------- Row 1: Income --------
ax = axes[1][0]
vals = safe_vals(income_grp, "demand_proxy", q_labels)
bar_chart(ax, q_labels, vals, colors_g,
          ylabel="Avg demand_proxy",
          title="Income Quartile -> Avg Demand\n(Q1=poorest, Q4=richest)")

ax = axes[1][1]
vals = safe_vals(income_grp, "p_nonzero", q_labels)
bar_chart(ax, q_labels, vals, amber_g,
          ylabel="Avg p_nonzero",
          title="Income Quartile -> P(any demand)")

ax = axes[1][2]
if "svi_overall" in income_grp.columns:
    vals = safe_vals(income_grp, "svi_overall", q_labels)
else:
    vals = np.zeros(4)
bar_chart(ax, q_labels, vals, slate_g,
          ylabel="Avg svi_overall",
          title="Income Quartile -> Avg SVI\n(correlation check)")

# -------- Row 2: Rural / Urban --------
urban_labels = ["Rural", "Suburban", "Urban"]

ax = axes[2][0]
vals = safe_vals(urban_grp, "demand_proxy", urban_labels)
bar_chart(ax, urban_labels, vals, colors_g3,
          ylabel="Avg demand_proxy",
          title="Rurality -> Avg Demand\n(Rural=bottom 25% log_pop)")

ax = axes[2][1]
vals = safe_vals(urban_grp, "p_nonzero", urban_labels)
bar_chart(ax, urban_labels, vals, amber_g[:3],
          ylabel="Avg p_nonzero",
          title="Rurality -> P(any demand)")

# -------- Row 2, Col 2: Intersectionality heatmap --------
ax = axes[2][2]

# Build 4x4 pivot: minority_q (rows) x income_q (cols) -> avg demand_proxy
try:
    pivot = (
        pred.groupby(["minority_q", "income_q"], observed=True)["demand_proxy"]
        .mean()
        .unstack("income_q")
        .reindex(index=q_labels, columns=q_labels)
        .fillna(0.0)
    )
    heatmap_data = pivot.values.astype(float)
except Exception as e:
    print(f"  [WARN] Heatmap pivot failed: {e}")
    heatmap_data = np.zeros((4, 4))

vmin = heatmap_data.min()
vmax = heatmap_data.max() if heatmap_data.max() > vmin else vmin + 1e-6

cmap = plt.cm.YlOrRd
im = ax.imshow(heatmap_data, aspect="auto", cmap=cmap,
               vmin=vmin, vmax=vmax, origin="lower")

ax.set_xticks(range(4))
ax.set_xticklabels(["Inc Q1", "Inc Q2", "Inc Q3", "Inc Q4"], fontsize=7)
ax.set_yticks(range(4))
ax.set_yticklabels(["Min Q1", "Min Q2", "Min Q3", "Min Q4"], fontsize=7)
ax.set_title("Intersectionality Heatmap\nMinority x Income -> Avg Demand",
             fontsize=9, fontweight="bold", color=DARK, pad=6)

cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=7)
cbar.set_label("Avg demand_proxy", fontsize=7, color=MID)

# Annotate cells
for i in range(4):
    for j in range(4):
        val = heatmap_data[i, j]
        text_color = "white" if (val - vmin) / (vmax - vmin + 1e-9) > 0.6 else DARK
        ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                fontsize=6.5, color=text_color, fontweight="bold")

# -------- Row labels on left side --------
row_labels = ["Minority\nStatus", "Income\nLevel", "Rural /\nUrban"]
for r, label in enumerate(row_labels):
    # Use figure-level text positioned at the left of each row
    ax_left    = axes[r][0]
    bbox       = ax_left.get_position()
    ycenter    = (bbox.y0 + bbox.y1) / 2
    fig.text(0.01, ycenter, label,
             ha="left", va="center",
             fontsize=9, fontweight="bold", color=DARK,
             rotation=90)

# -------- Save --------
plt.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"Figure saved -> {FIGURE_PATH}")

# ---------------------------------------------------------------------------
# 6. Print final summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("FAIRNESS ANALYSIS SUMMARY")
print("=" * 70)

print("\n-- Dimension 1: Minority Status --")
for q in q_labels:
    if q in minority_grp.index:
        row = minority_grp.loc[q]
        dp  = row.get("demand_proxy", float("nan"))
        pnz = row.get("p_nonzero",   float("nan"))
        cnt = int(row.get("count", 0))
        print(f"  {q}: n={cnt:6,} | avg_demand={dp:.4f} | avg_p_nonzero={pnz:.4f}")

print("\n-- Dimension 2: Income --")
for q in q_labels:
    if q in income_grp.index:
        row = income_grp.loc[q]
        dp  = row.get("demand_proxy", float("nan"))
        pnz = row.get("p_nonzero",   float("nan"))
        svi = row.get("svi_overall", float("nan"))
        print(f"  {q}: avg_demand={dp:.4f} | avg_p_nonzero={pnz:.4f} | avg_svi={svi:.4f}")

print("\n-- Dimension 3: Rural/Suburban/Urban --")
for cat in urban_labels:
    if cat in urban_grp.index:
        row = urban_grp.loc[cat]
        dp  = row.get("demand_proxy", float("nan"))
        pnz = row.get("p_nonzero",   float("nan"))
        cnt = int(row.get("count", 0))
        print(f"  {cat}: n={cnt:6,} | avg_demand={dp:.4f} | avg_p_nonzero={pnz:.4f}")

print("\n-- Intersectionality (high-SVI + high-minority + low-income) --")
print(f"  Number of county-storm obs: {n_intersect:,}")
print(f"  Avg pred_demand  : {avg_pred_demand_int:.4f}")
print(f"  Avg demand_proxy : {avg_actual_demand_int:.4f}")

print("\n[DONE] 14_fairness_analysis.py completed.")
