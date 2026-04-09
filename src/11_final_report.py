"""
11_final_report.py  -- Capstone Summary Figure
==============================================
Generates a single publication-quality figure that tells the complete
project story:

  Row 1 (KPI Banner) : 4 headline cards -- before / after comparison
  Row 2 (Model)      : Hurdle model performance (Stage-1 AUC histogram,
                       Stage-2 actual vs predicted) + SHAP bar chart
  Row 3 (Results)    : Policy comparison bars, equity tradeoff scatter,
                       per-storm violin + robustness band

Output: outputs/figures/capstone_summary.png   (22 x 16 in, 180 dpi)
"""

from pathlib import Path
import json
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# -- Paths ---------------------------------------------------------------------
BASE   = Path(".")
PROC   = BASE / "data"  / "processed"
ALLOC  = BASE / "outputs" / "allocations"
MODELS = BASE / "outputs" / "models"
FIGS   = BASE / "outputs" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# -- Design tokens -------------------------------------------------------------
BG    = "#f8fafc"
DARK  = "#0f172a"
MID   = "#475569"
LIGHT = "#cbd5e1"
GRID  = "#e2e8f0"
WHITE = "#ffffff"
CARD  = "#f1f5f9"

PC = {
    "population_proportional": "#94a3b8",
    "svi_weighted":            "#f59e0b",
    "model_lp_equity":         "#10b981",
}
PL = {
    "population_proportional": "Population\nProportional",
    "svi_weighted":            "SVI-Weighted",
    "model_lp_equity":         "Model+LP\n+Equity",
}
POLICIES = list(PC.keys())

pct_fmt = mticker.FuncFormatter(lambda v, _: f"{v:.0%}")

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    WHITE,
    "axes.edgecolor":    LIGHT,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        GRID,
    "grid.linewidth":    0.6,
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "axes.titlepad":     8,
    "axes.labelsize":    10,
    "axes.labelcolor":   MID,
    "xtick.color":       MID,
    "ytick.color":       MID,
    "legend.frameon":    False,
    "legend.fontsize":   9,
    "savefig.facecolor": BG,
    "savefig.dpi":       180,
    "savefig.bbox":      "tight",
})

# -- Load data -----------------------------------------------------------------
print("Loading data ...")
results = pd.read_parquet(ALLOC / "all_storms_comparison.parquet")
preds   = pd.read_parquet(PROC  / "predictions.parquet")

avg = results.groupby("policy")[[
    "pct_unmet", "coverage_overall",
    "coverage_high_svi", "coverage_low_svi", "equity_gap",
]].mean()

lp_df  = results[results["policy"] == "model_lp_equity"].copy()
pop_df = results[results["policy"] == "population_proportional"].copy()
svi_df = results[results["policy"] == "svi_weighted"].copy()

# Model metrics from predictions
TARGET = "demand_proxy"
has_hurdle = "p_nonzero" in preds.columns

if has_hurdle:
    from sklearn.metrics import roc_auc_score, average_precision_score, r2_score
    try:
        from sklearn.metrics import root_mean_squared_error as _rmse
    except ImportError:
        def _rmse(y, yp): return float(np.sqrt(np.mean((np.array(y) - np.array(yp))**2)))

    y_test     = preds[TARGET]
    y_pred     = preds["pred_demand"]
    y_bin      = (y_test > 0).astype(int)
    p_nonzero  = preds["p_nonzero"]

    try:
        clf_auc = roc_auc_score(y_bin, p_nonzero)
        clf_ap  = average_precision_score(y_bin, p_nonzero)
    except Exception:
        clf_auc, clf_ap = 0, 0

    nz_mask   = y_test > 0
    y_nz      = np.log1p(y_test[nz_mask])
    try:
        model_obj = __import__("xgboost").XGBRegressor()
        model_obj.load_model(str(MODELS / "hurdle_reg.json"))
        feats     = json.loads((MODELS / "features.json").read_text())
        Xnz       = preds.loc[nz_mask, feats].fillna(0)
        reg_pred  = model_obj.predict(Xnz)
        reg_r2    = r2_score(y_nz, reg_pred)
        reg_rmse  = _rmse(y_nz, reg_pred)
    except Exception:
        reg_r2, reg_rmse = None, None
        reg_pred = None

    overall_rmse = _rmse(y_test, y_pred)
    naive_rmse   = _rmse(y_test, np.full(len(y_test), y_test.mean()))
    overall_r2   = r2_score(y_test, y_pred)
    rmse_gain    = (naive_rmse - overall_rmse) / naive_rmse
else:
    clf_auc = clf_ap = reg_r2 = reg_rmse = overall_rmse = overall_r2 = rmse_gain = None
    y_nz = reg_pred = None
    y_test = preds[TARGET]
    y_pred = preds["pred_demand"]
    y_bin  = (y_test > 0).astype(int)

# Improvement metrics (LP vs pop-proportional)
pop_unmet   = avg.loc["population_proportional", "pct_unmet"]
lp_unmet    = avg.loc["model_lp_equity",         "pct_unmet"]
pop_hisvi   = avg.loc["population_proportional", "coverage_high_svi"]
lp_hisvi    = avg.loc["model_lp_equity",         "coverage_high_svi"]
pop_gap     = avg.loc["population_proportional", "equity_gap"]
lp_gap      = avg.loc["model_lp_equity",         "equity_gap"]
unmet_reduc = (pop_unmet - lp_unmet) / pop_unmet

print("  Data loaded. Building figure ...")

# ==============================================================================
# Figure layout
# ==============================================================================
fig = plt.figure(figsize=(22, 16), facecolor=BG)

# Main title
fig.text(
    0.5, 0.985,
    "Equity-Based Predictive Disaster Response & Resource Optimization",
    ha="center", va="top",
    fontsize=20, fontweight="bold", color=DARK,
)
fig.text(
    0.5, 0.967,
    "Two-Stage Hurdle Model  .  Equity-Constrained LP Allocation  .  46 Hurricane Events (2018 - 2023)",
    ha="center", va="top",
    fontsize=12, color=MID, style="italic",
)

outer = gridspec.GridSpec(
    3, 1, figure=fig,
    top=0.955, bottom=0.04,
    left=0.04, right=0.97,
    hspace=0.46,
    height_ratios=[1.15, 1.6, 1.6],
)

# ------------------------------------------------------------------------------
# ROW 0: 4 KPI cards
# ------------------------------------------------------------------------------
row0 = gridspec.GridSpecFromSubplotSpec(
    1, 4, subplot_spec=outer[0], wspace=0.06,
)

kpis = [
    {
        "title": "Unmet Demand Reduction",
        "before": f"{pop_unmet:.1%}",
        "after":  f"{lp_unmet:.1%}",
        "delta":  f"v {unmet_reduc:.1%} less unmet",
        "color":  "#10b981",
        "icon":   "📦",
        "label_before": "Pop-Prop",
        "label_after":  "Model+LP",
    },
    {
        "title": "High-SVI County Coverage",
        "before": f"{pop_hisvi:.1%}",
        "after":  f"{lp_hisvi:.1%}",
        "delta":  f"^ +{(lp_hisvi - pop_hisvi):.1%} coverage",
        "color":  "#6366f1",
        "icon":   "🏘",
        "label_before": "Pop-Prop",
        "label_after":  "Model+LP",
    },
    {
        "title": "Equity Gap  (High − Low SVI)",
        "before": f"{pop_gap:+.1%}",
        "after":  f"{lp_gap:+.1%}",
        "delta":  f"^ +{(lp_gap - pop_gap):.1%} equity",
        "color":  "#f59e0b",
        "icon":   "⚖",
        "label_before": "Pop-Prop",
        "label_after":  "Model+LP",
    },
    {
        "title": "Model: Stage-1 ROC-AUC",
        "before": "0.50" if clf_auc is None else "naive",
        "after":  f"{clf_auc:.3f}" if clf_auc else "N/A",
        "delta":  (f"Stage-2 R2 = {reg_r2:.3f}" if reg_r2 else "Hurdle model"),
        "color":  "#0ea5e9",
        "icon":   "🤖",
        "label_before": "Random",
        "label_after":  "Hurdle",
    },
]

for col, kpi in enumerate(kpis):
    ax = fig.add_subplot(row0[col])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    # Card background
    rect = mpatches.FancyBboxPatch(
        (0.03, 0.04), 0.94, 0.92,
        boxstyle="round,pad=0.025",
        linewidth=1.5, edgecolor=kpi["color"],
        facecolor=WHITE, transform=ax.transAxes, zorder=0, alpha=0.9,
    )
    ax.add_patch(rect)

    # Accent bar at top
    accent = mpatches.FancyBboxPatch(
        (0.03, 0.88), 0.94, 0.08,
        boxstyle="round,pad=0.01",
        linewidth=0, facecolor=kpi["color"],
        transform=ax.transAxes, zorder=1, alpha=0.15,
    )
    ax.add_patch(accent)

    ax.text(0.5, 0.91, kpi["title"], ha="center", va="center",
            fontsize=10, fontweight="bold", color=DARK, transform=ax.transAxes)

    # Before / After
    ax.text(0.25, 0.66, kpi["label_before"], ha="center", va="bottom",
            fontsize=8, color=MID, transform=ax.transAxes)
    ax.text(0.25, 0.60, kpi["before"], ha="center", va="top",
            fontsize=20, fontweight="bold", color="#94a3b8", transform=ax.transAxes)

    ax.text(0.5,  0.62, "->", ha="center", va="center",
            fontsize=18, color=LIGHT, transform=ax.transAxes)

    ax.text(0.75, 0.66, kpi["label_after"], ha="center", va="bottom",
            fontsize=8, color=MID, transform=ax.transAxes)
    ax.text(0.75, 0.60, kpi["after"], ha="center", va="top",
            fontsize=20, fontweight="bold", color=kpi["color"], transform=ax.transAxes)

    # Delta badge
    badge = mpatches.FancyBboxPatch(
        (0.10, 0.10), 0.80, 0.22,
        boxstyle="round,pad=0.02",
        linewidth=0, facecolor=kpi["color"],
        transform=ax.transAxes, zorder=1, alpha=0.12,
    )
    ax.add_patch(badge)
    ax.text(0.5, 0.21, kpi["delta"], ha="center", va="center",
            fontsize=10, fontweight="bold", color=kpi["color"],
            transform=ax.transAxes)


# ------------------------------------------------------------------------------
# ROW 1: Model performance (2 panels) + Policy comparison bars (1 panel)
# ------------------------------------------------------------------------------
row1 = gridspec.GridSpecFromSubplotSpec(
    1, 3, subplot_spec=outer[1], wspace=0.38,
)

# -- 1a: Stage-1 score distribution -------------------------------------------
ax1a = fig.add_subplot(row1[0])
if has_hurdle and clf_auc:
    ax1a.hist(preds.loc[y_bin == 0, "p_nonzero"], bins=40,
              alpha=0.60, color="#94a3b8", label="True zero demand", density=True)
    ax1a.hist(preds.loc[y_bin == 1, "p_nonzero"], bins=40,
              alpha=0.70, color="#10b981", label="True non-zero demand", density=True)
    ax1a.set_xlabel("Predicted P(demand > 0)")
    ax1a.set_ylabel("Density")
    ax1a.set_title(
        f"Stage-1 Classifier: Score Separation\n"
        f"ROC-AUC = {clf_auc:.3f}  |  Avg Precision = {clf_ap:.3f}"
    )
    ax1a.legend(fontsize=8.5)
else:
    ax1a.text(0.5, 0.5, "Hurdle model\nnot yet trained\n(run 08_train_model.py first)",
              ha="center", va="center", fontsize=11, color=MID, transform=ax1a.transAxes)
    ax1a.set_title("Stage-1 Classifier")

# -- 1b: Stage-2 actual vs predicted ------------------------------------------
ax1b = fig.add_subplot(row1[1])
if has_hurdle and reg_pred is not None and y_nz is not None:
    ax1b.scatter(y_nz.values, reg_pred,
                 alpha=0.45, s=16, color="#10b981", linewidths=0, zorder=3)
    lo = min(float(y_nz.min()), float(reg_pred.min()))
    hi = max(float(y_nz.max()), float(reg_pred.max()))
    ax1b.plot([lo, hi], [lo, hi], "--", color=LIGHT, linewidth=1.5, zorder=2)
    ax1b.set_xlabel("Actual log1₊demand")
    ax1b.set_ylabel("Predicted log1₊demand")
    ax1b.set_title(
        f"Stage-2 Regressor: Actual vs Predicted\n"
        f"R2 = {reg_r2:.3f}  RMSE = {reg_rmse:.3f}  (non-zero rows only)"
    )
else:
    # Fallback: show overall pred vs actual
    ax1b.scatter(y_test.clip(upper=y_test.quantile(0.99)),
                 y_pred.clip(upper=y_pred.quantile(0.99)),
                 alpha=0.30, s=10, color="#10b981", linewidths=0)
    ax1b.set_xlabel("Actual demand proxy")
    ax1b.set_ylabel("Predicted demand proxy")
    r2_v = overall_r2 if overall_r2 is not None else float("nan")
    ax1b.set_title(f"Predicted vs Actual Demand\nR2 = {r2_v:.3f}")

# -- 1c: Policy comparison bars (3 metrics side by side) ----------------------
ax1c = fig.add_subplot(row1[2])

bar_metrics = [
    ("pct_unmet",         "% Unmet",     True),
    ("coverage_high_svi", "Hi-SVI Cov.", False),
    ("equity_gap",        "Equity Gap",  False),
]
n_met  = len(bar_metrics)
n_pol  = len(POLICIES)
x      = np.arange(n_met)
w      = 0.22
offsets = np.linspace(-(n_pol - 1) * w / 2, (n_pol - 1) * w / 2, n_pol)

for j, pol in enumerate(POLICIES):
    vals = [avg.loc[pol, m] for m, _, _ in bar_metrics]
    bars = ax1c.bar(x + offsets[j], vals, width=w,
                    color=PC[pol], label=PL[pol].replace("\n", " "),
                    zorder=3, linewidth=0)
    for bar, val in zip(bars, vals):
        ypos = bar.get_height() + (0.005 if val >= 0 else -0.025)
        ax1c.text(bar.get_x() + bar.get_width() / 2, ypos,
                  f"{val:.0%}", ha="center", va="bottom",
                  fontsize=7.5, color=DARK, fontweight="bold")

ax1c.set_xticks(x)
ax1c.set_xticklabels([lbl for _, lbl, _ in bar_metrics], fontsize=9)
ax1c.yaxis.set_major_formatter(pct_fmt)
ax1c.axhline(0, color=LIGHT, linewidth=0.8)
ax1c.set_title("Policy Comparison: 3 Key Metrics\n(averages across 46 test storms)")
ax1c.set_ylabel("Rate")
ax1c.legend(loc="upper right", fontsize=8)
ax1c.spines["left"].set_visible(False)


# ------------------------------------------------------------------------------
# ROW 2: Equity scatter + per-storm violin + robustness timeline
# ------------------------------------------------------------------------------
row2 = gridspec.GridSpecFromSubplotSpec(
    1, 3, subplot_spec=outer[2], wspace=0.38,
)

# -- 2a: Equity tradeoff scatter -----------------------------------------------
ax2a = fig.add_subplot(row2[0])

for pol in POLICIES:
    sub = results[results["policy"] == pol]
    ax2a.scatter(
        sub["coverage_low_svi"], sub["coverage_high_svi"],
        color=PC[pol], s=45, alpha=0.65, zorder=3, linewidths=0,
        label=PL[pol].replace("\n", " "),
    )
    mx = sub["coverage_low_svi"].mean()
    my = sub["coverage_high_svi"].mean()
    ax2a.scatter(mx, my, color=PC[pol], s=200, zorder=6,
                 edgecolors=DARK, linewidths=1.8, marker="D")

# Perfect equality line
ax2a.plot([0, 1.05], [0, 1.05], "--", color=LIGHT, linewidth=1.4,
          label="Perfect equality")
ax2a.fill_between([0, 1.05], [0, 1.05], [1.05, 1.05],
                  alpha=0.04, color="#10b981")
ax2a.text(0.05, 0.97, "High-SVI\nfavoured ^",
          fontsize=8, color="#10b981", alpha=0.7, va="top")

ax2a.set_xlabel("Low-SVI County Coverage")
ax2a.set_ylabel("High-SVI County Coverage")
ax2a.set_title("Equity Tradeoff\n(diamonds = policy avg; dots = individual storms)")
ax2a.set_xlim(0, 1.05); ax2a.set_ylim(0, 1.05)
ax2a.set_aspect("equal")
ax2a.xaxis.set_major_formatter(pct_fmt)
ax2a.yaxis.set_major_formatter(pct_fmt)

legend_handles = [
    Line2D([0], [0], marker="o", color="w",
           markerfacecolor=PC[p], markersize=8,
           label=PL[p].replace("\n", " "))
    for p in POLICIES
] + [Line2D([0], [0], color=LIGHT, linewidth=1.5,
            linestyle="--", label="Perfect equality")]
ax2a.legend(handles=legend_handles, loc="lower right", fontsize=8)


# -- 2b: Per-storm violin (equity gap distribution) ----------------------------
ax2b = fig.add_subplot(row2[1])

pos    = np.arange(len(POLICIES))
groups = [results.loc[results["policy"] == p, "equity_gap"].dropna().values
          for p in POLICIES]

parts = ax2b.violinplot(groups, positions=pos, widths=0.55,
                        showmedians=False, showextrema=False)
for body, pol in zip(parts["bodies"], POLICIES):
    body.set_facecolor(PC[pol])
    body.set_alpha(0.30)
    body.set_edgecolor(PC[pol])

bp = ax2b.boxplot(groups, positions=pos, widths=0.18,
                  patch_artist=True, notch=False,
                  whiskerprops=dict(color=MID, linewidth=1.2),
                  capprops=dict(color=MID, linewidth=1.2),
                  flierprops=dict(marker="o", markerfacecolor=MID,
                                  markersize=3, linestyle="none"),
                  medianprops=dict(color=DARK, linewidth=2.2))
for patch, pol in zip(bp["boxes"], POLICIES):
    patch.set_facecolor(PC[pol])
    patch.set_alpha(0.80)

rng = np.random.default_rng(42)
for i, (vals, pol) in enumerate(zip(groups, POLICIES)):
    jitter = rng.uniform(-0.12, 0.12, size=len(vals))
    ax2b.scatter(pos[i] + jitter, vals, color=PC[pol],
                 s=18, alpha=0.50, zorder=4, linewidths=0)

ax2b.axhline(0, color=LIGHT, linewidth=1.0, linestyle="--")
ax2b.set_xticks(pos)
ax2b.set_xticklabels([PL[p] for p in POLICIES], fontsize=8.5)
ax2b.yaxis.set_major_formatter(pct_fmt)
ax2b.set_title("Equity Gap Distribution\nAcross 46 Test Storms")
ax2b.set_ylabel("Equity Gap (High-SVI − Low-SVI Coverage)")
ax2b.text(0.97, 0.03, "↑ positive = high-SVI favoured",
          transform=ax2b.transAxes, ha="right", va="bottom",
          fontsize=8, color=LIGHT, style="italic")
ax2b.spines["left"].set_visible(False)


# -- 2c: Robustness - LP pct_unmet per storm with +/-20% shock band -------------
ax2c = fig.add_subplot(row2[2])

rob = (
    lp_df[["storm_id", "storm_name", "storm_year",
           "pct_unmet", "unmet_shock_plus20",
           "unmet_shock_minus20", "total_demand"]]
    .sort_values(["storm_year", "storm_name"])
    .reset_index(drop=True)
)

rob["pct_shock_hi"] = (rob["unmet_shock_plus20"]  /
                       (rob["total_demand"] * 1.20)).fillna(rob["pct_unmet"])
rob["pct_shock_lo"] = (rob["unmet_shock_minus20"] /
                       (rob["total_demand"] * 0.80)).fillna(rob["pct_unmet"])

x_r   = np.arange(len(rob))
pop_s = pop_df.sort_values(["storm_year", "storm_name"]).reset_index(drop=True)

ax2c.fill_between(x_r, rob["pct_shock_lo"], rob["pct_shock_hi"],
                  alpha=0.18, color="#10b981", label="+/-20% demand shock range")
ax2c.plot(x_r, rob["pct_unmet"], "o-",
          color="#10b981", linewidth=2.0, markersize=5,
          markeredgecolor=DARK, markeredgewidth=0.7,
          label="LP+Equity (baseline)", zorder=4)
ax2c.plot(x_r, pop_s["pct_unmet"], "s--",
          color="#94a3b8", linewidth=1.4, markersize=4,
          alpha=0.60, label="Pop-Proportional", zorder=3)

# Year dividers
yr_breaks = rob[rob["storm_year"] != rob["storm_year"].shift()].index.tolist()
for b in yr_breaks[1:]:
    ax2c.axvline(b - 0.5, color=LIGHT, linewidth=0.9, linestyle=":")

ax2c.set_title("LP+Equity Robustness per Storm\n(+/-20% demand shock sensitivity band)")
ax2c.set_ylabel("% Demand Unmet")
ax2c.set_xticks(x_r[::max(1, len(x_r) // 10)])
ax2c.set_xticklabels(
    rob["storm_name"].iloc[x_r[::max(1, len(x_r) // 10)]].str.title(),
    rotation=35, ha="right", fontsize=7.5,
)
ax2c.yaxis.set_major_formatter(pct_fmt)
ax2c.set_xlim(-0.5, len(rob) - 0.5)
ax2c.legend(loc="upper right", fontsize=8)
ax2c.spines["left"].set_visible(False)


# -- Footer --------------------------------------------------------------------
fig.text(
    0.5, 0.013,
    "Data sources: HURDAT2 (NOAA), FEMA OpenFEMA, NCEI Storm Events, CDC SVI 2020, ACS 5-Year Estimates  .  "
    "Model: Two-Stage XGBoost Hurdle  .  Optimisation: PuLP/CBC Linear Program",
    ha="center", va="bottom", fontsize=8, color=MID, style="italic",
)

# -- Save ----------------------------------------------------------------------
out_path = FIGS / "capstone_summary.png"
plt.savefig(out_path, dpi=180, bbox_inches="tight")
plt.close()

print(f"\n[OK] Capstone summary saved -> {out_path.resolve()}")

# -- Print headline stats -------------------------------------------------------
print(f"\n{'='*62}")
print(f"  CAPSTONE PROJECT -- KEY RESULTS")
print(f"{'='*62}")
print(f"\n  Model Performance (Hurdle, Two-Stage XGBoost)")
if clf_auc:
    print(f"    Stage-1 ROC-AUC      : {clf_auc:.4f}")
    print(f"    Stage-1 Avg Precision: {clf_ap:.4f}")
if reg_r2 is not None:
    print(f"    Stage-2 R2           : {reg_r2:.4f}")
    print(f"    Stage-2 RMSE         : {reg_rmse:.4f}")
if rmse_gain is not None:
    print(f"    Combined RMSE gain   : {rmse_gain:.1%} vs naive baseline")

print(f"\n  Resource Allocation (LP + Equity Constraint, 46 storms)")
for pol in POLICIES:
    r = avg.loc[pol]
    print(f"    {PL[pol].replace(chr(10),' '):<28}  "
          f"unmet={r['pct_unmet']:.1%}  "
          f"hi-SVI={r['coverage_high_svi']:.1%}  "
          f"gap={r['equity_gap']:+.1%}")

print(f"\n  Impact of Model+LP+Equity vs Population-Proportional")
print(f"    Unmet demand reduction : {unmet_reduc:.1%}")
print(f"    High-SVI coverage gain : +{lp_hisvi - pop_hisvi:.1%}")
print(f"    Equity gap improvement : +{lp_gap - pop_gap:.1%}")
print(f"\n  Figure: {out_path.resolve()}")
print(f"{'='*62}")
