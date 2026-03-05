"""
Chart 2: The Zero-Inflation Problem
Shows the core modelling challenge using actual data from the pipeline.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import pandas as pd
import numpy as np
from pathlib import Path

OUT   = "../outputs/figures/chart2_the_problem.png"
PROC  = Path("../data/processed")
ALLOC = Path("../outputs/allocations")

# ── Load data ─────────────────────────────────────────────────────────────────
panel = pd.read_parquet(PROC / "panel.parquet")
preds = pd.read_parquet(PROC / "predictions.parquet")

demand   = panel["demand_proxy"]
pred_all = preds["pred_demand"]
act_all  = preds["demand_proxy"]

# ── Colour palette ────────────────────────────────────────────────────────────
RED    = "#c0392b"
ORANGE = "#e67e22"
GREEN  = "#27ae60"
BLUE   = "#2980b9"
PURPLE = "#8e44ad"
GREY   = "#95a5a6"
DARK   = "#1a1a1a"
BG     = "#fdfefe"

fig = plt.figure(figsize=(20, 22), facecolor=BG)
gs  = gridspec.GridSpec(4, 3, figure=fig,
                        hspace=0.52, wspace=0.38,
                        top=0.93, bottom=0.04,
                        left=0.06, right=0.97)

# ── TITLE ─────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.97, "The Core Modelling Problem: Zero-Inflated Demand",
         ha="center", va="top", fontsize=17, fontweight="bold", color=DARK)
fig.text(0.5, 0.945,
         "79.1% of storm-county pairs have zero recorded demand  |  "
         "Model R² ≈ 0  |  Predictions cluster at mean",
         ha="center", va="top", fontsize=11, color="#555555")

# ─────────────────────────────────────────────────────────────────────────────
# PANEL A – Full distribution of demand_proxy (log scale histogram)
# ─────────────────────────────────────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])

zero_frac  = (demand == 0).mean()
nonzero    = demand[demand > 0]

ax_a.hist(nonzero, bins=60, color=BLUE, alpha=0.85,
          edgecolor="white", linewidth=0.4, log=True)
ax_a.axvline(nonzero.median(), color=ORANGE, lw=2, linestyle="--",
             label=f"Median (non-zero): {nonzero.median():.3f}")
ax_a.axvline(nonzero.mean(), color=RED, lw=2, linestyle=":",
             label=f"Mean (non-zero): {nonzero.mean():.3f}")
ax_a.set_xlabel("demand_proxy (supply units / capita × 1000)", fontsize=9)
ax_a.set_ylabel("Count  (log scale)", fontsize=9)
ax_a.set_title(f"A.  Non-Zero Demand Distribution\n(only {(1-zero_frac)*100:.1f}% of rows have demand > 0)",
               fontsize=10, fontweight="bold")
ax_a.legend(fontsize=8)

# Inset annotation
ax_a.text(0.98, 0.92, f"{zero_frac*100:.1f}% of rows\nare ZERO",
          transform=ax_a.transAxes, ha="right", va="top",
          fontsize=10, color=RED, fontweight="bold",
          bbox=dict(boxstyle="round,pad=0.3", facecolor="#fde8e8", alpha=0.9))

# ─────────────────────────────────────────────────────────────────────────────
# PANEL B – Pie chart of zero vs non-zero
# ─────────────────────────────────────────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])

sizes  = [zero_frac * 100, (1 - zero_frac) * 100]
labels = [f"Zero demand\n({zero_frac*100:.1f}%)\n{int(zero_frac*len(demand)):,} rows",
          f"Non-zero demand\n({(1-zero_frac)*100:.1f}%)\n{int((1-zero_frac)*len(demand)):,} rows"]
colors = [GREY, BLUE]
explode = (0.04, 0.04)

wedges, texts, autotexts = ax_b.pie(
    sizes, labels=labels, colors=colors, explode=explode,
    autopct="%1.1f%%", startangle=90,
    textprops={"fontsize": 9},
    wedgeprops={"edgecolor": "white", "linewidth": 2}
)
for at in autotexts:
    at.set_fontsize(10)
    at.set_fontweight("bold")
    at.set_color("white")
ax_b.set_title("B.  Label Distribution\n(31,045 storm-county pairs)",
               fontsize=10, fontweight="bold")

# ─────────────────────────────────────────────────────────────────────────────
# PANEL C – Demand by FEMA declared vs not
# ─────────────────────────────────────────────────────────────────────────────
ax_c = fig.add_subplot(gs[0, 2])

declared     = panel[panel["fema_declared"] == 1]["demand_proxy"]
not_declared = panel[panel["fema_declared"] == 0]["demand_proxy"]

dec_pos = declared[declared > 0]
ndec_pos = not_declared[not_declared > 0]

bins = np.linspace(0, 2, 40)
ax_c.hist(dec_pos,  bins=bins, alpha=0.7, color=GREEN,  label=f"FEMA declared (n={len(declared):,})", density=True, edgecolor="white", lw=0.3)
ax_c.hist(ndec_pos, bins=bins, alpha=0.7, color=ORANGE, label=f"Not declared (n={len(not_declared):,})", density=True, edgecolor="white", lw=0.3)
ax_c.set_xlabel("demand_proxy (clipped at 2)", fontsize=9)
ax_c.set_ylabel("Density", fontsize=9)
ax_c.set_title("C.  Demand by FEMA Declaration Status\n(Key: model must separate these two groups)",
               fontsize=10, fontweight="bold")
ax_c.legend(fontsize=8)
ax_c.text(0.98, 0.92,
          f"Declared: {(declared>0).mean()*100:.0f}% non-zero\nNot decl: {(not_declared>0).mean()*100:.0f}% non-zero",
          transform=ax_c.transAxes, ha="right", va="top", fontsize=9,
          bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f8e8", alpha=0.9))

# ─────────────────────────────────────────────────────────────────────────────
# PANEL D – Predicted vs Actual scatter  (test set)
# ─────────────────────────────────────────────────────────────────────────────
ax_d = fig.add_subplot(gs[1, :2])

clip_val = 3.0
act_c = act_all.clip(upper=clip_val)
pre_c = pred_all.clip(upper=clip_val)

# Colour by actual demand level
color_map = np.where(act_all == 0, GREY,
            np.where(act_all <= 0.1, BLUE,
            np.where(act_all <= 1.0, ORANGE, RED)))

sc = ax_d.scatter(act_c, pre_c, c=color_map, alpha=0.35, s=12, zorder=2)

# Perfect prediction line
max_val = clip_val
ax_d.plot([0, max_val], [0, max_val], "k--", lw=1.5, alpha=0.6,
          label="Perfect prediction (y = x)")

# Horizontal band showing clustering of predictions
pred_mean = pred_all.mean()
pred_std  = pred_all.std()
ax_d.axhspan(pred_mean - pred_std, pred_mean + pred_std,
             alpha=0.12, color=RED,
             label=f"Model prediction band\n(mean±std: {pred_mean:.3f} ± {pred_std:.3f})")
ax_d.axhline(pred_mean, color=RED, lw=1.5, linestyle=":", alpha=0.8)

ax_d.set_xlabel("Actual demand_proxy (clipped at 3.0)", fontsize=10)
ax_d.set_ylabel("Predicted demand (clipped at 3.0)", fontsize=10)
ax_d.set_title(
    "D.  Predicted vs Actual Demand (Test Set, 8,542 rows)\n"
    f"R² = -0.001  |  Model RMSE ≈ Naive baseline  |  "
    f"Predictions span [{pred_all.min():.3f}, {pred_all.max():.3f}] — almost all near mean",
    fontsize=10, fontweight="bold")
ax_d.legend(fontsize=9, loc="upper left")

# Legend for scatter colours
legend_patches = [
    mpatches.Patch(color=GREY,   label="Actual = 0  (zero demand)"),
    mpatches.Patch(color=BLUE,   label="Actual 0 < x ≤ 0.1"),
    mpatches.Patch(color=ORANGE, label="Actual 0.1 < x ≤ 1.0"),
    mpatches.Patch(color=RED,    label="Actual > 1.0  (high demand)"),
]
ax_d.legend(handles=legend_patches, fontsize=8, loc="upper right")
ax_d.set_xlim(-0.05, clip_val + 0.1)
ax_d.set_ylim(-0.01, 0.25)

# ─────────────────────────────────────────────────────────────────────────────
# PANEL E – Prediction variance breakdown
# ─────────────────────────────────────────────────────────────────────────────
ax_e = fig.add_subplot(gs[1, 2])

categories  = ["Actual\nDemand", "Model\nPredictions"]
means       = [act_all.mean(),  pred_all.mean()]
stds        = [act_all.std(),   pred_all.std()]
bar_colors  = [BLUE, RED]

bars = ax_e.bar(categories, means, color=bar_colors, alpha=0.85,
                edgecolor="white", linewidth=1.5, width=0.5)
ax_e.errorbar(categories, means, yerr=stds, fmt="none",
              ecolor="black", elinewidth=2, capsize=10, capthick=2)

for bar, m, s in zip(bars, means, stds):
    ax_e.text(bar.get_x() + bar.get_width()/2,
              m + s + 0.02,
              f"mean={m:.4f}\nstd={s:.4f}", ha="center", va="bottom",
              fontsize=9, fontweight="bold")

ax_e.set_ylabel("demand_proxy value", fontsize=9)
ax_e.set_title("E.  Actual vs Predicted Variance\n(Model collapses range to mean)",
               fontsize=10, fontweight="bold")
ax_e.text(0.5, 0.6,
          f"CV (actual):    {act_all.std()/act_all.mean():.2f}\n"
          f"CV (predicted): {pred_all.std()/pred_all.mean():.2f}",
          transform=ax_e.transAxes, ha="center", va="center",
          fontsize=10, color=DARK,
          bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff3cd", alpha=0.95))

# ─────────────────────────────────────────────────────────────────────────────
# PANEL F – Why the model fails: conceptual diagram
# ─────────────────────────────────────────────────────────────────────────────
ax_f = fig.add_subplot(gs[2, :])
ax_f.set_xlim(0, 20)
ax_f.set_ylim(0, 4.2)
ax_f.axis("off")
ax_f.set_title("F.  Root Cause Analysis: Why XGBoost Predicts Near-Mean for Everyone",
               fontsize=11, fontweight="bold", pad=8)

def fbox(ax, x, y, w, h, title, body, fc, tc="white", fs=8.5):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0,rounding_size=0.2",
                          facecolor=fc, edgecolor="white", lw=1.2, zorder=3)
    ax.add_patch(rect)
    ax.text(x+w/2, y+h*0.68, title, ha="center", va="center",
            color=tc, fontsize=fs, fontweight="bold", zorder=4)
    ax.text(x+w/2, y+h*0.28, body, ha="center", va="center",
            color=tc, fontsize=fs-1.2, zorder=4, linespacing=1.3)

def farr(ax, x1, y1, x2, y2, c="#555555"):
    ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=2, mutation_scale=14),
                zorder=5)

# Cause boxes
fbox(ax_f, 0.2, 1.8, 4.5, 2.0,
     "TRAINING DATA\nZero-Inflation",
     "79.1% of rows have\ndemand_proxy = 0\nXGBoost minimises\nRMSE by predicting\nnear the mean (~0.07)",
     RED)

fbox(ax_f, 5.3, 1.8, 4.5, 2.0,
     "LABEL CONSTRUCTION\nSupply ≠ Demand",
     "PA disbursements reflect\nwhat was SUPPLIED,\nnot what was NEEDED.\nCounties with zero PA\nmay have had real demand",
     ORANGE)

fbox(ax_f, 10.4, 1.8, 4.5, 2.0,
     "TRAINING SCOPE\nAll counties included",
     "Model trains on all\n250 km storm-county pairs,\nincluding counties\nthat were never affected\nor never declared",
     PURPLE)

fbox(ax_f, 15.3, 1.8, 4.5, 2.0,
     "EVALUATION METRIC\nRMSE penalises outliers",
     "High-demand events\n(hurricanes Katrina,\nHarvey, Maria) are rare.\nRMSE penalises predicting\nthese correctly",
     "#2c3e50")

# Consequence box
fbox(ax_f, 4.5, 0.1, 11.0, 1.5,
     "CONSEQUENCE: Model predicts near-constant demand for all counties",
     "All 8,542 test predictions fall in [0.066, 1.578]  |  81% of predictions within 5% of mean\n"
     "Optimization results (47.5% improvement) are driven by the EQUITY CONSTRAINT, not model intelligence",
     "#c0392b")

farr(ax_f, 2.45,  1.8, 7.0,  1.6,  c="#c0392b")
farr(ax_f, 7.55,  1.8, 9.0,  1.6,  c="#c0392b")
farr(ax_f, 12.65, 1.8, 11.0, 1.6, c="#c0392b")
farr(ax_f, 17.55, 1.8, 13.0, 1.6, c="#c0392b")

# ─────────────────────────────────────────────────────────────────────────────
# PANEL G – Impact on allocation: near-uniform vs ideal predictions
# ─────────────────────────────────────────────────────────────────────────────
ax_g = fig.add_subplot(gs[3, :2])

# Use actual Hurricane LAURA data from predictions
laura = preds[preds["storm_name"] == "LAURA"].copy()
if len(laura) == 0:
    laura = preds[preds["storm_id"] == preds["storm_id"].iloc[0]]

laura = laura.sort_values("pred_demand", ascending=False).head(40).reset_index()

x = np.arange(len(laura))
w = 0.35
ax_g.bar(x - w/2, laura["demand_proxy"],  width=w, color=BLUE,  alpha=0.8,
         label="Actual demand (FEMA PA proxy)", edgecolor="white", lw=0.3)
ax_g.bar(x + w/2, laura["pred_demand"],  width=w, color=RED,   alpha=0.8,
         label="Model predicted demand", edgecolor="white", lw=0.3)

ax_g.axhline(laura["pred_demand"].mean(), color=RED, lw=1.5, linestyle=":",
             alpha=0.7, label=f"Pred mean = {laura['pred_demand'].mean():.3f} (near-constant)")
ax_g.set_xlabel("County rank (sorted by predicted demand)", fontsize=9)
ax_g.set_ylabel("demand_proxy", fontsize=9)
ax_g.set_title(
    f"G.  Actual vs Predicted Demand — Top 40 Counties\n"
    f"(Storm: {laura['storm_name'].iloc[0]} {laura['storm_year'].iloc[0]})   "
    f"Model gives near-flat predictions, missing the high-demand counties",
    fontsize=10, fontweight="bold")
ax_g.legend(fontsize=9)
ax_g.set_xticks([])

# ─────────────────────────────────────────────────────────────────────────────
# PANEL H – Summary metrics table
# ─────────────────────────────────────────────────────────────────────────────
ax_h = fig.add_subplot(gs[3, 2])
ax_h.axis("off")

table_data = [
    ["Metric",              "Value",       "Verdict"],
    ["R² (test set)",       "-0.001",      "No predictive power"],
    ["RMSE (model)",        "0.7707",      "Matches naive baseline"],
    ["RMSE (naive mean)",   "0.7704",      "Baseline (0% improvement)"],
    ["Pred range",          "[0.07, 1.58]","Most at 0.07"],
    ["Pred std / mean",     "0.42",        "Low differentiation"],
    ["Actual std / mean",   "17.6",        "High variation in truth"],
    ["Zero-inflation",      "79.1%",       "Core problem"],
    ["Pinball loss q0.5",   "0.031",       "Median model OK"],
]

col_widths = [0.38, 0.28, 0.34]
row_colors = ["#2c3e50"] + (["#f8f9fa", "#ecf0f1"] * 4)
text_colors = ["white"] + ["#1a1a1a"] * 8

for ri, row in enumerate(table_data):
    for ci, (cell, cw) in enumerate(zip(row, col_widths)):
        x_pos = sum(col_widths[:ci]) + 0.01
        rect = FancyBboxPatch((x_pos, 0.88 - ri * 0.099), cw - 0.01, 0.09,
                              boxstyle="round,pad=0,rounding_size=0.01",
                              facecolor=row_colors[ri],
                              edgecolor="white", lw=0.8,
                              transform=ax_h.transAxes, zorder=3)
        ax_h.add_patch(rect)
        ax_h.text(x_pos + (cw-0.01)/2, 0.88 - ri*0.099 + 0.045, cell,
                  ha="center", va="center",
                  fontsize=7.5 if ri > 0 else 8,
                  fontweight="bold" if ri == 0 else "normal",
                  color=text_colors[ri],
                  transform=ax_h.transAxes, zorder=4)

ax_h.set_title("H.  Model Performance Summary", fontsize=10, fontweight="bold")

plt.savefig(OUT, dpi=160, bbox_inches="tight",
            facecolor=BG, edgecolor="none")
plt.close()
print(f"[OK] Saved {OUT}")
