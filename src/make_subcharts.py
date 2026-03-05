"""
make_subcharts.py
Saves every subchart from Chart 2 and Chart 3 as individual,
visually-polished PNG files into:
  outputs/figures/chart2/   (panels A-H)
  outputs/figures/chart3/   (11 subcharts)

Run from the hurricane-allocation project root:
  python src/make_subcharts.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROC  = Path("data/processed")
FIGS  = Path("outputs/figures")
DIR2  = FIGS / "chart2"
DIR3  = FIGS / "chart3"
DIR2.mkdir(parents=True, exist_ok=True)
DIR3.mkdir(parents=True, exist_ok=True)

# ── Palette ────────────────────────────────────────────────────────────────────
RED    = "#c0392b"; ORANGE = "#e67e22"; GREEN  = "#27ae60"
BLUE   = "#2980b9"; PURPLE = "#8e44ad"; GREY   = "#95a5a6"
DARK   = "#1a1a1a"; BG     = "#fdfefe"
C = {
    "source":   "#1a5276", "script": "#1e8449", "data":   "#d4e6f1",
    "model":    "#7d3c98", "optim":  "#b7770d", "output": "#922b21",
    "text_lt":  "#ffffff", "text_dk":"#1a1a1a",
    "approach1":"#1a5276", "approach2":"#1e8449","approach3":"#7d3c98",
    "pro":      "#1e8449", "con":    "#c0392b",  "dark":   "#1a1a1a",
    "light_1":  "#d4e6f1", "light_2":"#d5f5e3",  "light_3":"#e8daef",
    "header":   "#2c3e50",
}

# ── Shared helpers ─────────────────────────────────────────────────────────────
def box(ax, x, y, w, h, label, sublabel="", color="#1a5276",
        text_color="white", fontsize=9, radius=0.25):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=1.2, edgecolor="white", facecolor=color, zorder=3))
    if sublabel:
        ax.text(x+w/2, y+h*0.62, label,    ha="center", va="center",
                color=text_color, fontsize=fontsize,   fontweight="bold", zorder=4)
        ax.text(x+w/2, y+h*0.28, sublabel, ha="center", va="center",
                color=text_color, fontsize=fontsize-1.5, style="italic", zorder=4)
    else:
        ax.text(x+w/2, y+h/2, label, ha="center", va="center",
                color=text_color, fontsize=fontsize, fontweight="bold", zorder=4)

def arrow(ax, x1, y1, x2, y2, color="#555555", lw=1.8):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=14), zorder=2)

def fbox(ax, x, y, w, h, title, body, fc, tc="white", fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.2",
        facecolor=fc, edgecolor="white", lw=1.2, zorder=3))
    ax.text(x+w/2, y+h*0.68, title, ha="center", va="center",
            color=tc, fontsize=fs,       fontweight="bold", zorder=4)
    ax.text(x+w/2, y+h*0.28, body,  ha="center", va="center",
            color=tc, fontsize=fs-1.2,   zorder=4, linespacing=1.3)

def farr(ax, x1, y1, x2, y2, c="#555555"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=c, lw=2, mutation_scale=14), zorder=5)

def save(fig, path, tight=True):
    fig.patch.set_facecolor(BG)
    kw = dict(dpi=180, facecolor=BG)
    if tight:
        kw["bbox_inches"] = "tight"
    fig.savefig(path, **kw)
    plt.close(fig)
    print(f"  [OK] {path.name}")

def style_ax(ax):
    """Apply clean background + light grid to a regular axes."""
    ax.set_facecolor("#ffffff")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#dddddd")
    ax.spines["bottom"].set_color("#dddddd")
    ax.tick_params(colors="#555555")
    ax.yaxis.label.set_color("#555555")
    ax.xaxis.label.set_color("#555555")
    ax.title.set_color(DARK)
    ax.grid(color="#eeeeee", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data ...")
panel = pd.read_parquet(PROC / "panel.parquet")
preds = pd.read_parquet(PROC / "predictions.parquet")
demand   = panel["demand_proxy"]
pred_all = preds["pred_demand"]
act_all  = preds["demand_proxy"]
print(f"  panel {panel.shape} | preds {preds.shape} | "
      f"zeros {(demand==0).mean()*100:.1f}%")


# ════════════════════════════════════════════════════════════════════════════════
#  CHART 2 — The Core Modelling Problem
# ════════════════════════════════════════════════════════════════════════════════
print("\n-- Chart 2 subcharts --")

# ── Panel A — Non-Zero Demand Histogram ────────────────────────────────────────
zero_frac = (demand == 0).mean()
nonzero   = demand[demand > 0]

fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
n, bins, patches = ax.hist(nonzero, bins=65, color=BLUE, alpha=0.88,
                            edgecolor="white", linewidth=0.4, log=True, zorder=3)
ax.axvline(nonzero.median(), color=ORANGE, lw=2.2, linestyle="--",
           label=f"Median: {nonzero.median():.3f}")
ax.axvline(nonzero.mean(),   color=RED,    lw=2.2, linestyle=":",
           label=f"Mean:   {nonzero.mean():.3f}")
ax.set_xlabel("demand_proxy  (supply units / capita x 1000)", fontsize=11)
ax.set_ylabel("Count  (log scale)", fontsize=11)
ax.set_title(
    f"Panel A — Non-Zero Demand Distribution\n"
    f"{(1-zero_frac)*100:.1f}% of rows have demand > 0  |  heavy right tail",
    fontsize=13, fontweight="bold", pad=12
)
ax.legend(fontsize=10, framealpha=0.9)
ax.text(0.98, 0.93,
        f"{zero_frac*100:.1f}% of rows\nare ZERO",
        transform=ax.transAxes, ha="right", va="top", fontsize=12,
        color=RED, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#fde8e8", alpha=0.95, edgecolor=RED))
style_ax(ax)
save(fig, DIR2 / "A_nonzero_distribution.png")

# ── Panel B — Label Distribution Pie ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8), facecolor=BG)
sizes  = [zero_frac * 100, (1 - zero_frac) * 100]
labels = [
    f"Zero demand\n({zero_frac*100:.1f}%)\n{int(zero_frac*len(demand)):,} rows",
    f"Non-zero demand\n({(1-zero_frac)*100:.1f}%)\n{int((1-zero_frac)*len(demand)):,} rows",
]
wedge_props = dict(edgecolor="white", linewidth=2.5)
wedges, texts, autotexts = ax.pie(
    sizes, labels=labels,
    colors=[GREY, BLUE],
    explode=(0.04, 0.08),
    autopct="%1.1f%%",
    startangle=90,
    textprops={"fontsize": 11},
    wedgeprops=wedge_props,
    pctdistance=0.72,
)
for at in autotexts:
    at.set_fontsize(12); at.set_fontweight("bold"); at.set_color("white")
ax.set_title(
    "Panel B — Label Distribution\n31,045 storm-county pairs in the full panel",
    fontsize=13, fontweight="bold", pad=14
)
save(fig, DIR2 / "B_label_distribution.png")

# ── Panel C — Demand by FEMA Declaration Status ────────────────────────────────
declared     = panel[panel["fema_declared"] == 1]["demand_proxy"]
not_declared = panel[panel["fema_declared"] == 0]["demand_proxy"]

fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
bins_c = np.linspace(0, 2, 45)
ax.hist(declared[declared > 0],     bins=bins_c, alpha=0.75, color=GREEN,
        density=True, edgecolor="white", lw=0.3, zorder=3,
        label=f"FEMA-Declared  (n={len(declared):,})")
ax.hist(not_declared[not_declared > 0], bins=bins_c, alpha=0.75, color=ORANGE,
        density=True, edgecolor="white", lw=0.3, zorder=3,
        label=f"Not Declared  (n={len(not_declared):,})")
ax.set_xlabel("demand_proxy  (clipped at 2.0)", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title(
    "Panel C — Demand by FEMA Declaration Status\n"
    "Declared counties have ~41% non-zero demand; un-declared is near 0%",
    fontsize=13, fontweight="bold", pad=12
)
ax.legend(fontsize=10, framealpha=0.9)
ax.text(0.97, 0.93,
        f"Declared: {(declared>0).mean()*100:.0f}% non-zero\n"
        f"Not decl: {(not_declared>0).mean()*100:.0f}% non-zero",
        transform=ax.transAxes, ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#e8f8e8", alpha=0.95, edgecolor=GREEN))
style_ax(ax)
save(fig, DIR2 / "C_demand_by_fema_status.png")

# ── Panel D — Predicted vs Actual (scatter) ────────────────────────────────────
clip_val = 3.0
act_c = act_all.clip(upper=clip_val)
pre_c = pred_all.clip(upper=clip_val)
color_map = np.where(act_all == 0, GREY,
            np.where(act_all <= 0.1, BLUE,
            np.where(act_all <= 1.0, ORANGE, RED)))
pred_mean = pred_all.mean(); pred_std = pred_all.std()

fig, ax = plt.subplots(figsize=(11, 7), facecolor=BG)
ax.scatter(act_c, pre_c, c=color_map, alpha=0.32, s=14, zorder=3)
ax.plot([0, clip_val], [0, clip_val], "k--", lw=1.5, alpha=0.6, label="Perfect prediction")
ax.axhspan(pred_mean - pred_std, pred_mean + pred_std,
           alpha=0.10, color=RED,
           label=f"Pred band  {pred_mean:.3f} \u00b1 {pred_std:.3f}")
ax.axhline(pred_mean, color=RED, lw=1.5, linestyle=":", alpha=0.8)
ax.set_xlabel("Actual demand_proxy  (clipped at 3.0)", fontsize=11)
ax.set_ylabel("Predicted demand  (clipped at 3.0)", fontsize=11)
ax.set_title(
    f"Panel D — Predicted vs Actual Demand  (n={len(act_all):,} test rows)\n"
    f"R\u00b2 = -0.001  |  All predictions cluster near mean = {pred_mean:.3f}",
    fontsize=13, fontweight="bold", pad=12
)
legend_handles = [
    mpatches.Patch(color=GREY,   label="Actual = 0  (zero demand)"),
    mpatches.Patch(color=BLUE,   label="Actual 0 < x \u2264 0.1"),
    mpatches.Patch(color=ORANGE, label="Actual 0.1 < x \u2264 1.0"),
    mpatches.Patch(color=RED,    label="Actual > 1.0  (high demand)"),
    mpatches.Patch(color="none", label=f"Pred band \u00b1 1 std (red shading)"),
]
ax.legend(handles=legend_handles, fontsize=9, loc="upper right", framealpha=0.9)
ax.set_xlim(-0.05, clip_val + 0.1)
ax.set_ylim(-0.01, 0.28)
style_ax(ax)
save(fig, DIR2 / "D_predicted_vs_actual.png")

# ── Panel E — Variance Collapse ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG)
means = [act_all.mean(), pred_all.mean()]
stds  = [act_all.std(),  pred_all.std()]
bars  = ax.bar(["Actual\nDemand", "Model\nPredictions"], means,
               color=[BLUE, RED], alpha=0.88, edgecolor="white", lw=1.8, width=0.45, zorder=3)
ax.errorbar(["Actual\nDemand", "Model\nPredictions"], means, yerr=stds,
            fmt="none", ecolor=DARK, elinewidth=2.2, capsize=14, capthick=2.2, zorder=4)
for bar, m, s in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.015,
            f"mean = {m:.4f}\nstd  = {s:.4f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold", color=DARK)
ax.set_ylabel("demand_proxy value", fontsize=11)
ax.set_title(
    "Panel E — Variance Collapse\n"
    "Model compresses all variation into a narrow band near the mean",
    fontsize=13, fontweight="bold", pad=12
)
cv_act  = act_all.std()  / act_all.mean()
cv_pred = pred_all.std() / pred_all.mean()
ax.text(0.50, 0.58,
        f"CV (actual):    {cv_act:.2f}\nCV (predicted): {cv_pred:.2f}",
        transform=ax.transAxes, ha="center", va="center", fontsize=12,
        color=DARK, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#fff3cd",
                  alpha=0.97, edgecolor="#f59e0b", lw=1.5))
style_ax(ax)
save(fig, DIR2 / "E_variance_collapse.png")

# ── Panel F — Root Cause Diagram ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(20, 6), facecolor=BG)
ax.set_xlim(0, 20); ax.set_ylim(0, 4.8); ax.axis("off")
ax.set_title(
    "Panel F — Root Cause Analysis: Why XGBoost Predicts Near-Mean for Everyone",
    fontsize=14, fontweight="bold", color=DARK, pad=14
)

causes = [
    ("TRAINING DATA\nZero-Inflation",
     "79.1% of rows have demand = 0\nXGBoost minimises RMSE by predicting\n~mean (0.07)\u2014optimal but useless",
     RED),
    ("LABEL CONSTRUCTION\nSupply vs Demand",
     "PA disbursements = what was SUPPLIED,\nnot what was NEEDED\nZero PA may still mean real unmet need",
     ORANGE),
    ("TRAINING SCOPE\nAll Counties Included",
     "Trains on ALL 250 km pairs including\ncounties never affected or never declared\n=> noise dominates signal",
     PURPLE),
    ("EVALUATION METRIC\nRMSE Penalises Outliers",
     "Katrina/Harvey/Maria are rare events\nRMSE-optimal strategy = predict mean\n=> safe but useless",
     "#2c3e50"),
]
xs = [0.3, 5.1, 9.9, 14.7]
for (title, body, fc), x in zip(causes, xs):
    fbox(ax, x, 1.9, 4.5, 2.2, title, body, fc, fs=8.8)

fbox(ax, 3.5, 0.08, 13.0, 1.65,
     "CONSEQUENCE: Model predicts near-constant demand for all counties",
     "All 8,542 predictions in [0.066, 1.578]  |  81% within 5% of mean\n"
     "47.5% optimisation improvement = equity CONSTRAINT effect, NOT model intelligence",
     "#c0392b", fs=9.5)

arrow_coords = [
    (2.55, 1.9, 7.0, 1.73),
    (7.35, 1.9, 9.0, 1.73),
    (12.15, 1.9, 11.0, 1.73),
    (16.95, 1.9, 13.0, 1.73),
]
for x1, y1, x2, y2 in arrow_coords:
    farr(ax, x1, y1, x2, y2, c="#c0392b")

save(fig, DIR2 / "F_root_cause_analysis.png")

# ── Panel G — Per-Storm Prediction Failure (LAURA) ────────────────────────────
laura = preds[preds["storm_name"].str.upper() == "LAURA"].copy()
if len(laura) == 0:
    laura = preds[preds["storm_id"] == preds["storm_id"].iloc[0]].copy()
storm_label = f"{laura['storm_name'].iloc[0].title()} {laura['storm_year'].iloc[0]}"
laura_top = laura.sort_values("pred_demand", ascending=False).head(40).reset_index(drop=True)
x = np.arange(len(laura_top)); w = 0.38

fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG)
ax.bar(x - w/2, laura_top["demand_proxy"], width=w, color=BLUE, alpha=0.82,
       label="Actual demand (FEMA PA proxy)", edgecolor="white", lw=0.3, zorder=3)
ax.bar(x + w/2, laura_top["pred_demand"],  width=w, color=RED,  alpha=0.82,
       label="Model predicted demand",       edgecolor="white", lw=0.3, zorder=3)
ax.axhline(laura_top["pred_demand"].mean(), color=RED, lw=2.0, linestyle=":",
           alpha=0.75, label=f"Pred mean = {laura_top['pred_demand'].mean():.3f}  (near-flat)")
ax.set_xlabel("County rank (sorted by predicted demand)", fontsize=11)
ax.set_ylabel("demand_proxy", fontsize=11)
ax.set_title(
    f"Panel G — Hurricane {storm_label}: Actual vs Predicted Demand (Top 40 Counties)\n"
    "Model gives near-flat predictions \u2014 misses every high-demand county",
    fontsize=13, fontweight="bold", pad=12
)
ax.legend(fontsize=10, framealpha=0.9)
ax.set_xticks([])
style_ax(ax)
save(fig, DIR2 / "G_per_storm_prediction_failure.png")

# ── Panel H — Model Performance Summary Table ─────────────────────────────────
rows_h = [
    ("Metric",           "Value",         "Verdict"),
    ("R\u00b2 (test set)", "-0.001",       "No predictive power"),
    ("RMSE (model)",     "0.7707",         "Matches naive baseline"),
    ("RMSE (naive mean)","0.7704",         "0% improvement"),
    ("Pred range",       "[0.07, 1.58]",   "Almost all near 0.07"),
    ("Pred CV",          "0.42",           "Low differentiation"),
    ("Actual CV",        "17.6",           "High variation in truth"),
    ("Zero-inflation",   "79.1%",          "Core problem"),
    ("Pinball q0.5",     "0.031",          "Median model OK"),
]
col_w  = [0.38, 0.28, 0.34]
row_bg = ["#2c3e50"] + ["#f8f9fa", "#ecf0f1"] * 4
row_fg = ["white"]   + ["#1a1a1a"] * 8

fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
ax.axis("off")
for ri, row in enumerate(rows_h):
    for ci, (cell, cw) in enumerate(zip(row, col_w)):
        xp = sum(col_w[:ci]) + 0.01
        yp = 0.88 - ri * 0.099
        ax.add_patch(FancyBboxPatch(
            (xp, yp), cw - 0.015, 0.093,
            boxstyle="round,pad=0,rounding_size=0.01",
            facecolor=row_bg[ri], edgecolor="white", lw=1.0,
            transform=ax.transAxes, zorder=3
        ))
        ax.text(xp + (cw - 0.015) / 2, yp + 0.047, cell,
                ha="center", va="center",
                fontsize=9 if ri > 0 else 10,
                fontweight="bold" if ri == 0 else "normal",
                color=row_fg[ri],
                transform=ax.transAxes, zorder=4)
ax.set_title("Panel H — Model Performance Summary",
             fontsize=13, fontweight="bold", pad=18, color=DARK)
save(fig, DIR2 / "H_model_performance_summary.png")


# ════════════════════════════════════════════════════════════════════════════════
#  CHART 3 — Solution Approaches
# ════════════════════════════════════════════════════════════════════════════════
print("\n-- Chart 3 subcharts --")

# ── Subchart 3-00 — Approach Overview Cards ───────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 4), facecolor=BG)
fig.subplots_adjust(wspace=0.06)
cards = [
    ("APPROACH 1  [RECOMMENDED]", "Two-Part Hurdle Model",                   C["approach1"]),
    ("APPROACH 2",                 "Filter to FEMA-Declared + Log-Transform", C["approach2"]),
    ("APPROACH 3",                 "Spatial Block CV + Ensemble",             C["approach3"]),
]
for ax, (title, sub, fc) in zip(axes, cards):
    ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.1, 0.1), 9.8, 2.8,
        boxstyle="round,pad=0,rounding_size=0.35",
        facecolor=fc, edgecolor="white", lw=2.5, zorder=3))
    ax.text(5, 2.0, title, ha="center", va="center",
            color="white", fontsize=12, fontweight="bold", zorder=4)
    ax.text(5, 1.1, sub, ha="center", va="center",
            color="white", fontsize=10.5, style="italic", zorder=4)
fig.suptitle("Three Approaches to Fix the Zero-Inflated Model",
             fontsize=14, fontweight="bold", y=1.04, color=DARK)
save(fig, DIR3 / "00_approach_overview.png")

# ── Subchart 3-01 — Architecture: Two-Part Hurdle ────────────────────────────
fig, ax = plt.subplots(figsize=(12, 8), facecolor=BG)
ax.set_xlim(0, 10); ax.set_ylim(0, 6.5); ax.axis("off")
ax.set_title("Architecture: Two-Part (Hurdle) Model",
             fontsize=14, fontweight="bold", color=C["approach1"], pad=14)

box(ax, 1, 5.6, 8, 0.75, "Input Features  (18 features per storm-county pair)",
    color=C["header"], fontsize=10)
box(ax, 0.3, 4.3, 4.2, 1.0, "Stage 1: Classifier",
    "XGBoost Classifier\ntarget: demand > 0  (yes / no)", color=C["approach1"], fontsize=9)
box(ax, 5.5, 4.3, 4.2, 1.0, "Stage 2: Regressor",
    "XGBoost Regressor\ntarget: log(demand) | demand > 0", color="#154360", fontsize=9)
arrow(ax, 5, 5.6, 2.4, 4.3 + 1.0, color=C["approach1"])
arrow(ax, 5, 5.6, 7.6, 4.3 + 1.0, color=C["approach1"])
box(ax, 0.3, 3.05, 4.2, 0.9, "P(demand > 0)",
    "Probability county has non-zero demand", color=C["light_1"], text_color=DARK, fontsize=9)
box(ax, 5.5, 3.05, 4.2, 0.9, "E[demand | demand > 0]",
    "Expected magnitude\n(log-space  ->  expm1)", color=C["light_1"], text_color=DARK, fontsize=9)
arrow(ax, 2.4, 4.3, 2.4, 3.05 + 0.9)
arrow(ax, 7.6, 4.3, 7.6, 3.05 + 0.9)
box(ax, 2.5, 1.85, 5.0, 0.9, "Combined Prediction",
    "pred = P(>0) * E[demand | >0]", color=C["approach1"], fontsize=10)
arrow(ax, 2.4, 3.05, 4.7, 1.85 + 0.9, color="#aaaaaa")
arrow(ax, 7.6, 3.05, 5.3, 1.85 + 0.9, color="#aaaaaa")
box(ax, 0.5, 0.25, 9.0, 1.2, "Training Notes",
    "Classifier trains on ALL 31K rows  |  Regressor trains on 6,492 non-zero rows only\n"
    "Handles zero-inflation by design  |  Log-transform reduces skew in the regressor",
    color="#eaf2ff", text_color=DARK, fontsize=9)
save(fig, DIR3 / "01_arch_two_part_hurdle.png")

# ── Subchart 3-02 — Architecture: Filter + Log-Transform ─────────────────────
fig, ax = plt.subplots(figsize=(12, 8), facecolor=BG)
ax.set_xlim(0, 10); ax.set_ylim(0, 6.5); ax.axis("off")
ax.set_title("Architecture: Filter to FEMA-Declared + Log-Transform",
             fontsize=14, fontweight="bold", color=C["approach2"], pad=14)

box(ax, 1, 5.6, 8, 0.75, "Full Panel  (31,045 rows)", color=C["header"], fontsize=10)
box(ax, 1, 4.3, 8, 0.95, "Filter: FEMA-Declared Counties Only",
    "Keep rows where disasterNumber is not null\n15,378 rows remain  (49.5% of full panel)",
    color="#922b21", fontsize=9)
arrow(ax, 5, 5.6, 5, 4.3 + 0.95, color="#922b21")
box(ax, 1, 3.05, 8, 0.95, "Label Transformation:  log1p(demand_proxy)",
    "Compress heavy right tail  |  Forces the model to learn relative magnitudes",
    color=C["approach2"], fontsize=9)
arrow(ax, 5, 4.3, 5, 3.05 + 0.95, color=C["approach2"])
box(ax, 1, 1.8, 8, 0.95, "Single XGBoost Regressor",
    "Trained on 15,378 declared rows  |  At inference:  preds = expm1(model.predict(X))",
    color="#1e5631", fontsize=9)
arrow(ax, 5, 3.05, 5, 1.8 + 0.95, color=C["approach2"])
box(ax, 0.5, 0.25, 9.0, 1.2, "Key Insight",
    "FEMA-declared = what FEMA considers 'affected'\n"
    "Within that set, demand variation is learnable  |  Simpler than hurdle: fewer moving parts",
    color=C["light_2"], text_color=DARK, fontsize=9)
save(fig, DIR3 / "02_arch_filter_log.png")

# ── Subchart 3-03 — Architecture: Spatial Block CV ───────────────────────────
fig, ax = plt.subplots(figsize=(12, 8), facecolor=BG)
ax.set_xlim(0, 10); ax.set_ylim(0, 6.5); ax.axis("off")
ax.set_title("Architecture: Spatial Block Cross-Validation + Ensemble",
             fontsize=14, fontweight="bold", color=C["approach3"], pad=14)

fold_defs = [
    ("Gulf\n(TX,LA,MS,AL)", "#6c3483"),
    ("Atlantic S\n(FL,GA,SC)", "#1a5276"),
    ("Atlantic N\n(NC,VA,MD+)", "#1e8449"),
    ("Northeast\n(NY,NJ,NE)", "#b7770d"),
    ("PR+VI\n(Caribbean)", "#922b21"),
]
for i, (fl, fc2) in enumerate(fold_defs):
    box(ax, 0.2 + i * 1.95, 4.8, 1.75, 1.2, fl, color=fc2, fontsize=8)
ax.text(5, 4.55, "5-Fold Spatial Block Cross-Validation",
        ha="center", va="center", fontsize=9.5, color=DARK, style="italic")
box(ax, 0.5, 3.3, 9, 1.05,
    "For each fold: train on 4 regions, validate on the held-out region",
    "Measures geographic generalisation beyond temporal holdout\n"
    "Catches region-specific overfit that temporal splits miss",
    color=C["approach3"], fontsize=9)
arrow(ax, 5, 4.8, 5, 3.3 + 1.05, color=C["approach3"])
box(ax, 0.5, 2.1, 9, 0.95,
    "Ensemble: 5 fold-models + temporal model = 6 total",
    "Average predictions  |  Reduces prediction variance by ~20-30%",
    color="#4a235a", fontsize=9)
arrow(ax, 5, 3.3, 5, 2.1 + 0.95, color=C["approach3"])
box(ax, 0.5, 0.25, 9.0, 1.4, "Expected Benefits",
    "Spatial CV catches region-specific overfit\n"
    "Ensemble reduces prediction variance ~20-30%  |  Orthogonal to Approach 1/2 - stackable\n"
    "Directly proves spatial generalizability for the capstone report",
    color=C["light_3"], text_color=DARK, fontsize=9)
save(fig, DIR3 / "03_arch_spatial_cv.png")

# ── Subcharts 3-04, 3-05, 3-06 — Code blocks ─────────────────────────────────
CB = "#1e1e2e"; CF = "#cdd6f4"; KW = "#89b4fa"; FN = "#a6e3a1"
CM = "#6c7086"; ST = "#f38ba8"

code_blocks = [
    ("04_code_hurdle.png",
     "Code \u2014 Approach 1: Two-Part Hurdle  (08_train_model.py)",
     [
         ("# Stage 1: binary classifier",                   CM),
         ("clf = xgb.XGBClassifier(",                       KW),
         ("    n_estimators=500, max_depth=5,",             CF),
         ("    scale_pos_weight=4.0,  # handle imbalance",  FN),
         ("    eval_metric='logloss')",                     CF),
         ("y_binary = (y_train > 0).astype(int)",          KW),
         ("clf.fit(X_train, y_binary, ...)",               FN),
         ("",                                              CF),
         ("# Stage 2: regressor on non-zero rows only",    CM),
         ("mask = y_train > 0",                            KW),
         ("reg  = xgb.XGBRegressor(...)",                  KW),
         ("reg.fit(X_train[mask], np.log1p(y_train[mask]))", FN),
         ("",                                              CF),
         ("# Multiply at inference",                       CM),
         ("p_pos      = clf.predict_proba(X_test)[:,1]",   FN),
         ("log_demand = reg.predict(X_test)",              FN),
         ("pred       = p_pos * np.expm1(log_demand)",     KW),
     ]),
    ("05_code_filter_log.png",
     "Code \u2014 Approach 2: Filter+Log  (06_build_panel.py + 08_train_model.py)",
     [
         ("# 06_build_panel.py - filter to declared",  CM),
         ("panel = panel[",                            KW),
         ("    panel['fema_declared'] == 1].copy()",  CF),
         ("",                                          CF),
         ("# 08_train_model.py - log-transform target", CM),
         ("y_train_log = np.log1p(y_train)",           KW),
         ("",                                          CF),
         ("model.fit(X_train, y_train_log, ...)",      FN),
         ("",                                          CF),
         ("# Back-transform predictions",              CM),
         ("preds_log = model.predict(X_test)",         KW),
         ("preds     = np.expm1(preds_log)",           KW),
         ("",                                          CF),
         ("# Evaluate on original scale",              CM),
         ("rmse = root_mean_squared_error(y_test, preds)", FN),
     ]),
    ("06_code_spatial_cv.png",
     "Code \u2014 Approach 3: Spatial Block CV  (08_train_model.py)",
     [
         ("# State FIPS prefixes by region",           CM),
         ("GULF   = ['22','28','01','48']  # LA,MS,AL,TX", ST),
         ("ATL_S  = ['12','13','45']       # FL,GA,SC", ST),
         ("ATL_N  = ['37','51','24','10']  # NC,VA,MD,DE", ST),
         ("NORTHE = ['36','34','25','09']  # NY,NJ,MA,CT", ST),
         ("CARIB  = ['72','78']            # PR, VI",    ST),
         ("",                                           CF),
         ("folds = [GULF, ATL_S, ATL_N, NORTHE, CARIB]", KW),
         ("fold_models = []",                          KW),
         ("for hold in folds:",                        KW),
         ("    mask = ~df['fips'].str[:2].isin(hold)", FN),
         ("    m = xgb.XGBRegressor(...).fit(X[mask], y[mask])", FN),
         ("    fold_models.append(m)",                 FN),
         ("",                                          CF),
         ("# Ensemble average",                        CM),
         ("preds = np.mean([m.predict(X_test)",        KW),
         ("                 for m in fold_models], axis=0)", CF),
     ]),
]

for fname, title, lines in code_blocks:
    fig, ax = plt.subplots(figsize=(13, 6.5), facecolor=CB)
    ax.set_xlim(0, 10); ax.set_ylim(0, len(lines) + 1.2)
    ax.set_facecolor(CB)
    for sp in ax.spines.values():
        sp.set_color("#45475a")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set_title(title, fontsize=12, fontweight="bold", color="#cdd6f4", pad=12)
    for i, (line, color) in enumerate(lines):
        ax.text(0.25, len(lines) - i + 0.3, line,
                ha="left", va="top", fontsize=9.5, color=color, fontfamily="monospace")
    save(fig, DIR3 / fname)

# ── Subchart 3-07 — Pros & Cons ───────────────────────────────────────────────
pro_con_data = [
    (
        ["Theoretically correct for zero-inflated data",
         "SHAP interpretable per stage",
         "Classifier PR-AUC is meaningful",
         "Quantile models can run on regressor",
         "Best expected R\u00b2 improvement"],
        ["Two models to tune and maintain",
         "Stage 2 trains on only 6.5K rows",
         "Combined metric harder to explain",
         "More implementation effort (~3 hrs)"],
    ),
    (
        ["Simple: 1 model, 1 change",
         "Focuses on actionable declared cases",
         "Log transform well-established",
         "Works with existing 08 script",
         "Clear framing: declared = affected"],
        ["Loses 50.5% of training data",
         "Cannot predict un-declared events",
         "Label still supply-driven, not need-driven",
         "Moderate improvement expected"],
    ),
    (
        ["Tests generalisation property",
         "Required for rigorous capstone eval",
         "Ensemble reduces prediction variance",
         "Works with either Approach 1 or 2",
         "Directly addresses rubric criterion"],
        ["Does not fix zero-inflation by itself",
         "PR/VI folds may be small",
         "Adds ~1-2h of compute/code",
         "Must implement after Approach 1 or 2"],
    ),
]
names_pc  = ["Approach 1: Two-Part Hurdle", "Approach 2: Filter+Log", "Approach 3: Spatial CV"]
colors_pc = [C["approach1"], C["approach2"], C["approach3"]]

fig, axes = plt.subplots(1, 3, figsize=(20, 8), facecolor=BG)
fig.subplots_adjust(wspace=0.04)
for ax, (pros, cons), ac, name in zip(axes, pro_con_data, colors_pc, names_pc):
    ax.set_xlim(0, 10); ax.set_ylim(-2.8, 6.2); ax.axis("off")
    ax.set_facecolor("#fafafa")
    # Title bar
    ax.add_patch(FancyBboxPatch((0, 5.4), 10, 0.65,
        boxstyle="round,pad=0,rounding_size=0.2",
        facecolor=ac, edgecolor="white", lw=1.2, zorder=3))
    ax.text(5, 5.73, name, ha="center", va="center",
            color="white", fontsize=11, fontweight="bold", zorder=4)
    # PROS header
    ax.add_patch(FancyBboxPatch((0.2, 3.75), 9.6, 0.52,
        boxstyle="round,pad=0,rounding_size=0.15",
        facecolor=C["pro"], edgecolor="white", lw=1, zorder=3))
    ax.text(5, 4.01, "PROS", ha="center", va="center",
            color="white", fontsize=10, fontweight="bold", zorder=4)
    for i, t in enumerate(pros):
        ax.text(0.5,  3.55 - i * 0.56, "+",   ha="left",  va="top", color=C["pro"],  fontsize=12, fontweight="bold")
        ax.text(1.1,  3.55 - i * 0.56, t,     ha="left",  va="top", color=DARK,      fontsize=9)
    # CONS header
    ax.add_patch(FancyBboxPatch((0.2, -0.4), 9.6, 0.52,
        boxstyle="round,pad=0,rounding_size=0.15",
        facecolor=C["con"], edgecolor="white", lw=1, zorder=3))
    ax.text(5, -0.14, "CONS", ha="center", va="center",
            color="white", fontsize=10, fontweight="bold", zorder=4)
    for i, t in enumerate(cons):
        ax.text(0.5, -0.65 - i * 0.56, "\u2212",  ha="left", va="top", color=C["con"],  fontsize=12, fontweight="bold")
        ax.text(1.1, -0.65 - i * 0.56, t,         ha="left", va="top", color=DARK,      fontsize=9)

fig.suptitle("Pros & Cons \u2014 All Three Solution Approaches",
             fontsize=14, fontweight="bold", y=1.02, color=DARK)
save(fig, DIR3 / "07_pros_cons.png")

# ── Subchart 3-08 — Effort vs Impact Matrix ───────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 8), facecolor=BG)
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.set_xlabel("Implementation Effort (hours)", fontsize=11)
ax.set_ylabel("Expected Model Impact", fontsize=11)
ax.set_title("Effort vs. Impact Matrix", fontsize=14, fontweight="bold", pad=12)
ax.axvline(5, color="#cccccc", lw=1.2, linestyle="--")
ax.axhline(5, color="#cccccc", lw=1.2, linestyle="--")
# Quadrant labels
for tx, ty, txt in [
    (2.5, 9.0, "HIGH IMPACT\nLOW EFFORT"),
    (7.5, 9.0, "HIGH IMPACT\nHIGH EFFORT"),
    (2.5, 1.2, "LOW IMPACT\nLOW EFFORT"),
    (7.5, 1.2, "LOW IMPACT\nHIGH EFFORT"),
]:
    ax.text(tx, ty, txt, ha="center", va="center", fontsize=9.5,
            color="#aaaaaa", style="italic")
# Points
for lbl, ex, ey, ec in [
    ("Approach 2\n(Filter+Log)",  3, 7, C["approach2"]),
    ("Approach 1\n(Two-Part)",    5, 9, C["approach1"]),
    ("Approach 3\n(Spatial CV)",  6, 6, C["approach3"]),
    ("Current\nBaseline",         0.2, 0.8, "#e74c3c"),
]:
    ax.scatter(ex, ey, s=320, color=ec, zorder=5, edgecolors="white", lw=2)
    ax.text(ex + 0.35, ey + 0.35, lbl, fontsize=9.5, color=ec, fontweight="bold")
ax.set_xticks([0, 2, 4, 6, 8, 10])
ax.set_xticklabels(["0h", "2h", "4h", "6h", "8h", "10h"], fontsize=10)
ax.set_yticks([0, 2, 4, 6, 8, 10])
ax.set_yticklabels(["None", "Low", "Mid", "High", "V.High", "Max"], fontsize=10)
style_ax(ax)
save(fig, DIR3 / "08_effort_impact_matrix.png")

# ── Subchart 3-09 — Recommended Execution Order ───────────────────────────────
fig, ax = plt.subplots(figsize=(11, 8), facecolor=BG)
ax.set_xlim(0, 10); ax.set_ylim(0, 7.5); ax.axis("off")
ax.set_title("Recommended Execution Order",
             fontsize=14, fontweight="bold", color=DARK, pad=14)
steps_ex = [
    ("Step 1  (Today, 2 hrs)",
     "Approach 2: filter to declared + log1p target\nFast win, low risk, clear interpretation",
     C["approach2"]),
    ("Step 2  (Tomorrow, 3 hrs)",
     "Approach 1: two-part hurdle model\nClassifier + regressor, verify R\u00b2 > 0.10",
     C["approach1"]),
    ("Step 3  (Day 3, 2 hrs)",
     "Approach 3: spatial block CV on top\n5-region holdout + ensemble average",
     C["approach3"]),
    ("Step 4  (Day 4, 2 hrs)",
     "Re-run scripts 07-10 with the new model\nRegenerate all figures + verify report metrics",
     C["header"]),
]
y_pos = 7.0
for i, (title, body, fc) in enumerate(steps_ex):
    ax.add_patch(FancyBboxPatch((0.3, y_pos - 1.4), 9.4, 1.2,
        boxstyle="round,pad=0,rounding_size=0.25",
        facecolor=fc, edgecolor="white", lw=1.5, zorder=3))
    ax.text(5, y_pos - 0.68, title, ha="center", va="center",
            color="white", fontsize=11, fontweight="bold", zorder=4)
    ax.text(5, y_pos - 1.1,  body,  ha="center", va="center",
            color="white", fontsize=9.5, zorder=4, linespacing=1.4)
    if i < len(steps_ex) - 1:
        ax.annotate("", xy=(5, y_pos - 1.4), xytext=(5, y_pos - 1.62),
            arrowprops=dict(arrowstyle="-|>", color="#555555", lw=2.8, mutation_scale=16),
            zorder=5)
    y_pos -= 1.72
save(fig, DIR3 / "09_execution_order.png")

# ── Subchart 3-10 — Expected Metrics Table ────────────────────────────────────
rows_m = [
    ("Metric",          "Current",      "After Approach 2",  "After Approach 1"),
    ("R\u00b2 (test)",  "-0.001",       "0.10 \u2013 0.25",  "0.25 \u2013 0.45"),
    ("RMSE improve",    "0%",           "10 \u2013 25%",     "25 \u2013 45%"),
    ("PR-AUC",          "N/A",          "N/A",               "0.65 \u2013 0.80"),
    ("Pred range",      "[0.07, 1.6]",  "[0, 4.0]",          "[0, 5.0]"),
    ("Differentiation", "Low",          "Medium",            "High"),
    ("Equity gap (LP)", "+52.1pp",      "+52.1pp",           "+55\u201360pp"),
]
cw_m   = [0.30, 0.16, 0.26, 0.26]
rbg_m  = [C["header"]] + ["#ecf0f1", "#f8f9fa"] * 3
rfg_m  = ["white"] + [DARK] * 6

fig, ax = plt.subplots(figsize=(12, 5.5), facecolor=BG)
ax.axis("off")
for ri, row in enumerate(rows_m):
    for ci, (cell, cw) in enumerate(zip(row, cw_m)):
        xp = sum(cw_m[:ci]) + 0.01
        yp = 0.95 - ri * 0.136
        fc_cell = (rbg_m[ri] if ri == 0 else
                   C["approach1"] if ci == 3 and ri > 0 else
                   C["approach2"] if ci == 2 and ri > 0 else rbg_m[ri])
        fc_text = ("white" if ri == 0 or (ci in (2, 3) and ri > 0) else rfg_m[ri])
        ax.add_patch(FancyBboxPatch((xp, yp), cw - 0.015, 0.125,
            boxstyle="round,pad=0,rounding_size=0.01",
            facecolor=fc_cell, edgecolor="white", lw=1.0,
            transform=ax.transAxes, zorder=3))
        ax.text(xp + (cw - 0.015) / 2, yp + 0.063, cell,
                ha="center", va="center",
                fontsize=9.5 if ri > 0 else 10.5,
                fontweight="bold" if ri == 0 or ci == 0 else "normal",
                color=fc_text, transform=ax.transAxes, zorder=4)
ax.set_title("Expected Metrics After Fix",
             fontsize=13, fontweight="bold", pad=20, color=DARK)
save(fig, DIR3 / "10_expected_metrics.png")


# ── Summary ────────────────────────────────────────────────────────────────────
c2 = sorted(DIR2.glob("*.png"))
c3 = sorted(DIR3.glob("*.png"))
print(f"\nSaved {len(c2)} subcharts to {DIR2}")
for f in c2:
    print(f"  {f.name}")
print(f"\nSaved {len(c3)} subcharts to {DIR3}")
for f in c3:
    print(f"  {f.name}")
print("\nDone.")
