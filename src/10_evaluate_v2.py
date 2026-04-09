"""
10_evaluate_v2.py
Redesigned, visually-polished evaluation figures for the hurricane
resource-allocation project.

Outputs (saved to outputs/figures/):
  fig1_policy_dashboard.png      – KPI cards + 3-metric bar comparison
  fig2_per_storm_distribution.png – Violin + strip plots across 46 storms
  fig3_equity_tradeoff.png        – High-SVI vs Low-SVI scatter by policy
  fig4_robustness_timeline.png    – Per-storm timeline with ±20% shock bands
  fig5_storm_ranking.png          – Horizontal lollipop sorted by LP unmet
  fig6_demand_map.png             – Geographic demand/SVI bubble map (LAURA 2020)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE   = Path(".")
PROC   = BASE / "data" / "processed"
ALLOC  = BASE / "outputs" / "allocations"
FIGS   = BASE / "outputs" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# ── Design tokens ──────────────────────────────────────────────────────────────
BG    = "#fafafa"
DARK  = "#0f172a"
MID   = "#64748b"
LIGHT = "#cbd5e1"
GRID  = "#e2e8f0"
WHITE = "#ffffff"

PC = {
    "population_proportional": "#94a3b8",   # slate
    "svi_weighted":            "#f59e0b",   # amber
    "model_lp_equity":         "#10b981",   # emerald
}
PL = {
    "population_proportional": "Population\nProportional",
    "svi_weighted":            "SVI-Weighted",
    "model_lp_equity":         "Model + LP\n+ Equity",
}
POLICIES = list(PC.keys())

# ── Global rcParams ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    WHITE,
    "axes.edgecolor":    LIGHT,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        GRID,
    "grid.linewidth":    0.7,
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.titlepad":     10,
    "axes.labelsize":    11,
    "axes.labelcolor":   MID,
    "xtick.color":       MID,
    "ytick.color":       MID,
    "legend.frameon":    False,
    "savefig.facecolor": BG,
    "savefig.dpi":       180,
    "savefig.bbox":      "tight",
})

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data …")
results = pd.read_parquet(ALLOC / "all_storms_comparison.parquet")
preds   = pd.read_parquet(PROC  / "predictions.parquet")

try:
    panel = pd.read_parquet(PROC / "panel.parquet")
    HAS_PANEL = True
except FileNotFoundError:
    HAS_PANEL = False
    print("  panel.parquet not found – skipping map figure")

# Convenience sub-frames
lp_df  = results[results["policy"] == "model_lp_equity"].copy()
pop_df = results[results["policy"] == "population_proportional"].copy()
svi_df = results[results["policy"] == "svi_weighted"].copy()

avg = results.groupby("policy")[
    ["pct_unmet", "coverage_overall", "coverage_high_svi", "coverage_low_svi", "equity_gap"]
].mean()

# ══════════════════════════════════════════════════════════════════════════════
# Fig 1 – Policy Dashboard (KPI cards + bar charts)
# ══════════════════════════════════════════════════════════════════════════════
print("Rendering Fig 1 – Policy Dashboard …")

fig = plt.figure(figsize=(18, 11), facecolor=BG)
fig.suptitle(
    "Hurricane Resource Allocation: Policy Comparison Dashboard",
    fontsize=20, fontweight="bold", color=DARK, y=0.97
)

# GridSpec: top row = 3 KPI panels; bottom row = 3 metric bar charts
gs = gridspec.GridSpec(
    2, 3,
    figure=fig,
    top=0.90, bottom=0.07,
    left=0.06, right=0.97,
    hspace=0.55, wspace=0.38,
)

# ─ KPI cards (top row) ─────────────────────────────────────────────────────
kpi_titles  = ["Avg % Demand Unmet", "High-SVI Coverage", "Equity Gap\n(High − Low SVI)"]
kpi_keys    = ["pct_unmet", "coverage_high_svi", "equity_gap"]
kpi_fmts    = ["{:.1%}", "{:.1%}", "{:+.1%}"]
kpi_arrows  = ["↓ lower is better", "↑ higher is better", "↑ positive = high-SVI favoured"]

for col, (title, key, fmt, arrow) in enumerate(zip(kpi_titles, kpi_keys, kpi_fmts, kpi_arrows)):
    ax = fig.add_subplot(gs[0, col])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(WHITE)

    # Card background
    rect = mpatches.FancyBboxPatch(
        (0.03, 0.05), 0.94, 0.90,
        boxstyle="round,pad=0.02",
        linewidth=1.2, edgecolor=GRID,
        facecolor=WHITE, transform=ax.transAxes, zorder=0,
    )
    ax.add_patch(rect)

    ax.text(0.5, 0.88, title, ha="center", va="top", fontsize=11,
            color=MID, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.72, arrow, ha="center", va="top", fontsize=8.5,
            color=LIGHT, fontstyle="italic", transform=ax.transAxes)

    y_val = 0.50
    for pol in POLICIES:
        val   = avg.loc[pol, key]
        color = PC[pol]
        label = PL[pol].replace("\n", " ")
        ax.text(0.08, y_val, f"● {label}", ha="left", va="center",
                fontsize=10, color=color, fontweight="bold",
                transform=ax.transAxes)
        ax.text(0.92, y_val, fmt.format(val), ha="right", va="center",
                fontsize=12, color=color, fontweight="bold",
                transform=ax.transAxes)
        y_val -= 0.18

# ─ Bar charts (bottom row) ─────────────────────────────────────────────────
bar_metrics = [
    ("pct_unmet",          "Average % Demand Unmet",             "% Unmet Demand",          True),
    ("coverage_high_svi",  "Average High-SVI Coverage",          "Coverage Rate",            False),
    ("coverage_low_svi",   "Average Low-SVI Coverage",           "Coverage Rate",            False),
]

x = np.arange(len(POLICIES))
w = 0.55

for col, (key, title, ylabel, invert) in enumerate(bar_metrics):
    ax = fig.add_subplot(gs[1, col])
    vals   = [avg.loc[p, key] for p in POLICIES]
    colors = [PC[p] for p in POLICIES]
    bars   = ax.bar(x, vals, width=w, color=colors, zorder=3, linewidth=0)

    # Value labels on bars
    for bar, val in zip(bars, vals):
        ypos = bar.get_height() + 0.005
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{val:.1%}", ha="center", va="bottom",
                fontsize=10, color=DARK, fontweight="bold")

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([PL[p] for p in POLICIES], fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylim(0, max(vals) * 1.22)
    ax.axhline(0, color=LIGHT, linewidth=0.8)
    ax.spines["left"].set_visible(False)
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)

    # Highlight best bar
    best_idx = np.argmin(vals) if invert else np.argmax(vals)
    bars[best_idx].set_edgecolor(DARK)
    bars[best_idx].set_linewidth(2.0)

plt.savefig(FIGS / "fig1_policy_dashboard.png")
plt.close()
print("  [OK] fig1_policy_dashboard.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2 – Per-storm distribution (violin + strip)
# ══════════════════════════════════════════════════════════════════════════════
print("Rendering Fig 2 – Per-storm distributions …")

metrics_v = [
    ("pct_unmet",         "% Demand Unmet",      True),
    ("coverage_high_svi", "High-SVI Coverage",   False),
    ("equity_gap",        "Equity Gap\n(High−Low SVI Coverage)", False),
]

fig, axes = plt.subplots(1, 3, figsize=(17, 7), facecolor=BG)
fig.suptitle(
    "Distribution of Policy Outcomes Across 46 Test Storms",
    fontsize=17, fontweight="bold", color=DARK, y=1.02
)

for ax, (key, title, lower_better) in zip(axes, metrics_v):
    pos    = np.arange(len(POLICIES))
    groups = [results.loc[results["policy"] == p, key].dropna().values for p in POLICIES]

    # Violin
    parts = ax.violinplot(groups, positions=pos, widths=0.55,
                          showmedians=False, showextrema=False)
    for i, (body, pol) in enumerate(zip(parts["bodies"], POLICIES)):
        body.set_facecolor(PC[pol])
        body.set_alpha(0.35)
        body.set_edgecolor(PC[pol])

    # Box + median
    bp = ax.boxplot(groups, positions=pos, widths=0.18,
                    patch_artist=True, notch=False,
                    whiskerprops=dict(color=MID, linewidth=1.2),
                    capprops=dict(color=MID, linewidth=1.2),
                    flierprops=dict(marker="o", markerfacecolor=MID,
                                   markersize=3, linestyle="none"),
                    medianprops=dict(color=DARK, linewidth=2.0))
    for patch, pol in zip(bp["boxes"], POLICIES):
        patch.set_facecolor(PC[pol])
        patch.set_alpha(0.85)

    # Strip (jittered points)
    rng = np.random.default_rng(42)
    for i, (vals, pol) in enumerate(zip(groups, POLICIES)):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(pos[i] + jitter, vals,
                   color=PC[pol], s=22, alpha=0.55,
                   zorder=4, linewidths=0)

    ax.set_title(title)
    ax.set_xticks(pos)
    ax.set_xticklabels([PL[p] for p in POLICIES], fontsize=9.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("")
    ax.spines["left"].set_visible(False)
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)

    direction_lbl = "↓ lower is better" if lower_better else "↑ higher is better"
    ax.text(0.97, 0.02, direction_lbl, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8.5, color=LIGHT, fontstyle="italic")

plt.tight_layout()
plt.savefig(FIGS / "fig2_per_storm_distribution.png")
plt.close()
print("  [OK] fig2_per_storm_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3 – Equity tradeoff scatter (High-SVI vs Low-SVI coverage)
# ══════════════════════════════════════════════════════════════════════════════
print("Rendering Fig 3 – Equity tradeoff scatter …")

fig, ax = plt.subplots(figsize=(10, 8), facecolor=BG)

for pol in POLICIES:
    sub = results[results["policy"] == pol]
    ax.scatter(
        sub["coverage_low_svi"], sub["coverage_high_svi"],
        color=PC[pol], s=55, alpha=0.70, zorder=3, linewidths=0, label=PL[pol].replace("\n", " ")
    )
    # Mean marker
    mx = sub["coverage_low_svi"].mean()
    my = sub["coverage_high_svi"].mean()
    ax.scatter(mx, my, color=PC[pol], s=220, zorder=5,
               edgecolors=DARK, linewidths=1.8, marker="D")

# Equity line (perfect equality)
lo, hi = 0, 1.05
ax.plot([lo, hi], [lo, hi], "--", color=LIGHT, linewidth=1.5, label="Perfect equality (High = Low)")
ax.fill_between([lo, hi], [lo, hi], [hi, hi],
                alpha=0.04, color=PC["model_lp_equity"], label="High-SVI favoured region")

ax.set_xlabel("Low-SVI County Coverage")
ax.set_ylabel("High-SVI County Coverage")
ax.set_title("Equity Tradeoff: High-SVI vs Low-SVI Coverage\n(diamonds = policy averages; dots = individual storms)")
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_aspect("equal")

ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

legend_handles = [
    Line2D([0], [0], color=PC[p], marker="o", markersize=8,
           linewidth=0, label=PL[p].replace("\n", " "))
    for p in POLICIES
] + [
    Line2D([0], [0], color=LIGHT, linewidth=1.5, linestyle="--", label="Perfect equality"),
]
ax.legend(handles=legend_handles, loc="lower right", fontsize=10)

plt.tight_layout()
plt.savefig(FIGS / "fig3_equity_tradeoff.png")
plt.close()
print("  [OK] fig3_equity_tradeoff.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4 – Robustness timeline (LP policy, per storm, with ±20% shock bands)
# ══════════════════════════════════════════════════════════════════════════════
print("Rendering Fig 4 – Robustness timeline …")

rob = (
    lp_df[["storm_id", "storm_name", "storm_year",
            "pct_unmet", "unmet_shock_plus20", "unmet_shock_minus20",
            "total_demand"]]
    .copy()
    .sort_values(["storm_year", "storm_name"])
    .reset_index(drop=True)
)

# Convert absolute unmet to pct_unmet for shock scenarios
rob["pct_shock_hi"]  = rob["unmet_shock_plus20"]  / (rob["total_demand"] * 1.20)
rob["pct_shock_lo"]  = rob["unmet_shock_minus20"] / (rob["total_demand"] * 0.80)

# Replace any NaN with central estimate
rob["pct_shock_hi"]  = rob["pct_shock_hi"].fillna(rob["pct_unmet"])
rob["pct_shock_lo"]  = rob["pct_shock_lo"].fillna(rob["pct_unmet"])

x_pos = np.arange(len(rob))
labels = rob.apply(lambda r: f"{r['storm_name'].title()}\n'{str(r['storm_year'])[2:]}", axis=1)

fig, ax = plt.subplots(figsize=(20, 7), facecolor=BG)

# Shock band (±20%)
ax.fill_between(x_pos, rob["pct_shock_lo"], rob["pct_shock_hi"],
                alpha=0.18, color=PC["model_lp_equity"], label="±20% demand shock range")

# Central estimate
ax.plot(x_pos, rob["pct_unmet"], "o-",
        color=PC["model_lp_equity"], linewidth=2.2, markersize=7,
        markeredgecolor=DARK, markeredgewidth=0.8,
        label="LP+Equity (baseline demand)", zorder=4)

# Pop-proportional for reference
pop_sorted = pop_df.sort_values(["storm_year", "storm_name"]).reset_index(drop=True)
ax.plot(x_pos, pop_sorted["pct_unmet"], "s--",
        color=PC["population_proportional"], linewidth=1.5, markersize=5,
        alpha=0.65, label="Pop-Proportional (baseline demand)")

# Year dividers
year_breaks = rob[rob["storm_year"] != rob["storm_year"].shift()].index.tolist()
for b in year_breaks[1:]:
    ax.axvline(b - 0.5, color=LIGHT, linewidth=1.0, linestyle=":")

# Year labels at top
for b_start, b_end in zip(year_breaks, year_breaks[1:] + [len(rob)]):
    yr = rob.loc[b_start, "storm_year"]
    mid = (b_start + b_end - 1) / 2
    ax.text(mid, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.05,
            str(yr), ha="center", va="bottom", fontsize=9.5,
            color=MID, fontweight="bold")

ax.set_title("LP+Equity Policy: % Demand Unmet per Storm with ±20% Demand Shock Sensitivity")
ax.set_ylabel("% Demand Unmet")
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_xlim(-0.6, len(rob) - 0.4)
ax.legend(loc="upper right", fontsize=10)
ax.spines["left"].set_visible(False)
ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)

# Re-annotate year labels now that y-lim is finalised
ylim_top = ax.get_ylim()[1]
for b_start, b_end in zip(year_breaks, year_breaks[1:] + [len(rob)]):
    yr  = rob.loc[b_start, "storm_year"]
    mid = (b_start + b_end - 1) / 2
    ax.text(mid, ylim_top * 0.98, str(yr),
            ha="center", va="top", fontsize=9.5,
            color=MID, fontweight="bold")

plt.tight_layout()
plt.savefig(FIGS / "fig4_robustness_timeline.png")
plt.close()
print("  [OK] fig4_robustness_timeline.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 5 – Storm ranking (horizontal lollipop, sorted by LP unmet)
# ══════════════════════════════════════════════════════════════════════════════
print("Rendering Fig 5 – Storm ranking …")

rank = (
    lp_df[["storm_id", "storm_name", "storm_year", "pct_unmet"]]
    .rename(columns={"pct_unmet": "lp_unmet"})
    .merge(
        pop_df[["storm_id", "pct_unmet"]].rename(columns={"pct_unmet": "pop_unmet"}),
        on="storm_id"
    )
    .sort_values("lp_unmet")
    .reset_index(drop=True)
)

rank["label"] = rank.apply(
    lambda r: f"{r['storm_name'].title()} ({int(r['storm_year'])})", axis=1
)
rank["gain"]  = rank["pop_unmet"] - rank["lp_unmet"]

n    = len(rank)
ypos = np.arange(n)

fig, ax = plt.subplots(figsize=(14, max(8, n * 0.38)), facecolor=BG)

# Pop baseline dots
ax.scatter(rank["pop_unmet"], ypos, color=PC["population_proportional"],
           s=55, zorder=4, label="Pop-Proportional", alpha=0.85)

# LP dots
ax.scatter(rank["lp_unmet"], ypos, color=PC["model_lp_equity"],
           s=65, zorder=5, label="LP+Equity", alpha=0.95)

# Connector lines (gain bars)
for i, row in rank.iterrows():
    color = "#10b981" if row["gain"] > 0 else "#ef4444"
    ax.plot([row["pop_unmet"], row["lp_unmet"]], [i, i],
            color=color, linewidth=1.8, alpha=0.55, zorder=3)

# Colour scale on gain: annotate large improvements
threshold = rank["gain"].quantile(0.85)
for i, row in rank.iterrows():
    if row["gain"] >= threshold:
        ax.text(row["lp_unmet"] - 0.005, i,
                f"−{row['gain']:.0%}", va="center", ha="right",
                fontsize=7.5, color=PC["model_lp_equity"], fontweight="bold")

ax.set_yticks(ypos)
ax.set_yticklabels(rank["label"], fontsize=8)
ax.set_xlabel("% Demand Unmet")
ax.set_title(
    "Storm-Level Unmet Demand: LP+Equity vs Population-Proportional Baseline\n"
    "(sorted by LP+Equity performance; green lines = improvement)"
)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.axvline(rank["lp_unmet"].mean(),  color=PC["model_lp_equity"],
           linewidth=1.2, linestyle="--", alpha=0.6)
ax.axvline(rank["pop_unmet"].mean(), color=PC["population_proportional"],
           linewidth=1.2, linestyle="--", alpha=0.6)

legend_elements = [
    Line2D([0], [0], marker="o", color="w",
           markerfacecolor=PC["population_proportional"], markersize=9,
           label="Pop-Proportional baseline"),
    Line2D([0], [0], marker="o", color="w",
           markerfacecolor=PC["model_lp_equity"], markersize=9,
           label="LP+Equity policy"),
    Line2D([0], [0], color="#10b981", linewidth=2, label="Improvement (LP < Pop)"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=10)
ax.spines["left"].set_visible(False)
ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(FIGS / "fig5_storm_ranking.png")
plt.close()
print("  [OK] fig5_storm_ranking.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 6 – Geographic demand / SVI bubble map (LAURA 2020)
# ══════════════════════════════════════════════════════════════════════════════
print("Rendering Fig 6 – Geographic demand map …")

if HAS_PANEL:
    laura = panel[
        (panel["storm_name"].str.upper() == "LAURA") &
        (panel["storm_year"] == 2020)
    ].copy()
else:
    # Fall back to predictions parquet
    laura_preds = preds[
        (preds["storm_name"].str.upper() == "LAURA") &
        (preds["storm_year"] == 2020)
    ].copy()
    laura = laura_preds

# We need lat/lon; try to load from a county centroids file if available
try:
    centroids = pd.read_parquet(PROC / "county_centroids.parquet")
    laura = laura.merge(centroids[["fips", "lat", "lon"]], on="fips", how="inner")
    HAS_COORDS = True
except FileNotFoundError:
    # Try county panel
    if "county_lat" in laura.columns and "county_lon" in laura.columns:
        laura = laura.rename(columns={"county_lat": "lat", "county_lon": "lon"})
        HAS_COORDS = True
    elif "lat" in laura.columns and "lon" in laura.columns:
        HAS_COORDS = True
    else:
        HAS_COORDS = False
        print("  No lat/lon data found – generating synthetic coordinate placeholder")

if HAS_COORDS and len(laura) > 0:
    # Determine demand column
    demand_col = "demand_proxy" if "demand_proxy" in laura.columns else \
                 "pred_demand"  if "pred_demand"  in laura.columns else None
    svi_col    = "svi_overall"  if "svi_overall"  in laura.columns else None

    if demand_col is None:
        print("  No demand column found – skipping map")
    else:
        demand_vals = laura[demand_col].clip(lower=0)
        svi_vals    = laura[svi_col].fillna(0.5) if svi_col else pd.Series(0.5, index=laura.index)

        # Size ∝ demand, colour ∝ SVI
        size_scale = 1200 / demand_vals.max() if demand_vals.max() > 0 else 1
        sizes = (demand_vals * size_scale).clip(lower=8)

        fig, ax = plt.subplots(figsize=(14, 9), facecolor=BG)

        sc = ax.scatter(
            laura["lon"], laura["lat"],
            s=sizes, c=svi_vals,
            cmap="RdYlGn_r",
            alpha=0.78, linewidths=0.5,
            edgecolors="white",
            vmin=0, vmax=1,
            zorder=4
        )

        cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("SVI Score (higher = more vulnerable)", fontsize=10)
        cbar.ax.tick_params(labelsize=9)

        # Legend for bubble size
        for ref_val in [0.2, 0.5, 1.0]:
            max_d = demand_vals.max()
            if max_d > 0 and ref_val <= max_d:
                ax.scatter([], [], s=ref_val * size_scale, c="grey", alpha=0.4,
                           label=f"Demand = {ref_val:.1f}")
        ax.legend(title="Demand Proxy", loc="lower left", fontsize=9, title_fontsize=9)

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(
            "Hurricane Laura (2020): County-Level Demand Proxy & SVI\n"
            "Bubble size ∝ predicted demand  |  Colour = SVI vulnerability score"
        )
        ax.set_facecolor("#f0f4f8")
        ax.grid(True, color=GRID, linewidth=0.5)

        plt.tight_layout()
        plt.savefig(FIGS / "fig6_demand_map.png")
        plt.close()
        print("  [OK] fig6_demand_map.png")
else:
    # Fallback: demand distribution for LAURA
    print("  Falling back to LAURA demand distribution chart")
    laura_p = preds[
        (preds["storm_name"].str.upper() == "LAURA") &
        (preds["storm_year"] == 2020)
    ].copy()

    if len(laura_p) == 0:
        print("  No LAURA predictions found – skipping fig 6")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)

        # Actual vs predicted
        axes[0].scatter(
            laura_p["demand_proxy"], laura_p["pred_demand"],
            color=PC["model_lp_equity"], alpha=0.6, s=50, linewidths=0
        )
        max_v = max(laura_p["demand_proxy"].max(), laura_p["pred_demand"].max())
        axes[0].plot([0, max_v], [0, max_v], "--", color=LIGHT, linewidth=1.5)
        axes[0].set_xlabel("Actual Demand Proxy")
        axes[0].set_ylabel("Predicted Demand")
        axes[0].set_title("Laura 2020: Actual vs Predicted Demand")

        # Demand vs SVI
        if "svi_overall" in laura_p.columns:
            axes[1].scatter(
                laura_p["svi_overall"], laura_p["demand_proxy"],
                color=PC["svi_weighted"], alpha=0.6, s=50, linewidths=0
            )
            axes[1].set_xlabel("SVI Score")
            axes[1].set_ylabel("Demand Proxy")
            axes[1].set_title("Laura 2020: Demand vs Vulnerability (SVI)")

        plt.tight_layout()
        plt.savefig(FIGS / "fig6_demand_map.png")
        plt.close()
        print("  [OK] fig6_demand_map.png (fallback distribution chart)")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 7 – SHAP feature importance (redesigned)
# ══════════════════════════════════════════════════════════════════════════════
print("Rendering Fig 7 – SHAP feature importance (redesigned) …")

try:
    import shap, pickle, xgboost as xgb

    MODELS = BASE / "outputs" / "models"
    model  = xgb.XGBRegressor()
    model.load_model(str(MODELS / "xgb_point.json"))

    import json as _json
    features = _json.loads((MODELS / "features.json").read_text())

    X_test = preds[features].fillna(0)
    explainer   = shap.TreeExplainer(model)
    sample_size = min(1500, len(X_test))
    X_sample    = X_test.sample(sample_size, random_state=42)
    shap_values = explainer.shap_values(X_sample)

    # Mean |SHAP| per feature
    mean_shap = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=features
    ).sort_values(ascending=True).tail(15)

    # Nicer feature names
    rename_map = {
        "peak_wind_kt":          "Peak Wind (kt)",
        "min_dist_km":           "Min Distance to Track (km)",
        "svi_overall":           "SVI Overall Score",
        "log_total_pop":         "log(Total Population)",
        "log_total_housing_units":"log(Housing Units)",
        "EP_NOVEH":              "% No Vehicle",
        "EP_AGE65":              "% Age 65+",
        "EP_POV150":             "% Below 150% Poverty",
        "EP_MOBILE":             "% Mobile Homes",
        "EP_DISABL":             "% Disabled",
        "EP_UNINSUR":            "% Uninsured",
        "EP_CROWD":              "% Crowded Housing",
        "prior_hits":            "Prior Storm Hits (county)",
        "median_hh_income_10k":  "Median HH Income ($10k)",
        "log_no_vehicle":        "log(# No Vehicle HH)",
        "log_pop_65plus":        "log(Pop. 65+)",
    }
    mean_shap.index = [rename_map.get(f, f) for f in mean_shap.index]

    fig, ax = plt.subplots(figsize=(11, 7), facecolor=BG)

    colors_bar = [
        "#10b981" if v >= mean_shap.quantile(0.75)
        else "#64748b" if v >= mean_shap.quantile(0.4)
        else "#cbd5e1"
        for v in mean_shap.values
    ]
    bars = ax.barh(mean_shap.index, mean_shap.values,
                   color=colors_bar, linewidth=0, height=0.65)

    # Value labels
    for bar, val in zip(bars, mean_shap.values):
        ax.text(val + mean_shap.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9, color=DARK)

    ax.set_xlabel("Mean |SHAP value|  (average impact on prediction)")
    ax.set_title("XGBoost Feature Importance: Top 15 Features by Mean |SHAP|\n"
                 "(green = highest impact features)")

    legend_els = [
        mpatches.Patch(color="#10b981", label="Top-quartile impact"),
        mpatches.Patch(color="#64748b", label="Mid-range impact"),
        mpatches.Patch(color="#cbd5e1", label="Lower impact"),
    ]
    ax.legend(handles=legend_els, loc="lower right", fontsize=9)
    ax.spines["bottom"].set_visible(False)
    ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.set_xlim(0, mean_shap.max() * 1.20)

    plt.tight_layout()
    plt.savefig(FIGS / "fig7_shap_importance.png")
    plt.close()
    print("  [OK] fig7_shap_importance.png")

except Exception as e:
    print(f"  SHAP fig skipped: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n-- Key Statistics -------------------------------------------------------")
for pol in POLICIES:
    row = avg.loc[pol]
    print(f"  {PL[pol].replace(chr(10), ' '):<34}  "
          f"unmet={row['pct_unmet']:.1%}  "
          f"hi-SVI cov={row['coverage_high_svi']:.1%}  "
          f"equity gap={row['equity_gap']:+.1%}")

print(f"\nAll figures saved to: {FIGS.resolve()}")
