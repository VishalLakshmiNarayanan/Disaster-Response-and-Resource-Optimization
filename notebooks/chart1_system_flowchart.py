"""
Chart 1: Complete System Architecture Flowchart
Shows every component from raw data sources to final outputs.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT = "../outputs/figures/chart1_system_flowchart.png"

# ── Colour palette ────────────────────────────────────────────────────────────
C = {
    "source":   "#1a5276",   # dark blue   – external data sources
    "script":   "#1e8449",   # dark green  – Python scripts
    "data":     "#d4e6f1",   # light blue  – parquet / stored data
    "model":    "#7d3c98",   # purple      – ML components
    "optim":    "#b7770d",   # gold        – optimization
    "output":   "#922b21",   # dark red    – final outputs
    "bg":       "#fdfefe",   # near-white  – background
    "arrow":    "#555555",   # grey        – arrows
    "text_lt":  "#ffffff",   # white       – text on dark bg
    "text_dk":  "#1a1a1a",   # near-black  – text on light bg
}

fig, ax = plt.subplots(figsize=(22, 28))
ax.set_xlim(0, 22)
ax.set_ylim(0, 28)
ax.axis("off")
fig.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["bg"])

# ── Helper functions ──────────────────────────────────────────────────────────

def box(ax, x, y, w, h, label, sublabel="", color="#1a5276",
        text_color="white", fontsize=9, radius=0.25):
    """Draw a rounded rectangle with label and optional sublabel."""
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad=0,rounding_size={radius}",
                          linewidth=1.2, edgecolor="white",
                          facecolor=color, zorder=3)
    ax.add_patch(rect)
    if sublabel:
        ax.text(x + w/2, y + h * 0.62, label,
                ha="center", va="center", color=text_color,
                fontsize=fontsize, fontweight="bold", zorder=4)
        ax.text(x + w/2, y + h * 0.28, sublabel,
                ha="center", va="center", color=text_color,
                fontsize=fontsize - 1.5, style="italic", zorder=4,
                wrap=True)
    else:
        ax.text(x + w/2, y + h/2, label,
                ha="center", va="center", color=text_color,
                fontsize=fontsize, fontweight="bold", zorder=4)

def arrow(ax, x1, y1, x2, y2, color="#555555", lw=1.8):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=14),
                zorder=2)

def section_label(ax, x, y, text):
    ax.text(x, y, text, ha="left", va="center",
            fontsize=8.5, color="#777777", style="italic", zorder=5)

def divider(ax, y, color="#dddddd"):
    ax.axhline(y, color=color, lw=0.8, linestyle="--", zorder=1)

# ─────────────────────────────────────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────────────────────────────────────
ax.text(11, 27.4, "Hurricane Resource Allocation – Full System Pipeline",
        ha="center", va="center", fontsize=16, fontweight="bold",
        color="#1a1a1a")
ax.text(11, 27.0, "End-to-end architecture: 5 data sources  →  10 scripts  →  3 allocation policies  →  6 output figures",
        ha="center", va="center", fontsize=10, color="#555555")

# ─────────────────────────────────────────────────────────────────────────────
# ROW 1 – External Data Sources  (y ≈ 25.0)
# ─────────────────────────────────────────────────────────────────────────────
divider(ax, 26.6)
section_label(ax, 0.2, 26.3, "LAYER 1 – EXTERNAL DATA SOURCES")

sources = [
    ("HURDAT2\n(NHC / NOAA)",          "1,700 storms\n1851–2023\nTrack + intensity"),
    ("OpenFEMA\nPublic Assistance API", "240K PA projects\n1990-present\nDisaster USD by county"),
    ("NOAA NCEI\nStorm Events DB",      "9,258 event-county rows\n1996–2023\nDamage + casualties"),
    ("CDC Social\nVulnerability Index", "3,143 counties\n2000–2022\n16 vulnerability flags"),
    ("U.S. Census\nACS 5-Year",         "35,426 county-years\n2012–2022\nDemographics + housing"),
]
src_w, src_h = 3.8, 1.55
src_xs = [0.3, 4.5, 8.7, 12.9, 17.1]
src_y  = 24.7
for (lbl, sub), sx in zip(sources, src_xs):
    box(ax, sx, src_y, src_w, src_h, lbl, sub,
        color=C["source"], fontsize=8.2)

# ─────────────────────────────────────────────────────────────────────────────
# ROW 2 – Ingestion Scripts  (y ≈ 22.8)
# ─────────────────────────────────────────────────────────────────────────────
divider(ax, 24.5)
section_label(ax, 0.2, 24.2, "LAYER 2 – INGESTION SCRIPTS  (src/01 – 05)")

scripts_top = [
    ("01_ingest_hurdat2.py",  "Parse fixed-width\ntrack file"),
    ("02_ingest_fema.py",     "Paginated REST API\n+ PA batch fetch"),
    ("03_ingest_ncei.py",     "Download 28 annual\nCSV.gz files"),
    ("04_ingest_svi.py",      "Direct CSV\nfrom CDC portal"),
    ("05_ingest_acs.py",      "Census API\n11 vintage years"),
]
scr_w, scr_h = 3.8, 1.35
scr_y = 22.8
for (lbl, sub), sx in zip(scripts_top, src_xs):
    box(ax, sx, scr_y, scr_w, scr_h, lbl, sub,
        color=C["script"], fontsize=7.8)
    # Arrow from source → script
    arrow(ax, sx + scr_w/2, src_y, sx + scr_w/2, scr_y + scr_h)

# ─────────────────────────────────────────────────────────────────────────────
# ROW 3 – Raw Parquet Files  (y ≈ 21.1)
# ─────────────────────────────────────────────────────────────────────────────
divider(ax, 22.6)
section_label(ax, 0.2, 22.3, "LAYER 3 – RAW PARQUET FILES  (data/raw/)")

raw_files = [
    ("tracks.parquet",             "594 storms\n11,857 waypoints"),
    ("declarations.parquet\npa_projects.parquet", "~500 disasters\n240K PA rows"),
    ("storm_events.parquet",       "9,258 rows\n1,480 unique FIPS"),
    ("svi_2020.parquet",           "3,143 counties\n24 SVI columns"),
    ("acs_county.parquet",         "35,426 county-yr rows\n8 ACS variables"),
]
raw_w, raw_h = 3.8, 1.25
raw_y = 20.9
for (lbl, sub), sx in zip(raw_files, src_xs):
    box(ax, sx, raw_y, raw_w, raw_h, lbl, sub,
        color=C["data"], text_color=C["text_dk"], fontsize=7.8)
    arrow(ax, sx + raw_w/2, scr_y, sx + raw_w/2, raw_y + raw_h)

# ─────────────────────────────────────────────────────────────────────────────
# ROW 4 – Panel Construction  (y ≈ 18.7)
# ─────────────────────────────────────────────────────────────────────────────
divider(ax, 20.7)
section_label(ax, 0.2, 20.4, "LAYER 4 – PANEL CONSTRUCTION  (src/06_build_panel.py)")

# Convergence arrows from all 5 raw files → panel script
for sx in src_xs:
    arrow(ax, sx + raw_w/2, raw_y, 11.0, 19.7, color="#999999", lw=1.2)

box(ax, 4.5, 19.15, 13.0, 1.2,
    "06_build_panel.py",
    "Haversine distance matching (250 km radius)  |  FIPS joins  |  ACS vintage alignment  |  PA label construction ($1k/capita = 1 supply unit)",
    color=C["script"], fontsize=8.5)

box(ax, 4.5, 17.6, 13.0, 1.35,
    "panel.parquet",
    "31,045 storm-county pairs  |  233 storms  |  1,535 unique counties  |  Label: demand_proxy  |  49.5% FEMA-declared  |  20.9% label > 0",
    color=C["data"], text_color=C["text_dk"], fontsize=8.5)
arrow(ax, 11.0, 19.15, 11.0, 17.6 + 1.35)

# ─────────────────────────────────────────────────────────────────────────────
# ROW 5 – Feature Engineering  (y ≈ 16.3)
# ─────────────────────────────────────────────────────────────────────────────
divider(ax, 17.45)
section_label(ax, 0.2, 17.2, "LAYER 5 – FEATURE ENGINEERING  (src/07_features.py)")

arrow(ax, 11.0, 17.6, 11.0, 16.65)
box(ax, 3.0, 16.1, 16.0, 1.25,
    "07_features.py  →  model_ready.parquet",
    "18 features: peak_wind_kt | min_dist_km | log_total_pop | log_housing_units | svi_overall | EP_NOVEH | EP_AGE65 | EP_POV150 | EP_MOBILE | EP_CROWD | EP_DISABL | EP_UNINSUR | log_no_vehicle | log_pop_65plus | median_hh_income_10k | median_year_built | prior_hits  +  log transformations",
    color=C["script"], fontsize=8.0)

# ─────────────────────────────────────────────────────────────────────────────
# ROW 6 – Modeling  (y ≈ 13.5)
# ─────────────────────────────────────────────────────────────────────────────
divider(ax, 15.9)
section_label(ax, 0.2, 15.6, "LAYER 6 – PREDICTION MODEL  (src/08_train_model.py)")

arrow(ax, 11.0, 16.1, 11.0, 15.2)

# 08 script box
box(ax, 4.0, 14.6, 14.0, 0.85,
    "08_train_model.py  |  Temporal holdout: train < 2018, test >= 2018  (22,503 train / 8,542 test rows)",
    color=C["script"], fontsize=8.5)
arrow(ax, 11.0, 14.6, 11.0, 14.05)

# Three model output boxes
m_xs = [0.5, 7.5, 14.5]
m_labels = [
    ("XGBoost Point Model\n(xgb_point.json)",
     "n_est=1000 | lr=0.05 | depth=6\nsubsample=0.8 | early_stop=50\nRMSE: 0.7707  R²: -0.001"),
    ("Quantile Models\n(quantile_models.pkl)",
     "tau = 0.10 / 0.50 / 0.90\nPinball loss: 0.006 / 0.031 / 0.073\nUsed for robust allocation"),
    ("SHAP Explanations\n(shap_bar.png / shap_beeswarm.png)",
     "TreeExplainer on 2,000 test rows\nTop drivers: peak_wind_kt,\nmin_dist_km, svi_overall"),
]
for (lbl, sub), mx in zip(m_labels, m_xs):
    box(ax, mx, 12.6, 6.8, 1.35, lbl, sub,
        color=C["model"], fontsize=7.8)
    arrow(ax, mx + 3.4, 14.6, mx + 3.4, 12.6 + 1.35, color="#999999", lw=1.2)

box(ax, 4.5, 11.35, 13.0, 1.05,
    "predictions.parquet",
    "8,542 test rows  |  Columns: pred_demand | pred_q10 | pred_q50 | pred_q90 | storm_id | fips | demand_proxy (actual)",
    color=C["data"], text_color=C["text_dk"], fontsize=8.5)
for mx in m_xs:
    arrow(ax, mx + 3.4, 12.6, 11.0, 11.35 + 1.05, color="#aaaaaa", lw=1.1)

# ─────────────────────────────────────────────────────────────────────────────
# ROW 7 – Optimization  (y ≈ 9.5)
# ─────────────────────────────────────────────────────────────────────────────
divider(ax, 11.2)
section_label(ax, 0.2, 10.9, "LAYER 7 – LP OPTIMIZATION  (src/09_optimize.py)")

arrow(ax, 11.0, 11.35, 11.0, 10.55)

box(ax, 3.5, 10.0, 15.0, 0.85,
    "09_optimize.py  |  Supply = 80% total predicted demand  |  46 test storms  |  PuLP / CBC solver",
    color=C["script"], fontsize=8.5)
arrow(ax, 11.0, 10.0, 11.0, 9.45)

# LP formulation box
box(ax, 0.4, 8.2, 8.0, 1.1,
    "LP Formulation",
    "min sum(w_i * unmet_i)  s.t.\n  unmet_i >= demand_i - alloc_i\n  sum(alloc) <= S  |  equity: sum(high_SVI) >= 0.4*S",
    color="#4a235a", fontsize=7.8)
arrow(ax, 11.0, 10.0, 4.4, 8.2 + 1.1, color="#999999", lw=1.1)

# 3 policy boxes
pol_xs  = [0.4, 7.6, 14.8]
pol_data = [
    ("Policy 1\nPopulation-Proportional",
     "Baseline: allocate proportional\nto county population\nUnmet: 52.7%  |  High-SVI cov: 50.6%"),
    ("Policy 2\nSVI-Weighted",
     "Allocate proportional to\nSVI vulnerability score\nUnmet: 22.7%  |  High-SVI cov: 94.9%"),
    ("Policy 3\nModel + LP + Equity",
     "Predicted demand + equity\nconstraint (beta=0.40)\nUnmet: 27.7%  |  High-SVI cov: 97.4%"),
]
pol_colors = ["#5d6d7e", "#1a5276", C["optim"]]
for (lbl, sub), px, pc in zip(pol_data, pol_xs, pol_colors):
    box(ax, px, 7.0, 6.6, 1.3, lbl, sub, color=pc, fontsize=7.8)
    arrow(ax, px + 3.3, 10.0, px + 3.3, 7.0 + 1.3, color="#aaaaaa", lw=1.1)

box(ax, 3.5, 5.85, 15.0, 0.95,
    "all_storms_comparison.parquet",
    "138 rows (46 storms × 3 policies)  |  Metrics: pct_unmet | coverage_high_svi | coverage_low_svi | equity_gap | CVaR shocks",
    color=C["data"], text_color=C["text_dk"], fontsize=8.5)
for px in pol_xs:
    arrow(ax, px + 3.3, 7.0, 11.0, 5.85 + 0.95, color="#aaaaaa", lw=1.1)

# ─────────────────────────────────────────────────────────────────────────────
# ROW 8 – Evaluation & Outputs  (y ≈ 3.5)
# ─────────────────────────────────────────────────────────────────────────────
divider(ax, 5.65)
section_label(ax, 0.2, 5.4, "LAYER 8 – EVALUATION & OUTPUTS  (src/10_evaluate.py)")

arrow(ax, 11.0, 5.85, 11.0, 5.1)

box(ax, 4.0, 4.55, 14.0, 0.85,
    "10_evaluate.py  |  Gini coefficient  |  CVaR robustness  |  Per-storm aggregation",
    color=C["script"], fontsize=8.5)
arrow(ax, 11.0, 4.55, 11.0, 3.95)

out_xs = [0.3, 4.9, 9.5, 14.1, 17.7]
out_labels = [
    ("policy_comparison.png",    "3-panel bar chart\nUnmet / High-SVI / Equity"),
    ("robustness.png",           "LP vs +20% demand shock\nper storm year"),
    ("demand_vs_svi.png",        "Scatter: predicted demand\nvs SVI score"),
    ("map_LAURA_2020.png",       "County map\ndemand + SVI heatmap"),
    ("shap_bar /\nbeeswarm.png", "Top 15 features\nSHAP importance"),
]
out_w = 3.5
for (lbl, sub), ox in zip(out_labels, out_xs):
    box(ax, ox, 2.85, out_w, 1.0, lbl, sub,
        color=C["output"], fontsize=7.5)
    arrow(ax, ox + out_w/2, 4.55, ox + out_w/2, 2.85 + 1.0, color="#aaaaaa", lw=1.1)

# ─────────────────────────────────────────────────────────────────────────────
# KEY RESULTS BANNER
# ─────────────────────────────────────────────────────────────────────────────
results_box = FancyBboxPatch((0.3, 0.35), 21.4, 2.3,
                              boxstyle="round,pad=0,rounding_size=0.3",
                              linewidth=2, edgecolor=C["optim"],
                              facecolor="#fef9e7", zorder=3)
ax.add_patch(results_box)
ax.text(11, 2.3, "KEY RESULTS",
        ha="center", va="center", fontsize=11, fontweight="bold", color=C["optim"])
results_text = (
    "Model + LP + Equity Policy  vs.  Population-Proportional Baseline:\n"
    "  47.5% reduction in total unmet demand     |     +46.8 pp improvement in high-SVI coverage     |     Equity gap: +52.1 pp\n"
    "Robustness: policy holds under +20% demand shock  |  SHAP top features: peak_wind_kt, min_dist_km, svi_overall"
)
ax.text(11, 1.2, results_text,
        ha="center", va="center", fontsize=9, color="#1a1a1a",
        linespacing=1.6)

# ─────────────────────────────────────────────────────────────────────────────
# LEGEND
# ─────────────────────────────────────────────────────────────────────────────
legend_items = [
    (C["source"], "External Data Source"),
    (C["script"], "Python Script"),
    (C["data"],   "Stored Data (Parquet)"),
    (C["model"],  "ML Model Component"),
    (C["optim"],  "Optimization Layer"),
    (C["output"], "Output Figure"),
]
lx, ly = 15.6, 27.6
for i, (clr, lbl) in enumerate(legend_items):
    rect = FancyBboxPatch((lx + (i % 3) * 2.1, ly - (i // 3) * 0.45),
                          0.35, 0.28,
                          boxstyle="round,pad=0,rounding_size=0.05",
                          facecolor=clr, edgecolor="white", lw=0.8, zorder=5)
    ax.add_patch(rect)
    ax.text(lx + (i % 3) * 2.1 + 0.45, ly - (i // 3) * 0.45 + 0.14,
            lbl, va="center", fontsize=7.2, color="#333333", zorder=5)

plt.tight_layout(pad=0.3)
plt.savefig(OUT, dpi=180, bbox_inches="tight",
            facecolor=C["bg"], edgecolor="none")
plt.close()
print(f"[OK] Saved {OUT}")
