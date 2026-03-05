"""
================================================================================
  HURRICANE RESOURCE ALLOCATION — Complete Pipeline Walkthrough
================================================================================
  Run this script to walk through everything we built, step by step.

  python pipeline_walkthrough.py

  What it does:
    STEP 1 — Shows the 5 raw data sources and their sizes
    STEP 2 — Panel construction: how we joined them (31,045 storm-county pairs)
    STEP 3 — Feature engineering (18 features + log transforms)
    STEP 4 — XGBoost model training & the zero-inflation problem
    STEP 5 — LP optimisation: 3 allocation policies vs 46 test storms
    STEP 6 — Key equity results
    STEP 7 — Saves a 6-panel "story" figure to outputs/figures/walkthrough_summary.png

  No GPU or internet needed — reads from data/processed/ and outputs/
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

# ── Paths ──────────────────────────────────────────────────────────────────────
PROC  = Path("data/processed")
RAW   = Path("data/raw")
ALLOC = Path("outputs/allocations")
FIGS  = Path("outputs/figures")
FIGS.mkdir(parents=True, exist_ok=True)

# ── Terminal colours (ANSI) ────────────────────────────────────────────────────
class C:
    HEADER  = "\033[95m"
    BLUE    = "\033[94m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    BOLD    = "\033[1m"
    END     = "\033[0m"
    CYAN    = "\033[96m"

def banner(title, width=72, char="="):
    print(f"\n{C.BOLD}{C.CYAN}{char*width}")
    print(f"  {title}")
    print(f"{char*width}{C.END}")

def section(title):
    print(f"\n{C.BOLD}{C.YELLOW}  -- {title}{C.END}")

def ok(msg):
    print(f"  {C.GREEN}[OK]{C.END} {msg}")

def info(msg):
    print(f"  {C.BLUE}    {msg}{C.END}")

def kpi(label, value, note=""):
    note_str = f"  ({note})" if note else ""
    print(f"  {C.BOLD}{label:<38}{C.END} {C.GREEN}{value}{C.END}{C.BLUE}{note_str}{C.END}")


# ════════════════════════════════════════════════════════════════════════════════
banner("HURRICANE RESOURCE ALLOCATION — Project Walkthrough")
print("""
  PROBLEM:  When a hurricane hits, FEMA needs to allocate limited supplies
            (water, food, generators) across hundreds of affected counties.
            Vulnerable communities are historically under-served.

  SOLUTION: Build a machine-learning pipeline that:
            1. Predicts demand in each county using hazard + vulnerability data
            2. Solves a Linear Program (LP) to allocate supply fairly
            3. Guarantees that high-SVI (most vulnerable) counties get >= 40%
               of total supply

  RESULT:   47.5% less unmet demand vs population-proportional baseline
            +46.8pp coverage for the most vulnerable counties
""")


# ════════════════════════════════════════════════════════════════════════════════
banner("STEP 1 — Data Sources  (5 datasets ingested by scripts 01-05)")
# ════════════════════════════════════════════════════════════════════════════════

data_sources = [
    ("HURDAT2 (NHC/NOAA)",    "data/raw/hurdat2/tracks.parquet",
     "Hurricane tracks 1851–2023. Gives us peak wind speed and distance\n"
     "     to each county for every storm. The core hazard signal."),
    ("OpenFEMA PA API",        "data/raw/fema/declarations.parquet",
     "FEMA Public Assistance dollar amounts. We use this as our\n"
     "     demand label: PA disbursements / county population / 1000."),
    ("NOAA NCEI Storm Events", "data/raw/ncei/storm_events.parquet",
     "Reported damage and casualties per event. Used as a cross-check\n"
     "     on the FEMA label — not a primary model feature."),
    ("CDC Social Vulnerability Index (SVI)", "data/raw/svi/svi_2020.parquet",
     "16 vulnerability indicators per county (poverty, no-vehicle,\n"
     "     mobile homes, disability, etc.). Core equity signal."),
    ("U.S. Census ACS 5-Year", "data/raw/acs/acs_county.parquet",
     "Population, housing units, median income, age 65+ per county\n"
     "     per year. Gives us exposure size and demographic context."),
]

for name, path, desc in data_sources:
    p = Path(path)
    if p.exists():
        try:
            df_tmp = pd.read_parquet(p)
            ok(f"{name:<36}  {len(df_tmp):>7,} rows  x  {df_tmp.shape[1]:>3} cols")
        except Exception:
            ok(f"{name:<36}  (loaded)")
    else:
        print(f"  {C.YELLOW}[--]{C.END} {name:<36}  file not found")
    info(desc)

print()
info("All raw data stored as Parquet files in data/raw/")
info("Ingestion scripts: src/01_ingest_hurdat2.py through src/05_ingest_acs.py")


# ════════════════════════════════════════════════════════════════════════════════
banner("STEP 2 — Panel Construction  (src/06_build_panel.py)")
# ════════════════════════════════════════════════════════════════════════════════

print("""
  We need one row per (storm, county) pair. Here is how we built it:

    a) HURDAT2 gives us storm tracks (lat/lon/wind at each 6-hour timestep).
       For each storm we find the peak wind at the track point closest to
       each county centroid.

    b) We link storms to counties using HAVERSINE distance:
       If any track point comes within 250 km of a county, that county
       is "in scope" for that storm.

    c) FEMA PA projects are matched to storms by disaster declaration date
       and county FIPS code.  Label = total PA dollars / population / 1000
       (supply units per capita).

    d) SVI and ACS data are joined by FIPS + year (vintage-matched).
""")

panel_path = PROC / "panel.parquet"
if panel_path.exists():
    panel = pd.read_parquet(panel_path)
    section("Panel statistics")
    kpi("Total storm-county pairs",   f"{len(panel):,}")
    kpi("Unique storms",              f"{panel['storm_id'].nunique():,}")
    kpi("Unique counties (FIPS)",     f"{panel['fips'].nunique():,}")
    kpi("Year range",                 f"{panel['storm_year'].min()} – {panel['storm_year'].max()}")
    kpi("Zero-demand rows (79.1%)",   f"{(panel['demand_proxy']==0).sum():,}",
        "counties with zero FEMA PA — the zero-inflation problem")
    kpi("Non-zero demand rows",       f"{(panel['demand_proxy']>0).sum():,}")
    kpi("FEMA-declared rows",         f"{panel['fema_declared'].sum():,}")
    kpi("Max demand_proxy value",     f"{panel['demand_proxy'].max():.3f}",
        "Hurricane Katrina peak county")
else:
    print("  panel.parquet not found — skipping stats")
    panel = None


# ════════════════════════════════════════════════════════════════════════════════
banner("STEP 3 — Feature Engineering  (src/07_features.py)")
# ════════════════════════════════════════════════════════════════════════════════

print("""
  We selected 18 features in 3 groups:

  HAZARD (what the storm did)
    peak_wind_kt      — maximum sustained wind speed (knots) at closest point
    min_dist_km       — closest track distance to county centroid (km)

  POPULATION EXPOSURE (how many people were exposed)
    log_total_pop           — log(total county population)
    log_total_housing_units — log(housing units)
    log_no_vehicle          — log(households with no vehicle)
    log_pop_65plus          — log(population aged 65+)
    prior_hits              — cumulative number of prior storms this county has had

  VULNERABILITY (CDC SVI + Census)
    svi_overall       — composite SVI score (0=least, 1=most vulnerable)
    EP_NOVEH          — % households with no vehicle
    EP_AGE65          — % population age 65+
    EP_POV150         — % below 150% poverty line
    EP_MOBILE         — % in mobile homes
    EP_CROWD          — % in crowded housing
    EP_DISABL         — % with a disability
    EP_UNINSUR        — % uninsured
    median_hh_income_10k — median household income / 10,000
""")

model_ready_path = PROC / "model_ready.parquet"
if model_ready_path.exists():
    mr = pd.read_parquet(model_ready_path)
    section("model_ready.parquet stats")
    kpi("Rows", f"{len(mr):,}")
    kpi("Feature columns", f"{len([c for c in mr.columns if c not in ['demand_proxy','storm_year','fips','storm_id','storm_name','fema_declared','feature_list']])}")
    kpi("Train rows (year < 2018)", f"{(mr['storm_year'] < 2018).sum():,}")
    kpi("Test rows  (year >= 2018)", f"{(mr['storm_year'] >= 2018).sum():,}")
else:
    print("  model_ready.parquet not found")


# ════════════════════════════════════════════════════════════════════════════════
banner("STEP 4 — Model Training  (src/08_train_model.py)")
# ════════════════════════════════════════════════════════════════════════════════

print("""
  MODEL:  XGBoost Regressor (point estimate) + 3 quantile models
          Temporal holdout split: train on years < 2018, test on >= 2018
          This mimics real deployment — train on historical storms,
          predict on future ones.

  PARAMETERS (point model):
    n_estimators=1000, learning_rate=0.05, max_depth=6
    subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=50
    eval_metric=rmse

  QUANTILE MODELS (for uncertainty):
    tau = 0.10 / 0.50 / 0.90  using  objective="reg:quantileerror"
    This gives us a prediction interval:  pred_q10 <= pred_q50 <= pred_q90
""")

preds_path = PROC / "predictions.parquet"
if preds_path.exists():
    preds = pd.read_parquet(preds_path)
    from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
    y_true = preds["demand_proxy"]
    y_pred = preds["pred_demand"]
    naive  = np.full(len(y_true), y_true.mean())

    rmse_model = root_mean_squared_error(y_true, y_pred)
    rmse_naive = root_mean_squared_error(y_true, naive)
    mae_model  = mean_absolute_error(y_true, y_pred)
    r2         = r2_score(y_true, y_pred)

    section("Model performance on 8,542 test rows")
    kpi("R²  (test set)",            f"{r2:.4f}",   "ideally close to 1.0")
    kpi("RMSE  (model)",             f"{rmse_model:.4f}")
    kpi("RMSE  (naive mean baseline)",f"{rmse_naive:.4f}",
        "model barely beats predicting the mean for everyone")
    kpi("MAE   (model)",             f"{mae_model:.4f}")
    kpi("Pred range",
        f"[{y_pred.min():.3f}, {y_pred.max():.3f}]",
        "almost all near the mean — zero-inflation problem")
    kpi("CV actual  (std/mean)",     f"{y_true.std()/y_true.mean():.2f}",
        "high variation in truth")
    kpi("CV predicted (std/mean)",   f"{y_pred.std()/y_pred.mean():.2f}",
        "model collapses variation to near-mean")

    print(f"""
  {C.YELLOW}WHY R² IS NEAR ZERO (the zero-inflation problem):{C.END}

    79.1% of rows have demand = 0  (counties with zero FEMA PA).
    XGBoost minimises RMSE, and predicting the mean for everyone
    is near-optimal under RMSE when 4 out of 5 labels are zero.

    The model "cheats" by ignoring county differences and predicting
    the global mean (~0.07) for nearly everyone.

    Key finding: our 47.5% optimisation improvement comes from the
    EQUITY CONSTRAINT in the LP, NOT from model intelligence.
    The model is effectively acting as a uniform demand estimate.
""")
else:
    preds = None
    print("  predictions.parquet not found")


# ════════════════════════════════════════════════════════════════════════════════
banner("STEP 5 — LP Optimisation  (src/09_optimize.py)")
# ════════════════════════════════════════════════════════════════════════════════

print("""
  For each of the 46 test storms we solve a Linear Program (LP):

  SETUP:
    Total supply  S = 80% of predicted total demand
    (We assume a mild supply shortage — real emergencies always have one)

  LP FORMULATION:
    minimise   sum( w_i * unmet_i )      for all counties i
    subject to:
      unmet_i  >= demand_i - alloc_i     (define unmet demand)
      unmet_i  >= 0
      alloc_i  >= 0
      sum(alloc_i)  <= S                 (can't exceed supply)
      sum(alloc_i for i in high_SVI)  >= 0.40 * S   (EQUITY CONSTRAINT)

  WEIGHTS:
    w_i = 2.0  if county SVI >= 0.75  (top quartile = "high vulnerability")
    w_i = 1.0  otherwise

  THREE POLICIES compared on every storm:
    1. Population-Proportional   — alloc proportional to county population
    2. SVI-Weighted              — alloc proportional to SVI score
    3. Model + LP + Equity       — our method (predicted demand + LP + equity)

  Solver: PuLP / CBC (open-source, runs in seconds per storm)
""")

results_path = ALLOC / "all_storms_comparison.parquet"
if results_path.exists():
    results = pd.read_parquet(results_path)
    avg = results.groupby("policy")[[
        "pct_unmet", "coverage_overall",
        "coverage_high_svi", "coverage_low_svi", "equity_gap"
    ]].mean()

    section("Average results across 46 test storms")
    print(f"\n  {'Policy':<28} {'Unmet %':>9} {'Coverage':>10} "
          f"{'HiSVI Cov':>11} {'LoSVI Cov':>11} {'Equity Gap':>11}")
    print(f"  {'-'*80}")

    LABELS = {
        "population_proportional": "Population-Proportional",
        "svi_weighted":            "SVI-Weighted",
        "model_lp_equity":         "Model + LP + Equity",
    }
    for pol in ["population_proportional", "svi_weighted", "model_lp_equity"]:
        if pol in avg.index:
            r = avg.loc[pol]
            highlight = C.GREEN + C.BOLD if pol == "model_lp_equity" else ""
            end       = C.END            if pol == "model_lp_equity" else ""
            print(f"  {highlight}{LABELS[pol]:<28}  {r['pct_unmet']:>8.1%}  "
                  f"{r['coverage_overall']:>9.1%}  "
                  f"{r['coverage_high_svi']:>10.1%}  "
                  f"{r['coverage_low_svi']:>10.1%}  "
                  f"{r['equity_gap']:>+10.1%}{end}")

    lp  = avg.loc["model_lp_equity"]
    pop = avg.loc["population_proportional"]
    print()
    section("Key wins vs Population-Proportional baseline")
    pct_imp = (pop["pct_unmet"] - lp["pct_unmet"]) / pop["pct_unmet"]
    cov_imp = lp["coverage_high_svi"] - pop["coverage_high_svi"]
    eq_imp  = lp["equity_gap"]        - pop["equity_gap"]
    kpi("Reduction in unmet demand",      f"{pct_imp:.1%}")
    kpi("High-SVI coverage improvement",  f"+{cov_imp:.1%} pp")
    kpi("Equity gap improvement",         f"+{eq_imp:.1%} pp")
    kpi("Storms where LP beats baseline", f"{(results[results['policy']=='model_lp_equity']['pct_unmet'].values < results[results['policy']=='population_proportional']['pct_unmet'].values).sum()} / 46")

    section("Robustness check (demand shock)")
    lp_df = results[results["policy"] == "model_lp_equity"]
    shock_hi = (lp_df["unmet_shock_plus20"]  / (lp_df["total_demand"] * 1.20)).mean()
    shock_lo = (lp_df["unmet_shock_minus20"] / (lp_df["total_demand"] * 0.80)).mean()
    kpi("LP unmet demand at nominal",  f"{lp['pct_unmet']:.1%}")
    kpi("LP unmet demand at +20% demand shock", f"{shock_hi:.1%}",
        "supply fixed, demand 20% higher")
    kpi("LP unmet demand at -20% demand shock", f"{shock_lo:.1%}",
        "supply fixed, demand 20% lower")
else:
    results = None
    print("  all_storms_comparison.parquet not found")


# ════════════════════════════════════════════════════════════════════════════════
banner("STEP 6 — Equity Analysis")
# ════════════════════════════════════════════════════════════════════════════════

print("""
  Social Vulnerability Index (SVI) is a CDC measure combining:
    poverty, housing crowding, disability, no vehicle, mobile homes,
    uninsured rate, minority status, and English proficiency.

  We split counties into two groups per storm:
    HIGH-SVI:  SVI >= 0.75  (top quartile — most vulnerable)
    LOW-SVI :  SVI <  0.75  (the rest)

  EQUITY GAP = Coverage(high-SVI) - Coverage(low-SVI)
    Positive = high-SVI counties are better served (good!)
    Negative = high-SVI counties are under-served (bad, status quo)
""")

if results is not None:
    section("Coverage breakdown by vulnerability group")
    for pol in ["population_proportional", "svi_weighted", "model_lp_equity"]:
        r   = avg.loc[pol]
        gap = r["equity_gap"]
        h   = C.GREEN + C.BOLD if pol == "model_lp_equity" else ""
        e   = C.END            if pol == "model_lp_equity" else ""
        col = C.GREEN if gap > 0 else C.RED
        print(f"  {h}{LABELS[pol]:<28}{e}  "
              f"Hi-SVI: {r['coverage_high_svi']:.1%}  "
              f"Lo-SVI: {r['coverage_low_svi']:.1%}  "
              f"Gap: {col}{gap:+.1%}{C.END}")

    print(f"""
  {C.BOLD}Interpretation:{C.END}
    Population-proportional gives high-SVI counties LESS coverage
    than low-SVI counties (equity gap = -2.4%) — the current status quo.

    Our LP with equity constraint flips this: high-SVI counties
    get 97.4% coverage vs 45.3% for low-SVI (equity gap = +52.1pp).
    The constraint forces supply to flow to the most vulnerable.
""")


# ════════════════════════════════════════════════════════════════════════════════
banner("STEP 7 — Generating Summary Figure")
# ════════════════════════════════════════════════════════════════════════════════

print("  Building 6-panel walkthrough summary figure ...")

PC = {
    "population_proportional": "#94a3b8",
    "svi_weighted":            "#f59e0b",
    "model_lp_equity":         "#10b981",
}
PL = {
    "population_proportional": "Pop-Proportional",
    "svi_weighted":            "SVI-Weighted",
    "model_lp_equity":         "Model + LP + Equity",
}
POLICIES = list(PC.keys())
BG    = "#fafafa"; DARK  = "#0f172a"; MID   = "#64748b"
LIGHT = "#cbd5e1"; GRID  = "#e2e8f0"; WHITE = "#ffffff"

plt.rcParams.update({
    "figure.facecolor": BG,  "axes.facecolor":    WHITE,
    "axes.edgecolor":   LIGHT,"axes.spines.top":   False,
    "axes.spines.right":False, "axes.grid":         True,
    "grid.color":       GRID,  "grid.linewidth":    0.6,
    "font.family":      "sans-serif", "font.size": 10,
    "axes.titlesize":   12,    "axes.titleweight":  "bold",
    "axes.titlepad":    10,    "axes.labelsize":    10,
    "axes.labelcolor":  MID,   "xtick.color":       MID,
    "ytick.color":      MID,   "legend.frameon":    False,
    "savefig.facecolor":BG,    "savefig.dpi":       180,
    "savefig.bbox":     "tight",
})

fig = plt.figure(figsize=(20, 14), facecolor=BG)
fig.suptitle(
    "Hurricane Resource Allocation  |  End-to-End Project Summary",
    fontsize=20, fontweight="bold", color=DARK, y=0.98
)

gs = gridspec.GridSpec(
    3, 3, figure=fig,
    top=0.93, bottom=0.06,
    left=0.06, right=0.97,
    hspace=0.52, wspace=0.38,
)

# ── Panel 0,0 — Pipeline flow diagram ─────────────────────────────────────────
ax00 = fig.add_subplot(gs[0, 0])
ax00.set_xlim(0, 10); ax00.set_ylim(0, 10); ax00.axis("off")
ax00.set_facecolor(WHITE)
ax00.set_title("The Pipeline (10 scripts)", pad=10)

steps_flow = [
    ("5 Raw Data Sources",              "#1a5276", 8.8),
    ("06  Build Panel  (31K rows)",     "#1e8449", 7.2),
    ("07  Feature Engineering (18 feat)","#1e8449", 5.6),
    ("08  XGBoost Model  (R\u00b2=-0.001)","#7d3c98", 4.0),
    ("09  LP Optimisation  (46 storms)","#b7770d", 2.4),
    ("10  Evaluate  (3 policies)",      "#922b21", 0.8),
]
for label, fc, y in steps_flow:
    ax00.add_patch(FancyBboxPatch(
        (0.3, y - 0.55), 9.4, 1.05,
        boxstyle="round,pad=0,rounding_size=0.2",
        facecolor=fc, edgecolor="white", lw=1.2, zorder=3
    ))
    ax00.text(5, y + 0.0, label,
              ha="center", va="center", color="white",
              fontsize=9.5, fontweight="bold", zorder=4)
    if y > 0.8:
        ax00.annotate("", xy=(5, y - 0.55), xytext=(5, y - 0.75),
            arrowprops=dict(arrowstyle="-|>", color="#888888", lw=1.8, mutation_scale=12),
            zorder=5)

# ── Panel 0,1 — Zero inflation pie ────────────────────────────────────────────
ax01 = fig.add_subplot(gs[0, 1])
ax01.set_title("The Core Problem:\nZero-Inflated Demand Labels")
if panel is not None:
    zf  = (panel["demand_proxy"] == 0).mean()
    ax01.pie(
        [zf * 100, (1 - zf) * 100],
        labels=[f"Zero demand\n{zf*100:.1f}%", f"Non-zero\n{(1-zf)*100:.1f}%"],
        colors=["#94a3b8", "#3b82f6"],
        explode=(0.03, 0.08),
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 9},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        pctdistance=0.72,
    )
    ax01.text(0, -1.45,
              "XGBoost predicts near-mean for everyone\nbecause 4 in 5 labels are zero",
              ha="center", va="center", fontsize=8.5, color=MID,
              fontstyle="italic", transform=ax01.transAxes,
              bbox=dict(boxstyle="round,pad=0.3", facecolor="#fef3c7",
                        edgecolor="#f59e0b", alpha=0.9))

# ── Panel 0,2 — SHAP feature importance ───────────────────────────────────────
ax02 = fig.add_subplot(gs[0, 2])
ax02.set_title("Top Features Driving Predictions\n(Mean |SHAP value|)")
try:
    import shap, xgboost as xgb, json as _json, pickle
    MODELS = Path("outputs/models")
    model_xgb = xgb.XGBRegressor()
    model_xgb.load_model(str(MODELS / "xgb_point.json"))
    feats     = _json.loads((MODELS / "features.json").read_text())
    if preds is not None:
        X_s     = preds[feats].fillna(0).sample(min(1200, len(preds)), random_state=42)
        exp     = shap.TreeExplainer(model_xgb)
        sv      = exp.shap_values(X_s)
        ms      = pd.Series(np.abs(sv).mean(axis=0), index=feats).sort_values().tail(12)
        rename  = {
            "peak_wind_kt": "Peak Wind (kt)",
            "min_dist_km":  "Min Distance (km)",
            "svi_overall":  "SVI Score",
            "log_total_pop":"log(Population)",
            "EP_NOVEH":     "% No Vehicle",
            "EP_AGE65":     "% Age 65+",
            "EP_POV150":    "% Poverty",
            "EP_MOBILE":    "% Mobile Homes",
            "EP_DISABL":    "% Disabled",
            "EP_UNINSUR":   "% Uninsured",
            "prior_hits":   "Prior Storm Hits",
            "median_hh_income_10k": "HH Income",
            "log_total_housing_units": "log(Housing)",
        }
        ms.index = [rename.get(f, f) for f in ms.index]
        colors_shap = ["#10b981" if v >= ms.quantile(0.75) else "#64748b" for v in ms.values]
        ax02.barh(ms.index, ms.values, color=colors_shap, linewidth=0, height=0.65)
        ax02.set_xlabel("Mean |SHAP|")
        ax02.spines["left"].set_visible(False)
except Exception as e:
    ax02.text(0.5, 0.5, f"SHAP unavailable\n{e}", ha="center", va="center",
              transform=ax02.transAxes, fontsize=9, color=MID)

# ── Panel 1,0 — Policy bar chart (% unmet) ────────────────────────────────────
ax10 = fig.add_subplot(gs[1, 0])
ax10.set_title("Average % Demand Unmet\n(lower is better)")
if results is not None:
    vals   = [avg.loc[p, "pct_unmet"] for p in POLICIES]
    colors = [PC[p] for p in POLICIES]
    bars   = ax10.bar(range(3), vals, color=colors, linewidth=0, width=0.5, zorder=3)
    for bar, v in zip(bars, vals):
        ax10.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + 0.005, f"{v:.1%}",
                  ha="center", va="bottom", fontsize=10, fontweight="bold")
    bars[np.argmin(vals)].set_edgecolor(DARK); bars[np.argmin(vals)].set_linewidth(2)
    ax10.set_xticks(range(3))
    ax10.set_xticklabels([PL[p] for p in POLICIES], fontsize=8.5)
    ax10.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax10.set_ylim(0, max(vals) * 1.25)
    ax10.spines["left"].set_visible(False)

# ── Panel 1,1 — Equity tradeoff scatter ───────────────────────────────────────
ax11 = fig.add_subplot(gs[1, 1])
ax11.set_title("Equity Tradeoff\nHigh-SVI vs Low-SVI Coverage")
if results is not None:
    for pol in POLICIES:
        sub = results[results["policy"] == pol]
        ax11.scatter(sub["coverage_low_svi"], sub["coverage_high_svi"],
                     color=PC[pol], s=40, alpha=0.65, zorder=3, linewidths=0)
        mx = sub["coverage_low_svi"].mean()
        my = sub["coverage_high_svi"].mean()
        ax11.scatter(mx, my, color=PC[pol], s=200, zorder=5,
                     edgecolors=DARK, linewidths=1.5, marker="D")
    ax11.plot([0, 1], [0, 1], "--", color=LIGHT, linewidth=1.5)
    ax11.set_xlabel("Low-SVI Coverage"); ax11.set_ylabel("High-SVI Coverage")
    ax11.set_xlim(0, 1.05); ax11.set_ylim(0, 1.05)
    ax11.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax11.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    handles = [Line2D([0],[0], marker="D", color="w", markerfacecolor=PC[p],
                       markersize=9, label=PL[p]) for p in POLICIES]
    ax11.legend(handles=handles, fontsize=8, loc="lower right")

# ── Panel 1,2 — Per-storm violin (% unmet) ────────────────────────────────────
ax12 = fig.add_subplot(gs[1, 2])
ax12.set_title("Distribution Across 46 Storms\n% Demand Unmet")
if results is not None:
    groups = [results.loc[results["policy"] == p, "pct_unmet"].values for p in POLICIES]
    parts  = ax12.violinplot(groups, positions=range(3), widths=0.5,
                              showmedians=False, showextrema=False)
    for body, pol in zip(parts["bodies"], POLICIES):
        body.set_facecolor(PC[pol]); body.set_alpha(0.35); body.set_edgecolor(PC[pol])
    bp = ax12.boxplot(groups, positions=range(3), widths=0.16,
                      patch_artist=True,
                      whiskerprops=dict(color=MID, linewidth=1.2),
                      capprops=dict(color=MID),
                      flierprops=dict(marker="o", markerfacecolor=MID, markersize=3),
                      medianprops=dict(color=DARK, linewidth=2.2))
    for patch, pol in zip(bp["boxes"], POLICIES):
        patch.set_facecolor(PC[pol]); patch.set_alpha(0.80)
    ax12.set_xticks(range(3))
    ax12.set_xticklabels([PL[p] for p in POLICIES], fontsize=8.5)
    ax12.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax12.spines["left"].set_visible(False)

# ── Panel 2,0-1 — Robustness timeline (wide) ──────────────────────────────────
ax20 = fig.add_subplot(gs[2, :2])
ax20.set_title("LP+Equity Policy: Per-Storm Unmet Demand with +/-20% Demand Shock Sensitivity")
if results is not None:
    lp_sorted = (results[results["policy"] == "model_lp_equity"]
                 .sort_values(["storm_year", "storm_name"])
                 .reset_index(drop=True))
    pop_sorted = (results[results["policy"] == "population_proportional"]
                  .sort_values(["storm_year", "storm_name"])
                  .reset_index(drop=True))
    n_s   = len(lp_sorted)
    xpos  = np.arange(n_s)
    sh_hi = (lp_sorted["unmet_shock_plus20"]  / (lp_sorted["total_demand"] * 1.20)).fillna(lp_sorted["pct_unmet"])
    sh_lo = (lp_sorted["unmet_shock_minus20"] / (lp_sorted["total_demand"] * 0.80)).fillna(lp_sorted["pct_unmet"])
    ax20.fill_between(xpos, sh_lo, sh_hi, alpha=0.15, color=PC["model_lp_equity"],
                      label="+/-20% demand shock range")
    ax20.plot(xpos, lp_sorted["pct_unmet"], "o-",
              color=PC["model_lp_equity"], lw=2, markersize=6,
              markeredgecolor=DARK, markeredgewidth=0.7,
              label="LP+Equity (baseline)", zorder=4)
    ax20.plot(xpos, pop_sorted["pct_unmet"], "s--",
              color=PC["population_proportional"], lw=1.5, markersize=4, alpha=0.6,
              label="Pop-Proportional (baseline)")
    yr_breaks = lp_sorted[lp_sorted["storm_year"] != lp_sorted["storm_year"].shift()].index.tolist()
    for b in yr_breaks[1:]:
        ax20.axvline(b - 0.5, color=LIGHT, lw=0.9, linestyle=":")
    ax20.set_ylabel("% Demand Unmet")
    ax20.set_xticks(xpos)
    ax20.set_xticklabels(
        [f"{r['storm_name'].title()}\n'{str(r['storm_year'])[2:]}"
         for _, r in lp_sorted.iterrows()],
        fontsize=6.5, rotation=40, ha="right"
    )
    ax20.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax20.legend(fontsize=9, loc="upper right")
    ax20.spines["left"].set_visible(False)

# ── Panel 2,2 — KPI summary card ──────────────────────────────────────────────
ax22 = fig.add_subplot(gs[2, 2])
ax22.set_xlim(0, 1); ax22.set_ylim(0, 1); ax22.axis("off")
ax22.set_facecolor(WHITE)

kpis_fig = [
    ("47.5%",   "less unmet demand\nvs pop-proportional",   "#10b981"),
    ("+46.8pp",  "high-SVI coverage\nimprovement",           "#10b981"),
    ("+52.1pp",  "equity gap\nimprovement",                  "#10b981"),
    ("Robust",   "holds under +20%\ndemand shock",           "#3b82f6"),
    ("3 policies","LP solves in < 2s\nper storm",            "#f59e0b"),
    ("46 storms","test set\n(2018-2023)",                    "#64748b"),
]
ax22.text(0.5, 0.97, "Key Results", ha="center", va="top",
          fontsize=13, fontweight="bold", color=DARK, transform=ax22.transAxes)
y_k = 0.86
for val, label, color in kpis_fig:
    ax22.text(0.15, y_k,      val,   ha="left", va="center", fontsize=13,
              fontweight="bold", color=color, transform=ax22.transAxes)
    ax22.text(0.15, y_k - 0.06, label, ha="left", va="center", fontsize=7.5,
              color=MID, transform=ax22.transAxes)
    y_k -= 0.155

out_path = FIGS / "walkthrough_summary.png"
plt.savefig(out_path)
plt.close()
ok(f"Summary figure saved to {out_path}")

# ════════════════════════════════════════════════════════════════════════════════
banner("ALL DONE  |  Files Generated")
# ════════════════════════════════════════════════════════════════════════════════

print(f"""
  {C.BOLD}outputs/figures/{C.END}
    walkthrough_summary.png     <- 6-panel summary (this run)
    fig1_policy_dashboard.png   <- KPI cards + bar charts
    fig2_per_storm_distribution.png
    fig3_equity_tradeoff.png
    fig4_robustness_timeline.png
    fig5_storm_ranking.png
    fig7_shap_importance.png
    chart2/  (8 subcharts: panels A-H)
    chart3/  (11 subcharts: architectures, code, pros/cons, metrics)

  {C.BOLD}Key scripts:{C.END}
    src/06_build_panel.py    <- joins all 5 data sources
    src/07_features.py       <- 18 features + log transforms
    src/08_train_model.py    <- XGBoost point + quantile models + SHAP
    src/09_optimize.py       <- LP solver (PuLP/CBC) x 46 storms x 3 policies
    src/10_evaluate_v2.py    <- all evaluation figures

  {C.BOLD}Next steps to improve the model:{C.END}
    1. Two-part hurdle model  (classifier x regressor)  — expected R² 0.25-0.45
    2. Filter to FEMA-declared only + log1p(target)     — expected R² 0.10-0.25
    3. Spatial block cross-validation + ensemble        — proves generalisability
""")
