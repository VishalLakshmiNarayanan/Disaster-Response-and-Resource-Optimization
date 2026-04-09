"""
15_spatial_cv.py
----------------
Evaluate model robustness using spatial block cross-validation.

Instead of the purely temporal holdout (train < 2018, test == 2018), this
script holds out entire geographic regions one at a time and trains on the
remaining regions (all years).  Four folds are defined by latitude bands:

  South     : lat < 30   (Gulf Coast, Florida)
  South-Mid : 30 <= lat < 33  (Deep South)
  Mid       : 33 <= lat < 36  (Mid-Atlantic South)
  North     : lat >= 36  (Northeast, Mid-Atlantic)

If lat/lon are unavailable the fips state prefix is used as a proxy.

Temporal holdout reference (from prior evaluation):
  RMSE ~0.770, R2 ~-0.0001, Stage-1 AUC ~0.65, Non-zero R2 ~0.022

Outputs:
  outputs/figures/spatial_cv.png  (14x8 in, dpi=180)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import os
import sys
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR         = r"C:\Users\shubb\Pytorch\Disaster-Response-and-Resource-Optimization"
MODEL_READY_PATH = os.path.join(BASE_DIR, "data", "processed", "model_ready.parquet")
PANEL_PATH       = os.path.join(BASE_DIR, "panel.parquet")
FIGURE_PATH      = os.path.join(BASE_DIR, "outputs", "figures", "spatial_cv.png")

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

FOLD_COLORS = {
    "South":     "#10b981",
    "South-Mid": "#f59e0b",
    "Mid":       "#6366f1",
    "North":     "#ec4899",
}
FOLD_ORDER = ["South", "South-Mid", "Mid", "North"]

# Temporal holdout reference metrics
TEMPORAL_RMSE = 0.770
TEMPORAL_R2   = -0.0001

TARGET  = "demand_proxy"
HOLDOUT_YEAR = 2018

# Non-numeric / metadata columns to exclude from features
META_COLS = {"demand_proxy", "storm_year", "fips", "storm_id",
             "storm_name", "fema_declared", "feature_list",
             "county_lat", "county_lon", "region"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_load(path, **kwargs):
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)
    return pd.read_parquet(path, **kwargs)


def assign_region_by_lat(lat):
    """Map a latitude value to one of four region strings."""
    if pd.isna(lat):
        return "North"
    if lat < 30:
        return "South"
    if lat < 33:
        return "South-Mid"
    if lat < 36:
        return "Mid"
    return "North"


# State FIPS prefixes for fallback region assignment
SOUTH_PREFIXES     = {"01", "12", "22", "28", "48"}   # AL FL LA MS TX
SOUTH_MID_PREFIXES = {"13", "45", "47"}               # GA SC TN
MID_PREFIXES       = {"37", "51"}                      # NC VA

def assign_region_by_fips(fips_str):
    """Map a 5-char FIPS string to a region using state-prefix lookup."""
    try:
        prefix = str(fips_str).zfill(5)[:2]
    except Exception:
        return "North"
    if prefix in SOUTH_PREFIXES:
        return "South"
    if prefix in SOUTH_MID_PREFIXES:
        return "South-Mid"
    if prefix in MID_PREFIXES:
        return "Mid"
    return "North"


def compute_metrics(y_true, y_pred):
    """Compute RMSE, MAE, R2 for regression predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    n = len(y_true)
    if n == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "n": 0}

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {"rmse": rmse, "mae": mae, "r2": r2, "n": n}


def compute_binary_metrics(y_true, y_pred, threshold=None):
    """Compute precision, recall, F1 for (demand > 0) classification."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if threshold is None:
        # Use median of positive predictions as threshold
        pos_preds = y_pred[y_pred > 0]
        threshold = float(np.median(pos_preds)) if len(pos_preds) > 0 else 0.0

    actual_pos  = (y_true > 0).astype(int)
    pred_pos    = (y_pred > threshold).astype(int)

    tp = int(np.sum((actual_pos == 1) & (pred_pos == 1)))
    fp = int(np.sum((actual_pos == 0) & (pred_pos == 1)))
    fn = int(np.sum((actual_pos == 1) & (pred_pos == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {"precision": precision, "recall": recall, "f1": f1,
            "threshold": threshold}


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("[1/4] Loading model_ready.parquet ...")
mr = safe_load(MODEL_READY_PATH)
print(f"      Loaded {len(mr):,} rows x {mr.shape[1]} cols")
print(f"      Columns: {list(mr.columns)}")

print("[2/4] Loading panel.parquet ...")
panel_exists = os.path.exists(PANEL_PATH)
if panel_exists:
    panel = safe_load(PANEL_PATH)
    print(f"      Panel loaded: {len(panel):,} rows, cols: {list(panel.columns)}")
else:
    print(f"      [WARN] panel.parquet not found at {PANEL_PATH}. Will use fips-prefix regions.")
    panel = pd.DataFrame()

# ---------------------------------------------------------------------------
# 2. Merge lat/lon and assign regions
# ---------------------------------------------------------------------------
print("[3/4] Assigning spatial regions ...")

use_lat = False

if panel_exists and len(panel) > 0:
    lat_col = "county_lat" if "county_lat" in panel.columns else None
    lon_col = "county_lon" if "county_lon" in panel.columns else None

    if lat_col and lon_col:
        # Deduplicate and merge
        panel_geo = (
            panel[["fips", lat_col, lon_col]]
            .drop_duplicates(subset=["fips"])
        )
        mr = mr.merge(panel_geo, on="fips", how="left")
        use_lat = mr["county_lat"].notna().sum() > 0
        print(f"      Merged lat/lon; {mr['county_lat'].notna().sum():,} rows have lat.")
    else:
        print("      [WARN] county_lat/county_lon not in panel. Using fips-prefix fallback.")
else:
    # Check if model_ready already has lat/lon
    if "county_lat" in mr.columns and mr["county_lat"].notna().sum() > 0:
        use_lat = True
        print("      county_lat already in model_ready.")

if use_lat:
    mr["region"] = mr["county_lat"].apply(assign_region_by_lat)
    print("      Regions assigned using latitude.")
else:
    mr["region"] = mr["fips"].astype(str).apply(assign_region_by_fips)
    print("      Regions assigned using fips-prefix fallback.")

region_counts = mr["region"].value_counts()
print("      Region distribution:")
for r in FOLD_ORDER:
    cnt = region_counts.get(r, 0)
    print(f"        {r}: {cnt:,} rows")

# ---------------------------------------------------------------------------
# 3. Identify feature columns
# ---------------------------------------------------------------------------
# Any column not in META_COLS and that is numeric
all_meta = META_COLS.copy()
# Also exclude latitude/longitude from features if present
for c in ["county_lat", "county_lon"]:
    all_meta.add(c)

feature_cols = [
    c for c in mr.columns
    if c not in all_meta and pd.api.types.is_numeric_dtype(mr[c])
]
print(f"\n      Feature columns ({len(feature_cols)}): {feature_cols}")

if not feature_cols:
    print("[ERROR] No feature columns found. Check model_ready.parquet schema.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 4. Spatial cross-validation
# ---------------------------------------------------------------------------
print("[4/4] Running spatial block cross-validation ...")

try:
    from xgboost import XGBRegressor
    xgb_available = True
except ImportError:
    print("      [WARN] xgboost not available; using sklearn GradientBoostingRegressor as fallback.")
    xgb_available = False
    try:
        from sklearn.ensemble import GradientBoostingRegressor
    except ImportError:
        print("[ERROR] Neither xgboost nor sklearn is available.")
        sys.exit(1)

# Fill NaN in features
mr[feature_cols] = mr[feature_cols].fillna(0.0)

fold_results = []

for holdout_region in FOLD_ORDER:
    print(f"\n  -- Fold: holdout={holdout_region} --")

    train_mask = mr["region"] != holdout_region
    test_mask  = mr["region"] == holdout_region

    X_train = mr.loc[train_mask, feature_cols].values
    y_train = mr.loc[train_mask, TARGET].values.astype(float)
    X_test  = mr.loc[test_mask,  feature_cols].values
    y_test  = mr.loc[test_mask,  TARGET].values.astype(float)

    n_train = len(X_train)
    n_test  = len(X_test)
    print(f"     Train: {n_train:,} rows | Test ({holdout_region}): {n_test:,} rows")

    if n_train == 0 or n_test == 0:
        print("     [WARN] Empty split -- skipping fold.")
        fold_results.append({
            "region": holdout_region,
            "n_train": n_train, "n_test": n_test,
            "rmse": float("nan"), "mae": float("nan"), "r2": float("nan"),
            "rmse_nonzero": float("nan"), "r2_nonzero": float("nan"),
            "precision": float("nan"), "recall": float("nan"), "f1": float("nan"),
        })
        continue

    # Train model
    if xgb_available:
        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
        )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    pred = np.clip(pred, 0.0, None)  # demand cannot be negative

    # Overall metrics
    metrics_all = compute_metrics(y_test, pred)
    print(f"     All rows  : RMSE={metrics_all['rmse']:.4f} | "
          f"MAE={metrics_all['mae']:.4f} | R2={metrics_all['r2']:.4f}")

    # Non-zero metrics
    nonzero_mask = y_test > 0
    if nonzero_mask.sum() > 5:
        metrics_nz = compute_metrics(y_test[nonzero_mask], pred[nonzero_mask])
        print(f"     Non-zero  : RMSE={metrics_nz['rmse']:.4f} | "
              f"MAE={metrics_nz['mae']:.4f} | R2={metrics_nz['r2']:.4f} "
              f"(n={metrics_nz['n']:,})")
    else:
        metrics_nz = {"rmse": float("nan"), "mae": float("nan"),
                      "r2": float("nan"), "n": 0}
        print("     Non-zero  : insufficient positive rows.")

    # Binary classification metrics
    bin_metrics = compute_binary_metrics(y_test, pred)
    print(f"     Binary (threshold={bin_metrics['threshold']:.4f}): "
          f"P={bin_metrics['precision']:.4f} | "
          f"R={bin_metrics['recall']:.4f} | "
          f"F1={bin_metrics['f1']:.4f}")

    fold_results.append({
        "region":       holdout_region,
        "n_train":      n_train,
        "n_test":       n_test,
        "rmse":         metrics_all["rmse"],
        "mae":          metrics_all["mae"],
        "r2":           metrics_all["r2"],
        "rmse_nonzero": metrics_nz["rmse"],
        "r2_nonzero":   metrics_nz["r2"],
        "precision":    bin_metrics["precision"],
        "recall":       bin_metrics["recall"],
        "f1":           bin_metrics["f1"],
    })

results_df = pd.DataFrame(fold_results).set_index("region").reindex(FOLD_ORDER)
print("\n  Spatial CV Results:")
print(results_df.to_string())

# ---------------------------------------------------------------------------
# 5. Build figure
# ---------------------------------------------------------------------------
print("\nBuilding spatial_cv.png ...")

fig, axes = plt.subplots(1, 3, figsize=(14, 8), facecolor=BG)
fig.suptitle(
    "Spatial Block Cross-Validation: Model Generalization by Geographic Region",
    fontsize=12, fontweight="bold", color=DARK, y=0.98
)

bar_colors = [FOLD_COLORS.get(r, SLATE) for r in FOLD_ORDER]

def style_ax(ax, title, ylabel):
    ax.set_title(title, fontsize=9, fontweight="bold", color=DARK, pad=6)
    ax.set_ylabel(ylabel, fontsize=8, color=MID)
    ax.set_facecolor(BG)
    ax.tick_params(axis="x", labelsize=8, colors=DARK)
    ax.tick_params(axis="y", labelsize=7, colors=MID)
    for spine in ax.spines.values():
        spine.set_edgecolor(LIGHT)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

# ---- Left: R2 by fold ----
ax = axes[0]
r2_vals = [results_df.loc[r, "r2"] if r in results_df.index else float("nan")
           for r in FOLD_ORDER]
r2_vals_safe = [v if not np.isnan(v) else 0.0 for v in r2_vals]

bars = ax.bar(FOLD_ORDER, r2_vals_safe, color=bar_colors,
              edgecolor=DARK, linewidth=0.6, width=0.6, zorder=2)

# Temporal reference line
ax.axhline(TEMPORAL_R2, color=DARK, linestyle="--", linewidth=1.4, zorder=3,
           label=f"Temporal holdout R2 = {TEMPORAL_R2:.4f}")
ax.legend(fontsize=7, loc="upper right")

style_ax(ax, "R2 by Spatial Fold", "R2")
ax.set_xticklabels(FOLD_ORDER, rotation=15, ha="right")

# Annotate bars
for bar, val in zip(bars, r2_vals):
    label = f"{val:.4f}" if not np.isnan(val) else "n/a"
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            label, ha="center", va="bottom", fontsize=7.5, color=DARK)

# ---- Middle: RMSE by fold ----
ax = axes[1]
rmse_vals = [results_df.loc[r, "rmse"] if r in results_df.index else float("nan")
             for r in FOLD_ORDER]
rmse_vals_safe = [v if not np.isnan(v) else 0.0 for v in rmse_vals]

bars = ax.bar(FOLD_ORDER, rmse_vals_safe, color=bar_colors,
              edgecolor=DARK, linewidth=0.6, width=0.6, zorder=2)

ax.axhline(TEMPORAL_RMSE, color=DARK, linestyle="--", linewidth=1.4, zorder=3,
           label=f"Temporal holdout RMSE = {TEMPORAL_RMSE:.3f}")
ax.legend(fontsize=7, loc="upper right")

style_ax(ax, "RMSE by Spatial Fold", "RMSE")
ax.set_xticklabels(FOLD_ORDER, rotation=15, ha="right")

for bar, val in zip(bars, rmse_vals):
    label = f"{val:.4f}" if not np.isnan(val) else "n/a"
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(rmse_vals_safe) * 0.01,
            label, ha="center", va="bottom", fontsize=7.5, color=DARK)

# ---- Right: Map-like scatter or fold-size bar ----
ax = axes[2]

has_coords = (
    use_lat
    and "county_lat" in mr.columns
    and "county_lon" in mr.columns
    and mr["county_lat"].notna().sum() > 10
)

if has_coords:
    # Scatter plot: lat vs lon, colored by region
    # Subsample for performance (up to 2000 per region)
    scatter_dfs = []
    for region in FOLD_ORDER:
        sub = mr[mr["region"] == region].drop_duplicates(subset=["fips"])
        if len(sub) > 2000:
            sub = sub.sample(2000, random_state=42)
        scatter_dfs.append(sub)
    scatter_df = pd.concat(scatter_dfs, ignore_index=True)

    for region in FOLD_ORDER:
        mask = scatter_df["region"] == region
        ax.scatter(
            scatter_df.loc[mask, "county_lon"],
            scatter_df.loc[mask, "county_lat"],
            c=FOLD_COLORS[region],
            s=8, alpha=0.6, linewidths=0, label=region, zorder=2
        )

    ax.set_xlabel("Longitude", fontsize=8, color=MID)
    ax.set_ylabel("Latitude", fontsize=8, color=MID)
    ax.set_title("County Locations by Fold\n(scatter: lon vs lat)",
                 fontsize=9, fontweight="bold", color=DARK, pad=6)
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(LIGHT)
    ax.tick_params(labelsize=7, colors=MID)

    # Add latitude band lines
    for lat_line, label_txt in [(30, "30 deg N"), (33, "33 deg N"), (36, "36 deg N")]:
        ax.axhline(lat_line, color=DARK, linestyle=":", linewidth=0.8, alpha=0.5)
        ax.text(-74, lat_line + 0.2, label_txt, fontsize=6.5, color=MID)

    ax.legend(fontsize=7, loc="lower right", markerscale=2)

else:
    # Fallback: bar chart of fold sizes
    fold_sizes = []
    for r in FOLD_ORDER:
        if r in results_df.index:
            n = results_df.loc[r, "n_test"]
            fold_sizes.append(int(n) if not np.isnan(n) else 0)
        else:
            fold_sizes.append(0)

    bars = ax.bar(FOLD_ORDER, fold_sizes, color=bar_colors,
                  edgecolor=DARK, linewidth=0.6, width=0.6, zorder=2)
    ax.set_title("Test-Set Size by Spatial Fold\n(lat/lon unavailable -- fips proxy used)",
                 fontsize=9, fontweight="bold", color=DARK, pad=6)
    ax.set_ylabel("Number of rows", fontsize=8, color=MID)
    ax.set_facecolor(BG)
    ax.tick_params(axis="x", labelsize=8, colors=DARK)
    ax.tick_params(axis="y", labelsize=7, colors=MID)
    for spine in ax.spines.values():
        spine.set_edgecolor(LIGHT)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xticklabels(FOLD_ORDER, rotation=15, ha="right")
    for bar, val in zip(bars, fold_sizes):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(fold_sizes) * 0.01,
                f"{val:,}", ha="center", va="bottom", fontsize=7.5, color=DARK)

    # Add legend patches for regions
    patches = [mpatches.Patch(color=FOLD_COLORS[r], label=r) for r in FOLD_ORDER]
    ax.legend(handles=patches, fontsize=7, loc="upper right")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"Figure saved -> {FIGURE_PATH}")

# ---------------------------------------------------------------------------
# 6. Print CV summary table
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("SPATIAL CV SUMMARY TABLE")
print("=" * 72)
header = (
    f"{'Region':<12} {'N_test':>8} {'RMSE':>8} {'MAE':>8} "
    f"{'R2':>9} {'RMSE_nz':>9} {'R2_nz':>9} {'F1':>7}"
)
print(header)
print("-" * 72)

for r in FOLD_ORDER:
    if r not in results_df.index:
        continue
    row   = results_df.loc[r]
    n_t   = int(row["n_test"]) if not np.isnan(row["n_test"]) else 0
    rmse  = f"{row['rmse']:.4f}"      if not np.isnan(row["rmse"])         else "  n/a"
    mae   = f"{row['mae']:.4f}"       if not np.isnan(row["mae"])          else "  n/a"
    r2    = f"{row['r2']:.4f}"        if not np.isnan(row["r2"])           else "  n/a"
    rmsenz= f"{row['rmse_nonzero']:.4f}" if not np.isnan(row["rmse_nonzero"]) else "  n/a"
    r2nz  = f"{row['r2_nonzero']:.4f}"  if not np.isnan(row["r2_nonzero"])   else "  n/a"
    f1    = f"{row['f1']:.4f}"        if not np.isnan(row["f1"])           else "  n/a"
    print(f"{r:<12} {n_t:>8,} {rmse:>8} {mae:>8} {r2:>9} {rmsenz:>9} {r2nz:>9} {f1:>7}")

print("-" * 72)
print(f"{'Temporal ref':12} {'--':>8} {TEMPORAL_RMSE:>8.4f} {'--':>8} "
      f"{TEMPORAL_R2:>9.4f} {'--':>9} {'~0.022':>9} {'--':>7}")
print("=" * 72)

# Summary comparison
print("\nCOMPARISON TO TEMPORAL HOLDOUT:")
valid_rmse = [v for v in rmse_vals if not np.isnan(v)]
valid_r2   = [v for v in r2_vals   if not np.isnan(v)]
if valid_rmse:
    avg_rmse = float(np.mean(valid_rmse))
    avg_r2   = float(np.mean(valid_r2))
    diff_rmse = avg_rmse - TEMPORAL_RMSE
    diff_r2   = avg_r2   - TEMPORAL_R2
    direction_rmse = "higher (worse)" if diff_rmse > 0 else "lower (better)"
    direction_r2   = "higher (better)" if diff_r2 > 0 else "lower (worse)"
    print(f"  Avg spatial RMSE: {avg_rmse:.4f}  vs temporal {TEMPORAL_RMSE:.4f}  "
          f"-> delta={diff_rmse:+.4f} ({direction_rmse})")
    print(f"  Avg spatial R2  : {avg_r2:.4f}  vs temporal {TEMPORAL_R2:.4f}  "
          f"-> delta={diff_r2:+.4f} ({direction_r2})")

print("\n[DONE] 15_spatial_cv.py completed.")
