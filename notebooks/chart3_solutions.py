"""
Chart 3: Possible Approaches to Fix the Model
Shows three concrete solution paths with pros/cons and architecture diagrams.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path
import pandas as pd

OUT  = "../outputs/figures/chart3_solutions.png"
PROC = Path("../data/processed")

# ── Colours ───────────────────────────────────────────────────────────────────
C = {
    "approach1": "#1a5276",  # Two-part model  – dark blue
    "approach2": "#1e8449",  # FEMA-only       – dark green
    "approach3": "#7d3c98",  # Spatial CV      – purple
    "pro":       "#1e8449",
    "con":       "#c0392b",
    "neutral":   "#d4ac0d",
    "bg":        "#fdfefe",
    "dark":      "#1a1a1a",
    "light_1":   "#d4e6f1",  # light blue
    "light_2":   "#d5f5e3",  # light green
    "light_3":   "#e8daef",  # light purple
    "header":    "#2c3e50",
    "arrow":     "#555555",
}

fig = plt.figure(figsize=(22, 26), facecolor=C["bg"])
gs = gridspec.GridSpec(5, 3, figure=fig,
                       hspace=0.45, wspace=0.32,
                       top=0.94, bottom=0.03,
                       left=0.04, right=0.97)

# ─────────────────────────────────────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.97,
         "Possible Approaches: Fixing the Zero-Inflated Demand Model",
         ha="center", va="top", fontsize=17, fontweight="bold", color=C["dark"])
fig.text(0.5, 0.945,
         "Three concrete paths ranked by expected impact and implementation effort",
         ha="center", va="top", fontsize=11, color="#555555")

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def box(ax, x, y, w, h, title, body="", fc="#1a5276", tc="white",
        fs_title=9, fs_body=8, radius=0.25):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad=0,rounding_size={radius}",
                          facecolor=fc, edgecolor="white",
                          linewidth=1.3, zorder=3)
    ax.add_patch(rect)
    if body:
        ax.text(x+w/2, y+h*0.65, title, ha="center", va="center",
                color=tc, fontsize=fs_title, fontweight="bold", zorder=4)
        ax.text(x+w/2, y+h*0.27, body, ha="center", va="center",
                color=tc, fontsize=fs_body, zorder=4, linespacing=1.35)
    else:
        ax.text(x+w/2, y+h/2, title, ha="center", va="center",
                color=tc, fontsize=fs_title, fontweight="bold", zorder=4)

def arrow(ax, x1, y1, x2, y2, color="#555555", lw=2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=14), zorder=5)

def bullet(ax, x, y, items, color, ax_transform=True):
    for i, (icon, text) in enumerate(items):
        ax.text(x, y - i*0.072, icon, ha="left", va="top",
                color=color, fontsize=10, fontweight="bold",
                transform=ax.transAxes if ax_transform else None)
        ax.text(x + 0.03, y - i*0.072, text, ha="left", va="top",
                color=C["dark"], fontsize=8.5,
                transform=ax.transAxes if ax_transform else None)

# ═════════════════════════════════════════════════════════════════════════════
# ROW 0 — COMPARISON HEADER
# ═════════════════════════════════════════════════════════════════════════════
for col, (title, sub, fc) in enumerate([
    ("APPROACH 1  [RECOMMENDED]",
     "Two-Part Hurdle Model",   C["approach1"]),
    ("APPROACH 2",
     "Filter to FEMA-Declared + Log-Transform",  C["approach2"]),
    ("APPROACH 3",
     "Spatial Block CV + Ensemble",  C["approach3"]),
]):
    ax = fig.add_subplot(gs[0, col])
    ax.set_xlim(0, 10); ax.set_ylim(0, 2.5); ax.axis("off")
    rect = FancyBboxPatch((0, 0), 10, 2.5,
                          boxstyle="round,pad=0,rounding_size=0.3",
                          facecolor=fc, edgecolor="white", lw=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(5, 1.65, title, ha="center", va="center",
            color="white", fontsize=10, fontweight="bold", zorder=4)
    ax.text(5, 0.75, sub, ha="center", va="center",
            color="white", fontsize=9.5, style="italic", zorder=4)

# ═════════════════════════════════════════════════════════════════════════════
# ROW 1 — ARCHITECTURE DIAGRAMS
# ═════════════════════════════════════════════════════════════════════════════

# ── APPROACH 1: Two-Part Model Architecture ──────────────────────────────────
ax1 = fig.add_subplot(gs[1, 0])
ax1.set_xlim(0, 10); ax1.set_ylim(0, 6); ax1.axis("off")
ax1.set_title("Architecture: Two-Part (Hurdle) Model",
              fontsize=10, fontweight="bold", color=C["approach1"])

box(ax1, 1, 5.0, 8, 0.7,
    "Input Features (18 features per storm-county pair)",
    fc=C["header"])

# Stage 1
box(ax1, 0.3, 3.8, 4.2, 0.9,
    "Stage 1: Classifier",
    "XGBoost Classifier\ntarget: demand > 0 (yes/no)",
    fc=C["approach1"])
# Stage 2
box(ax1, 5.5, 3.8, 4.2, 0.9,
    "Stage 2: Regressor",
    "XGBoost Regressor\ntarget: log(demand) | demand > 0",
    fc="#154360")

arrow(ax1, 5, 5.0, 2.4, 3.8+0.9, color=C["approach1"])
arrow(ax1, 5, 5.0, 7.6, 3.8+0.9, color=C["approach1"])

# Outputs
box(ax1, 0.3, 2.5, 4.2, 0.8,
    "P(demand > 0)",
    "Probability county\nhas non-zero demand",
    fc=C["light_1"], tc=C["dark"])
box(ax1, 5.5, 2.5, 4.2, 0.8,
    "E[demand | demand > 0]",
    "Expected magnitude\n(log-space → exponentiated)",
    fc=C["light_1"], tc=C["dark"])

arrow(ax1, 2.4, 3.8, 2.4, 2.5+0.8)
arrow(ax1, 7.6, 3.8, 7.6, 2.5+0.8)

# Combine
box(ax1, 2.5, 1.4, 5.0, 0.8,
    "Combined Prediction",
    "pred = P(>0) * E[demand | >0]",
    fc=C["approach1"])
arrow(ax1, 2.4, 2.5, 4.7, 1.4+0.8, color="#aaaaaa")
arrow(ax1, 7.6, 2.5, 5.3, 1.4+0.8, color="#aaaaaa")

# Training note
box(ax1, 0.5, 0.2, 9.0, 1.0,
    "Training Notes",
    "Classifier trains on ALL 31K rows  |  Regressor trains on 6,492 non-zero rows only\n"
    "Handles zero-inflation by design  |  Log-transform reduces skew in regressor",
    fc="#eaf2ff", tc=C["dark"])

# ── APPROACH 2: FEMA-only + Log-transform ────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 1])
ax2.set_xlim(0, 10); ax2.set_ylim(0, 6); ax2.axis("off")
ax2.set_title("Architecture: Filtered Dataset + Log Transform",
              fontsize=10, fontweight="bold", color=C["approach2"])

box(ax2, 1, 5.0, 8, 0.7,
    "Full Panel (31,045 rows)", fc=C["header"])

box(ax2, 1, 3.9, 8, 0.8,
    "Filter: FEMA-declared counties only",
    "Keep only rows where disasterNumber is not null\n15,378 rows remain (49.5% of panel)",
    fc="#922b21")
arrow(ax2, 5, 5.0, 5, 3.9+0.8, color="#922b21")

box(ax2, 1, 2.8, 8, 0.8,
    "Label Transformation: log1p(demand_proxy)",
    "Compress heavy right tail  |  Forces model to\nlearn relative magnitudes, not absolute values",
    fc=C["approach2"])
arrow(ax2, 5, 3.9, 5, 2.8+0.8, color=C["approach2"])

box(ax2, 1, 1.7, 8, 0.8,
    "Single XGBoost Regressor",
    "Trained on 15,378 declared rows\nlog1p(demand) target  |  At prediction: exp(pred) - 1",
    fc="#1e5631")
arrow(ax2, 5, 2.8, 5, 1.7+0.8, color=C["approach2"])

box(ax2, 0.5, 0.2, 9.0, 1.0,
    "Key Insight",
    "Declared counties model what FEMA considers 'affected'\n"
    "Within that set, demand variation becomes learnable\n"
    "Simpler than hurdle model — fewer moving parts",
    fc="#d5f5e3", tc=C["dark"])

# ── APPROACH 3: Spatial Block CV ─────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 2])
ax3.set_xlim(0, 10); ax3.set_ylim(0, 6); ax3.axis("off")
ax3.set_title("Architecture: Spatial Block CV + Ensemble",
              fontsize=10, fontweight="bold", color=C["approach3"])

# Folds diagram
fold_labels = ["Gulf\n(TX,LA,MS,AL)", "Atlantic S\n(FL,GA,SC)", "Atlantic N\n(NC,VA,MD+)", "Northeast\n(NY,NJ,NE)", "PR + VI\n(Caribbean)"]
fold_colors = ["#6c3483","#1a5276","#1e8449","#b7770d","#922b21"]
for i, (fl, fc2) in enumerate(zip(fold_labels, fold_colors)):
    bx = 0.2 + i*1.9
    box(ax3, bx, 4.3, 1.7, 1.2, fl, fc=fc2, fs_title=7.5)

ax3.text(5, 4.05, "5-Fold Spatial Block Cross-Validation", ha="center",
         va="center", fontsize=8.5, color=C["dark"],
         style="italic")

box(ax3, 0.5, 2.9, 9, 0.9,
    "For each fold: train on 4 regions, validate on held-out region",
    "Measures: does model generalise to geographically new areas?\n"
    "Better estimate of real-world performance than temporal holdout alone",
    fc=C["approach3"])
arrow(ax3, 5, 4.3, 5, 2.9+0.9, color=C["approach3"])

box(ax3, 0.5, 1.9, 9, 0.75,
    "Ensemble: 5 fold-models + temporal holdout model = 6 total",
    "Average predictions across all 6 models\nReduces variance from any single model's fit",
    fc="#4a235a")
arrow(ax3, 5, 2.9, 5, 1.9+0.75, color=C["approach3"])

box(ax3, 0.5, 0.2, 9.0, 1.4,
    "Expected Benefit",
    "Spatial CV catches region-specific overfit\n"
    "Ensemble reduces prediction variance by ~20-30%\n"
    "Works alongside Approach 1 or 2 — orthogonal improvement\n"
    "Critical for proving generalizability in the report",
    fc=C["light_3"], tc=C["dark"])

# ═════════════════════════════════════════════════════════════════════════════
# ROW 2 — CODE SKETCH (what changes in the scripts)
# ═════════════════════════════════════════════════════════════════════════════

code_bg  = "#1e1e2e"
code_fg  = "#cdd6f4"
kw_color = "#89b4fa"   # blue – keywords
fn_color = "#a6e3a1"   # green – function names
cm_color = "#6c7086"   # grey – comments
str_col  = "#f38ba8"   # red – strings

def code_ax(ax, title, lines, colors):
    """Draw a simple syntax-highlighted code block."""
    ax.set_xlim(0, 10); ax.set_ylim(0, len(lines) + 0.8)
    ax.set_facecolor(code_bg)
    for spine in ax.spines.values():
        spine.set_color("#45475a")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set_title(title, fontsize=9, fontweight="bold", color=C["dark"])
    for i, (line, color) in enumerate(zip(lines, colors)):
        ax.text(0.2, len(lines) - i - 0.1, line,
                ha="left", va="top", fontsize=7.5,
                color=color, fontfamily="monospace")

ax_c1 = fig.add_subplot(gs[2, 0])
ax_c2 = fig.add_subplot(gs[2, 1])
ax_c3 = fig.add_subplot(gs[2, 2])

code_ax(ax_c1, "Approach 1 – Key Code Change (08_train_model.py)", [
    "# Stage 1: binary classifier",
    "clf = xgb.XGBClassifier(",
    "    n_estimators=500, max_depth=5,",
    "    scale_pos_weight=4.0,  # handle imbalance",
    "    eval_metric='logloss')",
    "y_binary = (y_train > 0).astype(int)",
    "clf.fit(X_train, y_binary, ...)",
    "",
    "# Stage 2: regressor on non-zero subset",
    "mask = y_train > 0",
    "reg = xgb.XGBRegressor(...)",
    "reg.fit(X_train[mask], np.log1p(y_train[mask]))",
    "",
    "# Combine at inference",
    "p_nonzero = clf.predict_proba(X_test)[:,1]",
    "log_demand = reg.predict(X_test)",
    "pred = p_nonzero * np.expm1(log_demand)",
], [cm_color, kw_color, code_fg, fn_color, code_fg, kw_color, fn_color, code_fg,
    cm_color, kw_color, kw_color, fn_color, code_fg, cm_color, fn_color, fn_color, kw_color])

code_ax(ax_c2, "Approach 2 – Key Code Change (06_build_panel.py + 08)", [
    "# In 06_build_panel.py:",
    "# Filter to declared counties only",
    "panel_declared = panel[",
    "    panel['fema_declared'] == 1].copy()",
    "",
    "# In 08_train_model.py:",
    "# Log-transform the target",
    "y_train_log = np.log1p(y_train)",
    "y_test_log  = np.log1p(y_test)",
    "",
    "model.fit(X_train, y_train_log, ...)",
    "",
    "# Back-transform predictions",
    "preds_log = model.predict(X_test)",
    "preds = np.expm1(preds_log)",
    "",
    "# Evaluate on original scale",
    "rmse = root_mean_squared_error(y_test, preds)",
], [cm_color, cm_color, kw_color, code_fg, code_fg,
    cm_color, cm_color, kw_color, kw_color, code_fg,
    fn_color, code_fg, cm_color, kw_color, kw_color, code_fg, cm_color, fn_color])

code_ax(ax_c3, "Approach 3 – Key Code Change (08_train_model.py)", [
    "# Spatial CV regions",
    "GULF     = ['22','28','01','48']  # LA,MS,AL,TX",
    "ATL_S    = ['12','13','45']       # FL,GA,SC",
    "ATL_N    = ['37','51','24','10']  # NC,VA,MD,DE",
    "NORTHE   = ['36','34','25','09']  # NY,NJ,MA,CT",
    "CARIB    = ['72','78']            # PR, VI",
    "",
    "folds = [GULF, ATL_S, ATL_N, NORTHE, CARIB]",
    "fold_models = []",
    "for hold_states in folds:",
    "    mask = ~df['fips'].str[:2].isin(hold_states)",
    "    m = xgb.XGBRegressor(...)",
    "    m.fit(X[mask], y[mask])",
    "    fold_models.append(m)",
    "",
    "# Ensemble average",
    "preds = np.mean([m.predict(X_test)",
    "                 for m in fold_models], axis=0)",
], [cm_color, str_col, str_col, str_col, str_col, str_col, code_fg,
    kw_color, kw_color, kw_color, kw_color, fn_color, kw_color, fn_color, kw_color,
    cm_color, kw_color, code_fg, code_fg])

# ═════════════════════════════════════════════════════════════════════════════
# ROW 3 — PROS / CONS
# ═════════════════════════════════════════════════════════════════════════════

pro_con_data = [
    # Approach 1
    ([("✚", "Theoretically correct for zero-inflated data"),
      ("✚", "SHAP interpretable per stage"),
      ("✚", "Classifier PR-AUC is meaningful"),
      ("✚", "Quantile models can run on regressor"),
      ("✚", "Best expected R² improvement")],
     [("✖", "Two models to tune and maintain"),
      ("✖", "Stage 2 trains on only 6.5K rows"),
      ("✖", "Combined metric harder to explain"),
      ("✖", "More implementation work (~3 hrs")]),
    # Approach 2
    ([("✚", "Simple — 1 model, 1 change"),
      ("✚", "Focuses on actionable cases only"),
      ("✚", "Log transform well-established"),
      ("✚", "Works with existing 08 script"),
      ("✚", "Clear framing: declared = affected")],
     [("✖", "Loses 50.5% of training data"),
      ("✖", "Can't predict for un-declared events"),
      ("✖", "Label still supply-driven, not need-driven"),
      ("✖", "Moderate improvement expected")]),
    # Approach 3
    ([("✚", "Tests generalization properly"),
      ("✚", "Required for rigorous capstone eval"),
      ("✚", "Ensemble reduces prediction variance"),
      ("✚", "Works with either Approach 1 or 2"),
      ("✚", "Directly addresses rubric criterion")],
     [("✖", "Doesn't fix zero-inflation by itself"),
      ("✖", "PR/VI folds may be small"),
      ("✖", "Adds ~1-2h of compute/code"),
      ("✖", "Must implement after fix Approach 1/2")]),
]

for col, (pros, cons) in enumerate(pro_con_data):
    approach_colors = [C["approach1"], C["approach2"], C["approach3"]]
    ac = approach_colors[col]

    ax_pro = fig.add_subplot(gs[3, col])
    ax_pro.set_xlim(0, 10); ax_pro.set_ylim(0, 5.5); ax_pro.axis("off")
    ax_pro.set_title(f"Approach {col+1}  —  Pros & Cons",
                     fontsize=10, fontweight="bold", color=ac)

    # Pros header
    rect = FancyBboxPatch((0, 3.5), 10, 0.6,
                          boxstyle="round,pad=0,rounding_size=0.15",
                          facecolor=C["pro"], edgecolor="white", lw=1, zorder=3)
    ax_pro.add_patch(rect)
    ax_pro.text(5, 3.8, "PROS", ha="center", va="center",
                color="white", fontsize=9, fontweight="bold", zorder=4)

    for i, (icon, text) in enumerate(pros):
        ax_pro.text(0.4, 3.3 - i*0.52, icon, ha="left", va="top",
                    color=C["pro"], fontsize=10, fontweight="bold")
        ax_pro.text(1.1, 3.3 - i*0.52, text, ha="left", va="top",
                    color=C["dark"], fontsize=8.3)

    # Cons header
    rect2 = FancyBboxPatch((0, -0.2), 10, 0.6,
                           boxstyle="round,pad=0,rounding_size=0.15",
                           facecolor=C["con"], edgecolor="white", lw=1, zorder=3)
    ax_pro.add_patch(rect2)
    ax_pro.text(5, 0.1, "CONS", ha="center", va="center",
                color="white", fontsize=9, fontweight="bold", zorder=4)

    for i, (icon, text) in enumerate(cons):
        ax_pro.text(0.4, -0.5 - i*0.52, icon, ha="left", va="top",
                    color=C["con"], fontsize=10, fontweight="bold")
        ax_pro.text(1.1, -0.5 - i*0.52, text, ha="left", va="top",
                    color=C["dark"], fontsize=8.3)

# ═════════════════════════════════════════════════════════════════════════════
# ROW 4 — RECOMMENDATION + EFFORT vs IMPACT MATRIX + EXECUTION ORDER
# ═════════════════════════════════════════════════════════════════════════════

# ── Effort vs Impact scatter ──────────────────────────────────────────────────
ax_ei = fig.add_subplot(gs[4, 0])
ax_ei.set_xlim(0, 10); ax_ei.set_ylim(0, 10)
ax_ei.set_xlabel("Implementation Effort (hours)", fontsize=9)
ax_ei.set_ylabel("Expected Model Impact", fontsize=9)
ax_ei.set_title("Effort vs. Impact Matrix", fontsize=10, fontweight="bold")
ax_ei.axvline(5, color="#cccccc", lw=1, linestyle="--")
ax_ei.axhline(5, color="#cccccc", lw=1, linestyle="--")

# Quadrant labels
for (tx, ty, txt) in [(2.5, 8.5, "HIGH IMPACT\nLOW EFFORT"),
                       (7.5, 8.5, "HIGH IMPACT\nHIGH EFFORT"),
                       (2.5, 1.5, "LOW IMPACT\nLOW EFFORT"),
                       (7.5, 1.5, "LOW IMPACT\nHIGH EFFORT")]:
    ax_ei.text(tx, ty, txt, ha="center", va="center",
               fontsize=7.5, color="#aaaaaa", style="italic")

approaches = [
    ("Approach 2\n(Filter+Log)",  3, 7, C["approach2"]),
    ("Approach 1\n(Two-Part)",    5, 9, C["approach1"]),
    ("Approach 3\n(Spatial CV)",  6, 6, C["approach3"]),
    ("Current Model\n(baseline)", 0, 1, "#e74c3c"),
]
for (lbl, ex, ey, ec) in approaches:
    ax_ei.scatter(ex, ey, s=220, color=ec, zorder=5, edgecolors="white", lw=1.5)
    ax_ei.text(ex + 0.3, ey + 0.3, lbl, fontsize=8, color=ec, fontweight="bold")

ax_ei.set_xticks([0,2,4,6,8,10])
ax_ei.set_xticklabels(["0h","2h","4h","6h","8h","10h"], fontsize=8)
ax_ei.set_yticks([0,2,4,6,8,10])
ax_ei.set_yticklabels(["None","Low","Mid","High","V.High","Max"], fontsize=8)
ax_ei.grid(True, alpha=0.2)

# ── Execution Order ───────────────────────────────────────────────────────────
ax_ex = fig.add_subplot(gs[4, 1])
ax_ex.set_xlim(0, 10); ax_ex.set_ylim(0, 7); ax_ex.axis("off")
ax_ex.set_title("Recommended Execution Order", fontsize=10, fontweight="bold")

steps = [
    ("Step 1  (Today, 2 hrs)",
     "Implement Approach 2 first\nFilter to declared counties + log1p target\nFast win, low risk",
     C["approach2"]),
    ("Step 2  (Tomorrow, 3 hrs)",
     "Upgrade to Approach 1 (Two-Part)\nAdd classifier → multiply by regressor\nVerify R² improvement > 0.10",
     C["approach1"]),
    ("Step 3  (Day 3, 2 hrs)",
     "Add Approach 3 on top (Spatial CV)\n5-region block holdout + ensemble average\nFinalise evaluation metrics",
     C["approach3"]),
    ("Step 4  (Day 4, 2 hrs)",
     "Re-run scripts 07-10 with new model\nRegenerating all figures + final report metrics\nVerify results table for proposal",
     C["header"]),
]
y_pos = 6.2
for i, (title, body, fc) in enumerate(steps):
    rect = FancyBboxPatch((0.2, y_pos - 1.3), 9.6, 1.1,
                          boxstyle="round,pad=0,rounding_size=0.2",
                          facecolor=fc, edgecolor="white", lw=1.2, zorder=3)
    ax_ex.add_patch(rect)
    ax_ex.text(5, y_pos - 0.65, title, ha="center", va="center",
               color="white", fontsize=8.5, fontweight="bold", zorder=4)
    ax_ex.text(5, y_pos - 1.05, body, ha="center", va="center",
               color="white", fontsize=7.8, zorder=4)
    if i < len(steps) - 1:
        ax_ex.annotate("", xy=(5, y_pos - 1.3),
                       xytext=(5, y_pos - 1.5),
                       arrowprops=dict(arrowstyle="-|>", color="#555555",
                                       lw=2, mutation_scale=12), zorder=5)
    y_pos -= 1.65

# ── Expected Metrics After Fix ────────────────────────────────────────────────
ax_mt = fig.add_subplot(gs[4, 2])
ax_mt.set_xlim(0, 10); ax_mt.set_ylim(0, 7); ax_mt.axis("off")
ax_mt.set_title("Expected Metrics After Fix", fontsize=10, fontweight="bold")

metric_rows = [
    ("Metric",              "Current",  "After Approach 2", "After Approach 1"),
    ("R² (test set)",       "-0.001",   "0.10 – 0.25",      "0.25 – 0.45"),
    ("RMSE improvement",    "0%",       "10 – 25%",         "25 – 45%"),
    ("PR-AUC (classifier)", "N/A",      "N/A",              "0.65 – 0.80"),
    ("Pred range",          "[0.07,1.6]","[0, 4.0]",        "[0, 5.0]"),
    ("Differentiation",     "Low",      "Medium",           "High"),
    ("Eq. gap (LP)",        "+52.1pp",  "+52.1pp",          "+55–60pp"),
]
row_colors_t = [C["header"],"#ecf0f1","#f8f9fa","#ecf0f1","#f8f9fa","#ecf0f1","#f8f9fa"]
col_widths_t  = [0.30, 0.16, 0.26, 0.26]
for ri, row in enumerate(metric_rows):
    for ci, (cell, cw) in enumerate(zip(row, col_widths_t)):
        xp = sum(col_widths_t[:ci]) + 0.01
        yp = 0.97 - ri * 0.135
        rect = FancyBboxPatch((xp, yp), cw - 0.01, 0.12,
                              boxstyle="round,pad=0,rounding_size=0.01",
                              facecolor=row_colors_t[ri],
                              edgecolor="white", lw=0.8,
                              transform=ax_mt.transAxes, zorder=3)
        ax_mt.add_patch(rect)
        fc_text = "white" if ri == 0 else (
            C["approach1"] if ci == 3 and ri > 0 else
            C["approach2"] if ci == 2 and ri > 0 else C["dark"])
        fw = "bold" if ri == 0 or ci == 0 else "normal"
        ax_mt.text(xp + (cw-0.01)/2, yp + 0.06, cell,
                   ha="center", va="center",
                   fontsize=7.3 if ri > 0 else 7.8,
                   fontweight=fw,
                   color=fc_text,
                   transform=ax_mt.transAxes, zorder=4)

plt.savefig(OUT, dpi=160, bbox_inches="tight",
            facecolor=C["bg"], edgecolor="none")
plt.close()
print(f"[OK] Saved {OUT}")
