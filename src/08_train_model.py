"""
08_train_model.py  (v2 - Two-Stage Hurdle Model)
=================================================
Fixes the zero-inflation problem by using a two-stage approach:

  Stage 1 -- XGBoost Classifier:  P(demand > 0 | hazard, vulnerability)
  Stage 2 -- XGBoost Regressor:   E[log1p(demand) | demand > 0, features]
  Combined: pred = P(>0) x expm1( E[log1p|>0] )

This dramatically improves on the naive single-model approach (which
collapsed to predicting the mean due to 79 % zero labels).

Also trains quantile models (q10 / q50 / q90) for uncertainty bounds.

Outputs
-------
  outputs/models/hurdle_clf.json        Stage-1 XGB classifier
  outputs/models/hurdle_reg.json        Stage-2 XGB regressor
  outputs/models/xgb_point.json         Alias of hurdle_reg (keeps downstream compat)
  outputs/models/quantile_models.pkl    q10/q50/q90 quantile models
  outputs/models/features.json          Feature list for inference
  outputs/figures/shap_bar.png
  outputs/figures/shap_beeswarm.png
  outputs/figures/hurdle_calibration.png
  data/processed/predictions.parquet
"""

import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    def root_mean_squared_error(y_true, y_pred):
        from sklearn.metrics import mean_squared_error
        import numpy as _np
        return _np.sqrt(mean_squared_error(y_true, y_pred))

# -- Paths ---------------------------------------------------------------------
PROC   = Path("data/processed")
MODELS = Path("outputs/models")
FIGS   = Path("outputs/figures")
MODELS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

TARGET       = "demand_proxy"
HOLDOUT_YEAR = 2018   # temporal holdout: train < 2018, test >= 2018

# -- Design tokens (matches evaluate_v2) ---------------------------------------
BG    = "#fafafa"
DARK  = "#0f172a"
MID   = "#64748b"
LIGHT = "#cbd5e1"
GRID  = "#e2e8f0"
WHITE = "#ffffff"
GREEN = "#10b981"

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
    "axes.labelcolor":   MID,
    "xtick.color":       MID,
    "ytick.color":       MID,
    "savefig.facecolor": BG,
    "savefig.dpi":       180,
    "savefig.bbox":      "tight",
})


def get_features(df: pd.DataFrame) -> list:
    """Return feature columns (exclude meta and target columns)."""
    non_feature = {
        TARGET, "storm_year", "fips", "storm_id",
        "storm_name", "fema_declared", "feature_list",
    }
    return [c for c in df.columns if c not in non_feature]


# ==============================================================================
if __name__ == "__main__":

    df = pd.read_parquet(PROC / "model_ready.parquet")
    features = get_features(df)

    n_zero    = (df[TARGET] == 0).sum()
    n_nonzero = (df[TARGET] >  0).sum()
    print(f"{'='*60}")
    print(f"HURDLE MODEL -- Two-Stage Demand Prediction")
    print(f"{'='*60}")
    print(f"Dataset : {len(df):,} rows | {len(features)} features | target: {TARGET}")
    print(f"Zeros   : {n_zero:,} ({n_zero/len(df):.1%}) -- handled by Stage-1 classifier")
    print(f"Non-zero: {n_nonzero:,} ({n_nonzero/len(df):.1%}) -- modelled by Stage-2 regressor")
    print(f"Holdout : train < {HOLDOUT_YEAR}  |  test >= {HOLDOUT_YEAR}")

    # -- Train / test split ----------------------------------------------------
    train = df[df["storm_year"] <  HOLDOUT_YEAR].copy()
    test  = df[df["storm_year"] >= HOLDOUT_YEAR].copy()

    X_train = train[features].fillna(0)
    y_train = train[TARGET]
    X_test  = test[features].fillna(0)
    y_test  = test[TARGET]

    # Binary labels for classifier
    y_train_bin = (y_train > 0).astype(int)
    y_test_bin  = (y_test  > 0).astype(int)

    # Non-zero subsets for Stage-2 regressor (log1p target)
    nz_mask_train  = y_train > 0
    X_train_nz     = X_train[nz_mask_train]
    y_train_log    = np.log1p(y_train[nz_mask_train])

    nz_mask_test   = y_test > 0
    X_test_nz      = X_test[nz_mask_test]
    y_test_log     = np.log1p(y_test[nz_mask_test])

    pos_weight = (y_train_bin == 0).sum() / max((y_train_bin == 1).sum(), 1)

    print(f"\nTrain: {len(train):,} rows | Non-zero: {nz_mask_train.sum():,} "
          f"({nz_mask_train.mean():.1%})")
    print(f"Test : {len(test):,}  rows | Non-zero: {nz_mask_test.sum():,} "
          f"({nz_mask_test.mean():.1%})")
    print(f"Class-weight ratio (neg/pos): {pos_weight:.1f}x\n")

    # ==========================================================================
    # STAGE 1 -- XGBoost Classifier: P(demand > 0)
    # ==========================================================================
    print("[1/4] Training Stage-1 Classifier  P(demand > 0) ...")

    clf = xgb.XGBClassifier(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=pos_weight,
        eval_metric="auc",
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(
        X_train, y_train_bin,
        eval_set=[(X_test, y_test_bin)],
        verbose=200,
    )

    clf_proba = clf.predict_proba(X_test)[:, 1]
    clf_pred  = (clf_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test_bin, clf_proba)
    ap  = average_precision_score(y_test_bin, clf_proba)

    print(f"\n  -- Stage-1 Results --------------------------")
    print(f"  ROC-AUC            = {auc:.4f}")
    print(f"  Average Precision  = {ap:.4f}")
    print(classification_report(
        y_test_bin, clf_pred,
        target_names=["Zero demand", "Non-zero demand"],
        digits=3,
    ))

    clf.save_model(str(MODELS / "hurdle_clf.json"))
    print(f"[OK] Classifier saved -> {MODELS/'hurdle_clf.json'}")

    # ==========================================================================
    # STAGE 2 -- XGBoost Regressor: E[log1p(demand) | demand > 0]
    # ==========================================================================
    print("\n[2/4] Training Stage-2 Regressor  E[log1p(demand) | demand > 0] ...")

    reg = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.04,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.05,
        reg_alpha=0.1,
        reg_lambda=1.0,
        early_stopping_rounds=50,
        eval_metric="rmse",
        random_state=42,
        n_jobs=-1,
    )
    reg.fit(
        X_train_nz, y_train_log,
        eval_set=[(X_test_nz, y_test_log)],
        verbose=200,
    )

    reg_pred_log = reg.predict(X_test_nz)
    naive_log    = np.full(len(y_test_log), y_train_log.mean())

    reg_r2    = r2_score(y_test_log, reg_pred_log)
    reg_rmse  = root_mean_squared_error(y_test_log, reg_pred_log)
    reg_mae   = mean_absolute_error(y_test_log, reg_pred_log)
    naive_r2  = r2_score(y_test_log, naive_log)
    naive_rmse = root_mean_squared_error(y_test_log, naive_log)

    print(f"\n  -- Stage-2 Results (non-zero rows, log1p scale) -")
    print(f"  R2   = {reg_r2:.4f}  (naive mean baseline: {naive_r2:.4f})")
    print(f"  RMSE = {reg_rmse:.4f}  (naive: {naive_rmse:.4f}, "
          f"improvement: {(naive_rmse - reg_rmse) / naive_rmse:.1%})")
    print(f"  MAE  = {reg_mae:.4f}")

    reg.save_model(str(MODELS / "hurdle_reg.json"))
    reg.save_model(str(MODELS / "xgb_point.json"))   # backward-compat alias
    print(f"[OK] Regressor saved -> {MODELS/'hurdle_reg.json'} (also -> xgb_point.json)")

    # ==========================================================================
    # COMBINED PREDICTION: P(>0) x expm1(E[log|>0])
    # ==========================================================================
    print("\n[3/4] Generating combined hurdle predictions ...")

    p_pos      = clf.predict_proba(X_test)[:, 1]          # P(demand > 0)
    pred_raw   = np.expm1(reg.predict(X_test))             # E[demand | >0] (original scale)
    preds      = np.clip(p_pos * pred_raw, 0, None)        # combined

    naive_all  = np.full(len(y_test), y_train.mean())
    rmse_h     = root_mean_squared_error(y_test, preds)
    mae_h      = mean_absolute_error(y_test, preds)
    r2_h       = r2_score(y_test, preds)
    naive_rmse_all = root_mean_squared_error(y_test, naive_all)

    print(f"\n  -- Combined Hurdle (all {len(y_test):,} test rows) ------")
    print(f"  RMSE = {rmse_h:.4f}  (naive: {naive_rmse_all:.4f}, "
          f"improvement: {(naive_rmse_all - rmse_h) / naive_rmse_all:.1%})")
    print(f"  MAE  = {mae_h:.4f}")
    print(f"  R2   = {r2_h:.4f}")

    (MODELS / "features.json").write_text(json.dumps(features))
    print(f"[OK] Feature list saved -> {MODELS/'features.json'}")

    # -- Quantile models (on non-zero rows, log1p scale) ----------------------
    print("\n  Training quantile models (q10, q50, q90) on non-zero rows ...")
    quantile_models = {}
    for q in [0.1, 0.5, 0.9]:
        qm = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=q,
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        qm.fit(X_train_nz, y_train_log,
               eval_set=[(X_test_nz, y_test_log)], verbose=False)
        q_pred = qm.predict(X_test_nz)
        ql = np.mean(np.where(
            y_test_log >= q_pred,
            q * (y_test_log - q_pred),
            (1 - q) * (q_pred - y_test_log),
        ))
        print(f"    q={q:.1f}: pinball loss = {ql:.4f}")
        quantile_models[q] = qm

    with open(MODELS / "quantile_models.pkl", "wb") as f:
        pickle.dump(quantile_models, f)
    print(f"[OK] Quantile models saved -> {MODELS/'quantile_models.pkl'}")

    # ==========================================================================
    # SHAP -- Stage-2 regressor (most interpretable: trained on real demand)
    # ==========================================================================
    print("\n[4/4] Computing SHAP values on Stage-2 regressor ...")

    explainer   = shap.TreeExplainer(reg)
    sample_size = min(2000, len(X_train_nz))
    X_sample    = X_train_nz.sample(sample_size, random_state=42)
    shap_values = explainer.shap_values(X_sample)

    # Bar chart
    fig, _ = plt.subplots(figsize=(9, 6))
    shap.summary_plot(shap_values, X_sample, plot_type="bar",
                      show=False, max_display=15)
    plt.title("SHAP Feature Importance -- Stage-2: Demand Magnitude\n"
              "(trained on counties with real FEMA demand)")
    plt.tight_layout()
    plt.savefig(FIGS / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Beeswarm
    fig, _ = plt.subplots(figsize=(11, 8))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
    plt.title("SHAP Beeswarm -- Stage-2: Demand Magnitude")
    plt.tight_layout()
    plt.savefig(FIGS / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] SHAP plots saved -> {FIGS}")

    # -- Calibration / performance figure -------------------------------------
    print("    Generating calibration figure ...")
    fig = plt.figure(figsize=(18, 5), facecolor=BG)
    fig.suptitle(
        "Hurdle Model Performance -- Stage-1 (Classifier) & Stage-2 (Regressor)",
        fontsize=14, fontweight="bold", color=DARK, y=1.02,
    )
    gs_cal = gridspec.GridSpec(1, 3, figure=fig, wspace=0.40)

    # (a) Score distribution by class
    ax0 = fig.add_subplot(gs_cal[0])
    ax0.hist(clf_proba[y_test_bin == 0], bins=40, alpha=0.65,
             color="#94a3b8", label="True zero demand", density=True)
    ax0.hist(clf_proba[y_test_bin == 1], bins=40, alpha=0.70,
             color=GREEN,     label="True non-zero demand", density=True)
    ax0.set_xlabel("Predicted P(demand > 0)")
    ax0.set_ylabel("Density")
    ax0.set_title(f"Stage-1 Score Distribution\nROC-AUC = {auc:.3f}  |  Avg Prec = {ap:.3f}")
    ax0.legend(fontsize=9)

    # (b) Actual vs predicted (log1p scale, non-zero rows)
    ax1 = fig.add_subplot(gs_cal[1])
    ax1.scatter(y_test_log.values, reg_pred_log,
                alpha=0.45, s=18, color=GREEN, linewidths=0)
    lo1 = min(y_test_log.min(), reg_pred_log.min())
    hi1 = max(y_test_log.max(), reg_pred_log.max())
    ax1.plot([lo1, hi1], [lo1, hi1], "--", color=LIGHT, linewidth=1.5)
    ax1.set_xlabel("Actual log1p(demand)")
    ax1.set_ylabel("Predicted log1p(demand)")
    ax1.set_title(
        f"Stage-2: Actual vs Predicted\n"
        f"R2 = {reg_r2:.3f}  RMSE = {reg_rmse:.3f}  (non-zero rows)"
    )

    # (c) Combined hurdle: actual vs predicted (original scale)
    ax2 = fig.add_subplot(gs_cal[2])
    ax2.scatter(y_test.values, preds,
                alpha=0.35, s=12, color=GREEN, linewidths=0)
    lo2 = min(float(y_test.min()), float(preds.min()))
    hi2 = max(float(y_test.max()), float(preds.max()))
    ax2.plot([lo2, hi2], [lo2, hi2], "--", color=LIGHT, linewidth=1.5)
    ax2.set_xlabel("Actual demand proxy")
    ax2.set_ylabel("Predicted demand proxy")
    ax2.set_title(
        f"Combined Hurdle: Actual vs Predicted\n"
        f"RMSE = {rmse_h:.3f}  R2 = {r2_h:.3f}  (all test rows)"
    )

    plt.tight_layout()
    plt.savefig(FIGS / "hurdle_calibration.png", dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[OK] Calibration figure saved -> {FIGS/'hurdle_calibration.png'}")

    # -- Save predictions ------------------------------------------------------
    test_out = test.copy()
    test_out["p_nonzero"]   = p_pos
    test_out["pred_demand"] = preds

    # Quantile predictions (combined: p_pos x expm1(q_pred))
    test_out["pred_q10"] = np.clip(
        p_pos * np.expm1(quantile_models[0.1].predict(X_test)), 0, None)
    test_out["pred_q50"] = np.clip(
        p_pos * np.expm1(quantile_models[0.5].predict(X_test)), 0, None)
    test_out["pred_q90"] = np.clip(
        p_pos * np.expm1(quantile_models[0.9].predict(X_test)), 0, None)
    test_out["abs_error"] = (test_out["pred_demand"] - test_out[TARGET]).abs()

    test_out.to_parquet(PROC / "predictions.parquet", index=False)
    print(f"[OK] Predictions saved -> {PROC/'predictions.parquet'}  ({len(test_out):,} rows)")

    # -- Final summary ---------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  HURDLE MODEL FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  Stage-1  Classifier  (P(demand > 0))")
    print(f"    ROC-AUC           : {auc:.4f}")
    print(f"    Average Precision : {ap:.4f}")
    print(f"  Stage-2  Regressor  (log demand | demand > 0)")
    print(f"    R2                : {reg_r2:.4f}  (baseline: {naive_r2:.4f})")
    print(f"    RMSE improvement  : {(naive_rmse - reg_rmse) / naive_rmse:.1%}")
    print(f"  Combined  Hurdle    (all test rows)")
    print(f"    RMSE improvement  : {(naive_rmse_all - rmse_h) / naive_rmse_all:.1%}")
    print(f"    R2                : {r2_h:.4f}")
    print(f"  Models saved to     : {MODELS.resolve()}")
    print(f"  Figures saved to    : {FIGS.resolve()}")
    print(f"{'='*60}")
    print(f"\n  Next step: python src/09_optimize.py")
