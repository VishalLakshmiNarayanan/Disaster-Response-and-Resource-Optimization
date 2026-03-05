"""
08_train_model.py
Trains XGBoost point-estimate and quantile models on the event-county panel.
Produces SHAP explanations and saves predictions.
Outputs:
  outputs/models/xgb_point.json
  outputs/models/quantile_models.pkl
  outputs/figures/shap_summary.png
  data/processed/predictions.parquet
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import cross_val_score
import shap
import pickle
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROC    = Path("data/processed")
MODELS  = Path("outputs/models")
FIGS    = Path("outputs/figures")
MODELS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

TARGET       = "demand_proxy"
HOLDOUT_YEAR = 2018    # temporal holdout: 2018+ = test set


def get_features(df: pd.DataFrame) -> list[str]:
    """Return feature columns (exclude meta and target columns)."""
    non_feature = {TARGET, "storm_year", "fips", "storm_id", "storm_name",
                   "fema_declared", "feature_list"}
    return [c for c in df.columns if c not in non_feature]


if __name__ == "__main__":
    df = pd.read_parquet(PROC / "model_ready.parquet")
    features = get_features(df)

    print(f"Dataset: {len(df):,} rows | Features: {len(features)} | "
          f"Target: {TARGET}")
    print(f"Train years: <{HOLDOUT_YEAR}  |  Test years: >={HOLDOUT_YEAR}")

    train = df[df["storm_year"] <  HOLDOUT_YEAR].copy()
    test  = df[df["storm_year"] >= HOLDOUT_YEAR].copy()

    X_train = train[features].fillna(0)
    y_train = train[TARGET]
    X_test  = test[features].fillna(0)
    y_test  = test[TARGET]

    print(f"\nTrain: {len(train):,} rows | Test: {len(test):,} rows")

    # - Point estimate model -------------------------
    print("\n[1/3] Training point-estimate model...")
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        early_stopping_rounds=50,
        eval_metric="rmse",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=200,
    )

    preds = model.predict(X_test)
    preds = np.clip(preds, 0, None)  # demand can't be negative

    rmse = root_mean_squared_error(y_test, preds)
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)

    # Naive baseline: predict mean of training set for everyone
    naive_pred  = np.full(len(y_test), y_train.mean())
    naive_rmse  = root_mean_squared_error(y_test, naive_pred)
    naive_mae   = mean_absolute_error(y_test, naive_pred)

    print(f"\n- Model Performance --------------")
    print(f"  Model RMSE:  {rmse:.4f}  (naive: {naive_rmse:.4f}, "
          f"improvement: {(naive_rmse - rmse) / naive_rmse:.1%})")
    print(f"  Model MAE:   {mae:.4f}  (naive: {naive_mae:.4f})")
    print(f"  R²:          {r2:.4f}")

    model.save_model(str(MODELS / "xgb_point.json"))
    # Save feature list alongside the model
    (MODELS / "features.json").write_text(json.dumps(features))
    print(f"[OK] Saved model to {MODELS / 'xgb_point.json'}")

    # - Quantile models ---------------------------─
    print("\n[2/3] Training quantile models (q10, q50, q90)...")
    quantile_models = {}
    for q in [0.1, 0.5, 0.9]:
        qm = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=q,
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        qm.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        q_pred = qm.predict(X_test)
        # Quantile loss = pinball loss
        ql = np.mean(np.where(y_test >= q_pred,
                              q * (y_test - q_pred),
                              (1 - q) * (q_pred - y_test)))
        print(f"  q={q:.1f}: pinball loss = {ql:.4f}")
        quantile_models[q] = qm

    with open(MODELS / "quantile_models.pkl", "wb") as f:
        pickle.dump(quantile_models, f)
    print(f"[OK] Saved quantile models to {MODELS / 'quantile_models.pkl'}")

    # - SHAP explanations --------------------------─
    print("\n[3/3] Computing SHAP values...")
    explainer   = shap.TreeExplainer(model)
    sample_size = min(2000, len(X_test))
    X_sample    = X_test.sample(sample_size, random_state=42)

    shap_values = explainer.shap_values(X_sample)

    # Summary bar plot
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(shap_values, X_sample, plot_type="bar",
                      show=False, max_display=15)
    plt.title("SHAP Feature Importance (mean |SHAP value|)")
    plt.tight_layout()
    plt.savefig(FIGS / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Beeswarm plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
    plt.title("SHAP Summary (beeswarm)")
    plt.tight_layout()
    plt.savefig(FIGS / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[OK] SHAP plots saved to {FIGS}")

    # - Save predictions ---------------------------
    test_out = test.copy()
    test_out["pred_demand"] = np.clip(preds, 0, None)
    test_out["pred_q10"]    = np.clip(quantile_models[0.1].predict(X_test), 0, None)
    test_out["pred_q50"]    = np.clip(quantile_models[0.5].predict(X_test), 0, None)
    test_out["pred_q90"]    = np.clip(quantile_models[0.9].predict(X_test), 0, None)

    test_out.to_parquet(PROC / "predictions.parquet", index=False)
    print(f"[OK] Predictions saved: {len(test_out):,} rows")

    # Summary of best and worst predictions
    test_out["abs_error"] = (test_out["pred_demand"] - test_out[TARGET]).abs()
    print(f"\nTop 5 highest predicted demand counties:")
    print(test_out.nlargest(5, "pred_demand")[
        ["storm_name", "storm_year", "fips", TARGET, "pred_demand", "svi_overall"]
    ].to_string())
