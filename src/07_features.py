"""
07_features.py
Selects, engineers, and validates features for the demand prediction model.
Output: data/processed/model_ready.parquet
"""
import pandas as pd
import numpy as np
from pathlib import Path

PROC = Path("data/processed")

FEATURE_COLS = [
    # Hazard
    "peak_wind_kt",
    "min_dist_km",
    # Population exposure
    "total_pop",
    "total_housing_units",
    # Vulnerability (SVI)
    "svi_overall",
    "EP_NOVEH",       # % no vehicle
    "EP_AGE65",       # % age 65+
    "EP_POV150",      # % below 150% poverty
    "EP_MOBILE",      # % mobile homes
    "EP_CROWD",       # % crowded housing
    "EP_DISABL",      # % disabled
    "EP_UNINSUR",     # % uninsured
    # ACS features
    "no_vehicle",
    "pop_65plus",
    "median_hh_income",
    "median_year_built",
]

TARGET = "demand_proxy"


if __name__ == "__main__":
    df = pd.read_parquet(PROC / "panel.parquet")

    available_features = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"NOTE: {len(missing)} feature(s) not found (will be skipped): {missing}")

    df_model = df[available_features + [TARGET, "storm_year", "fips", "storm_id",
                                         "storm_name", "fema_declared"]].copy()

    # Coerce all feature columns to numeric
    for col in available_features:
        df_model[col] = pd.to_numeric(df_model[col], errors="coerce")

    # Drop rows with missing target
    df_model = df_model.dropna(subset=[TARGET])

    # Log-transform population features to reduce skew
    for col in ["total_pop", "total_housing_units", "no_vehicle", "pop_65plus"]:
        if col in df_model.columns:
            df_model[f"log_{col}"] = np.log1p(df_model[col])

    # Flag: county was ever hit before (cumulative exposure proxy)
    df_model = df_model.sort_values(["fips", "storm_year"])
    df_model["prior_hits"] = df_model.groupby("fips").cumcount()

    # Normalize income (divide by 10k so scale matches other features)
    if "median_hh_income" in df_model.columns:
        df_model["median_hh_income_10k"] = df_model["median_hh_income"] / 10_000

    # Final feature list (add log and derived features)
    final_features = (
        available_features +
        [f"log_{c}" for c in ["total_pop","total_housing_units","no_vehicle","pop_65plus"]
         if f"log_{c}" in df_model.columns] +
        ["prior_hits", "median_hh_income_10k"]
    )
    final_features = [f for f in final_features if f in df_model.columns]

    # Drop duplicate base features that now have log versions
    for col in ["total_pop", "total_housing_units", "no_vehicle", "pop_65plus", "median_hh_income"]:
        if f"log_{col}" in final_features and col in final_features:
            final_features.remove(col)

    df_model["feature_list"] = str(final_features)  # store for reference

    df_out = df_model[final_features + [TARGET, "storm_year", "fips", "storm_id",
                                         "storm_name", "fema_declared"]].copy()
    df_out.attrs["features"] = final_features

    df_out.to_parquet(PROC / "model_ready.parquet", index=False)

    print(f"[OK] Model-ready dataset: {len(df_out):,} rows, {len(final_features)} features")
    print(f"\nFeatures used ({len(final_features)}):")
    for f in final_features:
        n_missing = df_out[f].isna().sum()
        print(f"  {f:<35} missing={n_missing:,}")

    print(f"\nTarget ({TARGET}) stats:")
    print(df_out[TARGET].describe().round(4))
    print(f"\nLabel > 0: {(df_out[TARGET] > 0).sum():,} / {len(df_out):,} "
          f"({(df_out[TARGET] > 0).mean():.1%})")
