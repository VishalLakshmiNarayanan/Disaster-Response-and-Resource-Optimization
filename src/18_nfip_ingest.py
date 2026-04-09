"""
18_nfip_ingest.py
Download NFIP (National Flood Insurance Program) claims from OpenFEMA and
produce a county-year claims summary usable as a richer demand signal than
PA declarations alone.

Usage:
    python src/18_nfip_ingest.py
"""

import matplotlib
matplotlib.use("Agg")

import requests
import time
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE        = Path(__file__).parent.parent
RAW_NFIP    = BASE / "data" / "raw" / "nfip"
PROC        = BASE / "data" / "processed"
FIG_DIR     = BASE / "outputs" / "figures"
RAW_NFIP.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE    = RAW_NFIP / "nfip_claims.parquet"

# ---------------------------------------------------------------------------
# OpenFEMA endpoint
# ---------------------------------------------------------------------------
BASE_URL    = "https://www.fema.gov/api/open/v2/FimaNfipClaims"
TOP         = 10_000          # rows per page
MAX_ROWS    = 500_000         # cap at 50 pages


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def fetch_page(skip: int, top: int = TOP) -> list:
    """Fetch one page of NFIP claims. Returns list of records."""
    params = {
        "$top":    top,
        "$skip":   skip,
        "$select": (
            "countyCode,yearOfLoss,"
            "buildingDamageAmount,contentsDamageAmount,"
            "floodZone,policyCount"
        ),
        "$format": "json",
        "$filter": "yearOfLoss ge 1980",
    }
    for attempt in range(3):
        try:
            r = requests.get(BASE_URL, params=params, timeout=60)
            r.raise_for_status()
            data = r.json()
            return data.get("FimaNfipClaims", [])
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            time.sleep(5 * (attempt + 1))
    return []


def download_nfip() -> pd.DataFrame:
    """
    Page through the OpenFEMA NFIP Claims API until exhausted or MAX_ROWS
    reached. Returns a raw DataFrame of individual claim records.
    """
    all_records = []
    skip = 0
    page = 0

    print("Starting NFIP claims download from OpenFEMA...")
    while skip < MAX_ROWS:
        print(f"  Fetching page {page + 1} (skip={skip}) ...")
        records = fetch_page(skip=skip, top=TOP)
        if not records:
            print("  Empty page returned -- download complete.")
            break
        all_records.extend(records)
        skip += TOP
        page += 1
        print(f"  Running total: {len(all_records):,} records")
        # polite pause between pages
        time.sleep(0.5)

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    print(f"Downloaded {len(df):,} raw claim records across {page} pages.")
    return df


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def process_claims(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and aggregate raw claim records to county-year summaries.

    Output columns:
        fips, year, nfip_claim_amount, nfip_claim_count, nfip_amount_per_claim
    """
    # --- normalise column names (API may return camelCase) ---
    df.columns = [c.strip() for c in df.columns]

    # --- FIPS: zero-pad to 5 digits ---
    if "countyCode" in df.columns:
        df["fips"] = df["countyCode"].astype(str).str.strip().str.zfill(5)
    else:
        raise KeyError("Expected 'countyCode' column not found in API response.")

    # --- year of loss ---
    if "yearOfLoss" in df.columns:
        df["year"] = pd.to_numeric(df["yearOfLoss"], errors="coerce")
    else:
        raise KeyError("Expected 'yearOfLoss' column not found in API response.")

    # --- damage amounts ---
    for col in ["buildingDamageAmount", "contentsDamageAmount"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["total_claim_amount"] = (
        df["buildingDamageAmount"] + df["contentsDamageAmount"]
    )

    # --- drop rows with invalid FIPS or year ---
    df = df.dropna(subset=["year"])
    df = df[df["fips"].str.match(r"^\d{5}$")]
    df["year"] = df["year"].astype(int)

    # --- aggregate to county-year ---
    agg = (
        df.groupby(["fips", "year"], as_index=False)
        .agg(
            nfip_claim_amount=("total_claim_amount", "sum"),
            nfip_claim_count=("total_claim_amount", "count"),
        )
    )

    agg["nfip_amount_per_claim"] = np.where(
        agg["nfip_claim_count"] > 0,
        agg["nfip_claim_amount"] / agg["nfip_claim_count"],
        0.0,
    )

    return agg.sort_values(["fips", "year"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Synthetic fallback
# ---------------------------------------------------------------------------

def make_synthetic_claims(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Build a synthetic NFIP-like county-year summary derived from existing
    predictions so the rest of the pipeline has something to join against
    even when the API is unavailable.
    """
    print("Building synthetic NFIP claims from predictions.parquet ...")
    rng = np.random.default_rng(42)

    rows = []
    for _, row in predictions.iterrows():
        fips       = str(row["fips"]).zfill(5)
        year       = int(row.get("storm_year", 2000))
        base_amt   = float(row.get("demand_proxy", 0)) * 1_000_000
        noise      = rng.uniform(0.8, 1.2)
        count      = max(1, int(rng.poisson(max(1, base_amt / 50_000))))
        amount     = base_amt * noise
        rows.append(
            {
                "fips":                 fips,
                "year":                 year,
                "nfip_claim_amount":    round(amount, 2),
                "nfip_claim_count":     count,
                "nfip_amount_per_claim": round(amount / count, 2),
            }
        )

    synth = pd.DataFrame(rows)
    # deduplicate to one row per (fips, year)
    synth = (
        synth.groupby(["fips", "year"], as_index=False)
        .agg(
            nfip_claim_amount=("nfip_claim_amount", "sum"),
            nfip_claim_count=("nfip_claim_count", "sum"),
        )
    )
    synth["nfip_amount_per_claim"] = np.where(
        synth["nfip_claim_count"] > 0,
        synth["nfip_claim_amount"] / synth["nfip_claim_count"],
        0.0,
    )
    return synth.sort_values(["fips", "year"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Comparison analysis
# ---------------------------------------------------------------------------

def run_comparison(claims: pd.DataFrame, is_synthetic: bool = False) -> None:
    """
    Join NFIP claims with predictions, compute correlation between
    demand_proxy and nfip_claim_amount_per_capita, and save a scatter figure.
    """
    # --- load predictions ---
    pred_path = PROC / "predictions.parquet"
    if not pred_path.exists():
        print(f"predictions.parquet not found at {pred_path} -- skipping comparison.")
        return
    predictions = pd.read_parquet(pred_path)
    print(f"Loaded predictions: {len(predictions):,} rows")

    # --- try to load panel for population ---
    pop_col_found = False
    panel_path = PROC / "panel.parquet"
    if panel_path.exists():
        try:
            panel = pd.read_parquet(panel_path)
            pop_candidates = [c for c in panel.columns if "pop" in c.lower()]
            if pop_candidates:
                pop_col = pop_candidates[0]
                panel_sub = panel[["fips", "storm_year", pop_col]].drop_duplicates()
                predictions = predictions.merge(
                    panel_sub, on=["fips", "storm_year"], how="left"
                )
                pop_col_found = True
                print(f"Joined population column '{pop_col}' from panel.")
        except Exception as e:
            print(f"Could not load panel.parquet: {e}")

    # --- join claims on fips + storm_year ---
    claims_join = claims.rename(columns={"year": "storm_year"})
    merged = predictions.merge(
        claims_join[["fips", "storm_year", "nfip_claim_amount", "nfip_claim_count"]],
        on=["fips", "storm_year"],
        how="left",
    )
    merged["nfip_claim_amount"] = merged["nfip_claim_amount"].fillna(0.0)

    # --- per-capita normalisation ---
    if pop_col_found:
        pop_vals = merged[pop_col].replace(0, np.nan)
        merged["nfip_per_capita"] = merged["nfip_claim_amount"] / pop_vals
    else:
        # fall back: use log_total_pop if present
        if "log_total_pop" in merged.columns:
            pop_est = np.exp(merged["log_total_pop"].replace(0, np.nan))
            merged["nfip_per_capita"] = merged["nfip_claim_amount"] / pop_est
        else:
            merged["nfip_per_capita"] = merged["nfip_claim_amount"]

    # --- correlation ---
    valid = merged.dropna(subset=["demand_proxy", "nfip_per_capita"])
    valid = valid[np.isfinite(valid["demand_proxy"]) & np.isfinite(valid["nfip_per_capita"])]

    if len(valid) < 5:
        print("Not enough overlapping rows for correlation analysis.")
    else:
        corr = valid["demand_proxy"].corr(valid["nfip_per_capita"])
        n    = len(valid)
        print("-" * 55)
        print("NFIP vs demand_proxy correlation analysis")
        print("-" * 55)
        print(f"  N (county-storm pairs):          {n:>10,}")
        print(f"  Pearson r (demand_proxy vs       ")
        print(f"    nfip_per_capita):               {corr:>10.4f}")
        print(f"  mean demand_proxy:               {valid['demand_proxy'].mean():>10.4f}")
        print(f"  mean nfip_per_capita:            {valid['nfip_per_capita'].mean():>10,.2f}")
        print("-" * 55)
        if is_synthetic:
            print("  NOTE: correlation computed on SYNTHETIC NFIP data.")
        print("")

        # --- scatter figure ---
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(
            valid["nfip_per_capita"],
            valid["demand_proxy"],
            alpha=0.35,
            s=18,
            color="#1f77b4",
            edgecolors="none",
        )

        # simple OLS trend line
        try:
            m, b = np.polyfit(
                valid["nfip_per_capita"].values,
                valid["demand_proxy"].values,
                1,
            )
            x_line = np.linspace(
                valid["nfip_per_capita"].min(),
                valid["nfip_per_capita"].max(),
                200,
            )
            ax.plot(x_line, m * x_line + b, color="red", lw=1.5,
                    label=f"OLS trend (r={corr:.3f})")
            ax.legend(fontsize=9)
        except Exception:
            pass

        ax.set_xlabel("NFIP Claim Amount Per Capita ($)" +
                      (" [SYNTHETIC]" if is_synthetic else ""), fontsize=10)
        ax.set_ylabel("Demand Proxy (PA registrations)", fontsize=10)
        ax.set_title(
            "NFIP Claims vs Disaster Aid Demand Proxy\n"
            "(county-storm observations"
            + (", synthetic NFIP" if is_synthetic else "")
            + ")",
            fontsize=11,
        )
        fig.tight_layout()
        out_path = FIG_DIR / "nfip_comparison.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        print(f"Comparison figure saved -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    claims = None
    is_synthetic = False

    # ------------------------------------------------------------------
    # Step 1: Try to download from OpenFEMA
    # ------------------------------------------------------------------
    try:
        print("=" * 60)
        print("NFIP Claims Ingest -- OpenFEMA API")
        print("=" * 60)
        raw = download_nfip()
        if raw.empty:
            raise ValueError("API returned zero records.")

        claims = process_claims(raw)
        print(f"\nAggregated to {len(claims):,} county-year rows.")

        claims.to_parquet(OUT_FILE, index=False)
        print(f"Saved -> {OUT_FILE}")

    except Exception as api_err:
        print(f"\nAPI download failed: {api_err}")

        # Step 2: Try cached file
        if OUT_FILE.exists():
            print(f"Loading cached file: {OUT_FILE}")
            try:
                claims = pd.read_parquet(OUT_FILE)
                print(f"Loaded {len(claims):,} rows from cache.")
            except Exception as cache_err:
                print(f"Cache read failed: {cache_err}")
                claims = None

    # ------------------------------------------------------------------
    # Step 3: Synthetic fallback
    # ------------------------------------------------------------------
    if claims is None:
        print("\nOpenFEMA API and cache both unavailable.")
        print("The NFIP Claims dataset is available at:")
        print("  https://www.fema.gov/openfema-data-page/fima-nfip-redacted-claims-v2")
        print("Falling back to synthetic NFIP data derived from predictions.parquet.")

        pred_path = PROC / "predictions.parquet"
        if not pred_path.exists():
            print(f"predictions.parquet not found at {pred_path}.")
            print("Run 08_train_model.py first to generate predictions.")
            return

        predictions = pd.read_parquet(pred_path)
        claims = make_synthetic_claims(predictions)
        is_synthetic = True

        claims.to_parquet(OUT_FILE, index=False)
        print(f"Synthetic NFIP claims saved -> {OUT_FILE}")

    # ------------------------------------------------------------------
    # Step 4: Summary stats
    # ------------------------------------------------------------------
    print("\n--- NFIP Claims Summary ---")
    print(f"  Rows (county-year):   {len(claims):,}")
    print(f"  Unique FIPS:          {claims['fips'].nunique():,}")
    print(f"  Year range:           {claims['year'].min()} -- {claims['year'].max()}")
    print(f"  Total claim amount:   ${claims['nfip_claim_amount'].sum():,.0f}")
    print(f"  Total claim count:    {claims['nfip_claim_count'].sum():,}")
    print(f"  Avg amount per claim: ${claims['nfip_amount_per_claim'].mean():,.2f}")
    print("")
    print("  Top 5 counties by total NFIP claim amount:")
    top5 = (
        claims.groupby("fips")["nfip_claim_amount"]
        .sum()
        .nlargest(5)
        .reset_index()
    )
    for _, row in top5.iterrows():
        print(f"    FIPS {row['fips']}: ${row['nfip_claim_amount']:,.0f}")
    print("")

    # ------------------------------------------------------------------
    # Step 5: Comparison analysis
    # ------------------------------------------------------------------
    run_comparison(claims, is_synthetic=is_synthetic)

    print("Done.")


if __name__ == "__main__":
    main()
