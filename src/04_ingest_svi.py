"""
04_ingest_svi.py
Downloads the CDC 2020 Social Vulnerability Index (county level).
Output: data/raw/svi/svi_2020.parquet
"""
import requests
import pandas as pd
from pathlib import Path
import zipfile
import io

RAW = Path("data/raw/svi")
RAW.mkdir(parents=True, exist_ok=True)

# Direct CSV links — try in order until one works
CANDIDATE_URLS = [
    "https://svi.cdc.gov/Documents/Data/2020/csv/States_Counties/SVI_2020_US_county.csv",
    "https://svi.cdc.gov/Documents/Data/2022/csv/States_Counties/SVI_2022_US_county.csv",
]

KEEP_COLS = [
    "FIPS", "COUNTY", "STATE",
    "RPL_THEMES",                          # overall SVI rank (0–1)
    "RPL_THEME1", "RPL_THEME2", "RPL_THEME3", "RPL_THEME4",
    # Key component flags
    "EP_POV150", "EP_UNEMP", "EP_NOHSDP", "EP_UNINSUR",
    "EP_AGE65", "EP_AGE17", "EP_DISABL", "EP_SNGPNT",
    "EP_LIMENG", "EP_MINRTY",
    "EP_MUNIT", "EP_MOBILE", "EP_CROWD", "EP_NOVEH", "EP_GROUPQ",
    "E_TOTPOP",
]


if __name__ == "__main__":
    out_path = RAW / "svi_2020.parquet"
    if out_path.exists():
        print("svi_2020.parquet already exists, skipping.")
        df = pd.read_parquet(out_path)
        print(f"Loaded {len(df):,} counties")
        exit()

    df = None
    for url in CANDIDATE_URLS:
        print(f"Trying: {url}")
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            df = pd.read_csv(io.BytesIO(resp.content), encoding="latin-1", low_memory=False)
            print(f"  [OK] Downloaded {len(df):,} rows")
            break
        except Exception as e:
            print(f"  Failed: {e}")

    if df is None:
        print("All URLs failed. Check CDC SVI site manually.")
        exit(1)

    # Standardise FIPS to 5-char string
    df["fips"] = df["FIPS"].astype(str).str.zfill(5)

    # Keep only available columns
    available = [c for c in KEEP_COLS if c in df.columns]
    df = df[available + ["fips"]].copy()
    df = df.rename(columns={"RPL_THEMES": "svi_overall"})

    # -999 is CDC's nodata value — replace with NaN
    df = df.replace(-999, float("nan"))
    df = df.replace(-999.0, float("nan"))

    df.to_parquet(out_path, index=False)
    print(f"\n[OK] SVI saved: {len(df):,} counties, {len(available)} columns")
    print(df[["fips", "COUNTY", "STATE", "svi_overall"]].head(5).to_string())
